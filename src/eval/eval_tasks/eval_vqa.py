import os
from src.eval.data.vqa_dataset import VQADataset
from src.eval.models import eval_base_model
from src.eval.eval_tasks.utils.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from src.eval.eval_tasks.utils.ok_vqa_utils import postprocess_ok_vqa_generation
from src.data.base.text_render import render_text_with_pil_multiple
import more_itertools
from src.eval.eval_tasks.util import *
from tqdm import tqdm
import json
import uuid

def evaluate_vqa(
    config: dict,
    eval_model: eval_base_model.BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    eval_prompt_style: str = "flamingo",
    dataset_name: str = "vqav2",
):
    """
    ...
    Args:
        config (dict): Configuration dictionary.
        ...
        dataset_name (string): Type of VQA dataset, currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: Accuracy score
    """
    if num_shots <= 8:
        batch_size = config['general']['batch_size']
    else:
        batch_size = 4
    num_samples = config['general']['num_samples']
    query_set_size = config['general']['query_set_size']


    few_shot_flag = True

    if eval_prompt_style == "obelics":
        print("----Using obelics prompt style for vqa task----")
        vqa_prefix_prompt = eval_model.obelics_vqa_prefix_prompt()
        vqa_prompt_method = eval_model.obelics_vqa_prompt
    elif eval_prompt_style == "llava":
        print("----Using llava prompt style for vqa task----")
        vqa_prefix_prompt = eval_model.llava_vqa_prefix_prompt()
        if dataset_name == "vizwiz":
            print("----Using vizwiz prompt style for vqa task----")
            vqa_prompt_method = eval_model.llava_vizwiz_vqa_prompt
        else:
            vqa_prompt_method = eval_model.llava_vqa_prompt
        few_shot_flag = False
        if num_shots > 0:
            print("LLAVA only supports 0-shot setting, please setting num_shots to 0")
            return 0.0
    else:
        print("----Using flamingo prompt style for vqa task----")
        vqa_prefix_prompt = eval_model.vqa_prefix_prompt()
        vqa_prompt_method = eval_model.vqa_prompt
    # Get dataset configuration
    dataset_config = config['datasets'].get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_image_dir_path = os.path.join(config['general']['data_root'], dataset_config['train_image_dir_path'])
    train_questions_json_path = os.path.join(config['general']['data_root'], dataset_config['train_questions_json_path'])
    train_annotations_json_path = os.path.join(config['general']['data_root'], dataset_config['train_annotations_json_path'])
    test_image_dir_path = os.path.join(config['general']['data_root'], dataset_config['test_image_dir_path'])
    test_questions_json_path = os.path.join(config['general']['data_root'], dataset_config['test_questions_json_path'])
    test_annotations_json_path = os.path.join(config['general']['data_root'], dataset_config['test_annotations_json_path'])


    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    test_dataset = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        seed,
    )

    if few_shot_flag:
        # effective_num_shots = num_shots if num_shots > 0 else 2
        effective_num_shots = num_shots if num_shots > 0 else num_shots
        # effective_num_shots = num_shots # previously I always set effective_num_shots = num_shots
        in_context_samples = get_query_set(train_dataset, query_set_size, seed)
    else:
        effective_num_shots = 0
        in_context_samples = []

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running vqa inference {dataset_name.upper()} shots={num_shots}"),
        batch_size,
    ):
        batch_rendered_text_images = []
        if few_shot_flag:
            batch_demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, effective_num_shots, len(batch)
            )
            for i in range(len(batch)):
                rendered_text = vqa_prefix_prompt + "".join(
                    [
                        vqa_prompt_method(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in batch_demo_samples[i]
                    ]
                )
                rendered_text = rendered_text.replace("<visual>", "")
                if rendered_text == "":
                    rendered_text = "Answer the question with given image."
                rendered_text_images = render_text_with_pil_multiple(rendered_text, n_parts=1)
                # sqve the rendered text images for debugging
                # rendered_text_images[0].save(f"output/rendered_text_{dataset_name}_shot_{num_shots}_{i}.png")
                batch_rendered_text_images.append(rendered_text_images)


        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            context_text = vqa_prefix_prompt
            batch_images.append([batch[i]["image"]])
            batch_text.append(
                context_text + vqa_prompt_method(question=batch[i]["question"])
            )

        with torch.no_grad():
            outputs = eval_model.get_outputs_w_text_image(
                batch_images=batch_images,
                rendered_text_image=batch_rendered_text_images,
                batch_text=batch_text,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        process_function = (
            postprocess_vqa_generation
            if dataset_name == "vqav2"
            else postprocess_ok_vqa_generation
        )

        new_predictions = map(process_function, outputs)
        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
        # print(batch_text[-1])
        # print(predictions[-1])
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{dataset_name}results_{random_uuid}.json",
        test_questions_json_path,
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{dataset_name}results_{random_uuid}.json")
    return acc
