# interleaved sequence retrieval

import os
from tqdm import tqdm
import json
import more_itertools
import uuid
from collections import defaultdict
import torch
from src.eval.models import eval_base_model
from src.eval.data.seq_retrieval_dataset import SeqRetrievalDataset
from src.eval.eval_tasks.util import prepare_eval_samples, get_query_set, sample_batch_demos_from_query_set
from src.eval.eval_tasks.utils.retrieval_metric import t2v_metrics, v2t_metrics, sim_matrix


def evaluate_seq_retrieval(
    config: dict,
    eval_model: eval_base_model.BaseEvalModel,
    seed: int = 42,
    metric_fns: list = [t2v_metrics, v2t_metrics],
    dataset_name: str = "obelics",
    num_shots: int = 8,
    eval_prompt_style: str = "flamingo",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (eval_model.BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """
    num_samples = config['general']['num_samples']
    batch_size = 4
    if dataset_name == "obelics":
        json_dir_path = os.path.join(config['general']['data_root'], config['datasets']['obelics']['json_dir_path'])
        test_annotations_path = os.path.join(json_dir_path, "annotations.json")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for retrieval task")


    # for retrieval task, we only need to evaluate on test set and no query set is needed
    test_dataset =  SeqRetrievalDataset(
        json_dir_path=json_dir_path,
        annotation_json_file=test_annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    test_dataset = prepare_eval_samples(
        test_dataset,
        num_samples if num_samples > 0 else len(test_dataset),
        seed,
    )

    text_embed_list = []
    visual_embed_list = []
    seq_embed_list = []

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running retrieval inference {dataset_name.upper()}"),
        batch_size,
    ):
        batch_seqs = []
        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            batch_seqs.append([batch[i]["image_1"], batch[i]["text_1"]])
            batch_images.append([batch[i]["image_2"]])
            batch_text.append(batch[i]["text_2"])
        seq_embeds = eval_model.get_seq_embeddings(
            batch_seqs=batch_seqs
        )
        text_embeds = eval_model.encode_text(
            batch_text=batch_text
        )
        image_embeds = eval_model.encode_image(
            batch_images=batch_images
        )
        text_embed_list.append(text_embeds.float().cpu().detach())
        visual_embed_list.append(image_embeds.float().cpu().detach())
        seq_embed_list.append(seq_embeds.float().cpu().detach())

    text_embeds_mat = torch.cat(text_embed_list, dim=0)
    visual_embeds_mat = torch.cat(visual_embed_list, dim=0)
    seq_embed_list = torch.cat(seq_embed_list, dim=0)

    sims = sim_matrix(visual_embeds_mat, text_embeds_mat).numpy()
    nested_metrics = {}
    # print(metric_fns)
    for metric in metric_fns:
        metric_name = metric.__name__
        res = metric(sims)
        nested_metrics[metric_name] = res

    sims_seq_to_image = sim_matrix(seq_embed_list, visual_embeds_mat).numpy()
    nested_metrics["seq_to_image"] = t2v_metrics(sims_seq_to_image)
    sims_seq_to_text = sim_matrix(seq_embed_list, text_embeds_mat).numpy()
    nested_metrics["seq_to_text"] = v2t_metrics(sims_seq_to_text)
    return nested_metrics