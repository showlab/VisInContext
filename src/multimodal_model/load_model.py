from .language_model.load_language_model import load_language_model
from .vision_model.load_vision_model import load_vision_model
from utils.logo_util import format_num
import importlib
import yaml
# load different version of
with open('src/config/model_version/model_version.yaml') as f:
    config = yaml.safe_load(f)
model_version = config['model_architecture']['version']
model_module = importlib.import_module(f'multimodal_model.model_architecture{model_version}')
PPT = getattr(model_module, 'PPT')


def create_ppt(
    model_params: dict,
    lora_params: dict,
    instruction_tuning: bool = False,
    device: str = None,
    **ppt_kwargs,
):
    """
    Initialize a videogpt4 model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        vision_encoder: pretrained vision encoder
        lang_model: pretrained language decoder model
    Returns:
        model: videogpt4 model
    """
    vision_encoder_name = model_params['vision_encoder']['vision_encoder_name']
    vision_encoder_arch = model_params['vision_encoder']['vision_encoder_arch']
    vision_encoder_pretrained = model_params['vision_encoder']['vision_encoder_pretrained']
    vision_encoder_tuning = model_params['vision_encoder']['tuning']
    ckpt_path = model_params['vision_encoder']['ckpt_path']
    cache_dir = model_params['vision_encoder']['cache_dir']
    custom_augment = model_params['vision_encoder']['custom_augment']
    lang_model_path = model_params['lang_model']['lang_model_path']
    text_tokenizer_path = model_params['lang_model']['tokenizer_path']
    uni_modal_layers = model_params['lang_model']['unimodal_depth']
    interval_layer = model_params['lang_model']['interval_layer']
    use_memory_layer = model_params['lang_model']['use_memory_layer']
    dim_latents = model_params['multimodality_model']['latent_dim']
    contrastive_temperature = model_params['multimodality_model']['contrastive_temperature']
    cross_attention_compress_ratio = model_params['multimodality_model']['cross_attention_compress_ratio']
    contrastive_gather_way = model_params['multimodality_model']['contrastive_gather_way']
    only_attend_immediate_media = model_params['multimodality_model']['only_attend_immediate_media']
    qv_norm = model_params['multimodality_model']['qv_norm']

    # step1: first create a video encoder instance
    vision_encoder, image_processor, vis_dim = load_vision_model(vision_encoder_name=vision_encoder_name, 
                                                                vision_encoder_arch=vision_encoder_arch,
                                                                vision_encoder_pretrained=vision_encoder_pretrained,
                                                                ckpt_path=ckpt_path,
                                                                cache_dir=cache_dir,
                                                                custom_augment=custom_augment,
                                                                **ppt_kwargs
                                                                )
    # Ablation study, unfree sparseformer
    if vision_encoder_tuning:
        vision_encoder.requires_grad_(True)
    else:
        # Freeze Vision Encoder
        vision_encoder.requires_grad_(False)
    # step2: then create a language model instance
    lang_model, text_tokenizer, text_dim = load_language_model(vis_features_dim=vis_dim,
                                                     lang_model_path=lang_model_path, 
                                                     tokenizer_path=text_tokenizer_path, 
                                                     uni_modal_layers=uni_modal_layers,
                                                     interval_layer=interval_layer,
                                                     cross_attention_compress_ratio=cross_attention_compress_ratio,
                                                     use_memory_layer=use_memory_layer,
                                                     only_attend_immediate_media=only_attend_immediate_media,
                                                     qv_norm=qv_norm,
                                                     instruction_tuning=instruction_tuning,
                                                     device=device,
                                                     **ppt_kwargs
                                                     )
    # use lora to tune the language model, lora will auto freeze the language model besides lora parameters
    if lora_params is not None and lora_params["lora"]:
        from peft import (
            LoraConfig,
            get_peft_model,
        )
        config = LoraConfig(
            r=lora_params['lora_r'],
            lora_alpha=lora_params['lora_alpha'],
            target_modules=lora_params['lora_target_modules'],
            lora_dropout=lora_params['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        lang_model = get_peft_model(lang_model, config)
        # lang_model = lang_model.half()
        lang_model.print_trainable_parameters()
    else:
        lang_model.requires_grad_(False)
    model = PPT(
        vision_encoder,
        lang_model,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<visual>")[-1],
        vis_dim=vis_dim,
        text_dim=text_dim,
        uni_modal_layers=uni_modal_layers,
        dim_latents=dim_latents,
        contrastive_temperature=contrastive_temperature,
        vision_encoder_name=vision_encoder_name,
        contrastive_gather_way=contrastive_gather_way,
        qv_norm=qv_norm,
        **ppt_kwargs,
    )

    if instruction_tuning:
        print("Tuning lora parameters only")
        model.perceiver.requires_grad_(False)
        print("Unfreeze LM input embeddings")
        model.lang_model.get_input_embeddings().requires_grad_(True)
        # optionally unfreeze other layer
        if model.perceiver is not None:
            model.perceiver.requires_grad_(True)
            print("Unfreeze perceiver")
        print("Unfreeze gated_cross_attn_layers")
        model.lang_model.gated_cross_attn_layers.requires_grad_(True)
    else:
        # # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        if model.perceiver is not None:
            model.perceiver.requires_grad_(True)
            print("Unfreeze perceiver")
        if model.perceiver_text_image is not None:
            model.perceiver_text_image.requires_grad_(True)
            print("Unfreeze perceiver_text_image")
        print("Unfreeze gated_cross_attn_layers")
        model.lang_model.gated_cross_attn_layers.requires_grad_(True)
        # print("Unfreeze LM input embeddings")
        # model.lang_model.get_input_embeddings().requires_grad_(True)
        if model.lang_model.memory_layer is not None:
            print("Unfreeze memory layer")
            model.lang_model.memory_layer.requires_grad_(True)

    print("learnable parameters: ")
    for p in model.named_parameters():
        if p[1].requires_grad:
            print(p[0])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print('*'*80)
    print(f"ppt model initialized with {format_num(trainable_params)}/{format_num(all_params)} trainable parameters")
    print('*'*80)
    video_processor = None
    return model, image_processor, video_processor, text_tokenizer


def create_fuyu(model_params: dict, lora_params: dict, **fuyu_kwars):
    """
    Initalize fuyu model.
    """
    from transformers import AutoTokenizer, FuyuImageProcessor
    from .fuyu_model.processing_fuyu import FuyuProcessor
    from .fuyu_model.modeling_fuyu import FuyuForCausalLM

    lang_model_path = model_params['lang_model']['lang_model_path']
    text_tokenizer_path = model_params['lang_model']['tokenizer_path']

    processor = FuyuProcessor.from_pretrained(lang_model_path) # adept/fuyu-8b"
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path) # 'adept-hf-collab/fuyu-8b'
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = FuyuForCausalLM.from_pretrained(lang_model_path)
    model.requires_grad_ = False
    # Iterate through all named modules
    # for name, module in model.named_modules():
    #     print(name)  # Prints the name of each module
    # fuyu include 36 layers, from 0 to 35
    # unfreeze part of the model, we interlleave update the vision and language model
    # Make parameters in the 30th block learnable
    # Adjust the path according to your model's structure
    for param in model.language_model.model.layers[30].parameters():
        param.requires_grad = True

    video_processor = None
    return model, processor, video_processor, tokenizer