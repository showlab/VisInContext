CUDA_VISIBLE_DEVICES="1,2" torchrun --nproc_per_node=2 src/main_pretrain.py --base_config src/config/train_local/base.yaml --deepspeed src/config/deepspeed/deepspeed_config_mistral.json