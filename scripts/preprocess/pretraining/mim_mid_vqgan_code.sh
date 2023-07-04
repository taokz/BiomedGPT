#!/usr/bin/env

# for image-only data each line of the given input file should contain these information (separated by tabs):
# input format
#   image-id and image base64 string
# input example:
#   12455 /9j/4AAQSkZJ....UCP/2Q==
#
# output format
#   image-id and code
#   12455 6288 4495 4139...4691 4844 6464

# vqgan ckpt and yaml can be downloaded from https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/?p=%2Fconfigs&mode=list

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python generate_code.py \
  --file ./datasets/pretraining/chexpert_mid_image_string.tsv  \
  --outputs ./datasets/pretraining/chexpert_image_code.tsv \
  --selected_cols 0,1 \
  --code_image_size 128 \
  --vq_model vqgan \
  --vqgan_model_path ./checkpoints/vqgan_gumbel_f8_8192/ckpts/last.ckpt \
  --vqgan_config_path ./checkpoints/vqgan_gumbel_f8_8192/configs/model.yaml