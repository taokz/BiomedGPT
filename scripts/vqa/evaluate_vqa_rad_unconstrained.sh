#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

user_dir=../../module
bpe_dir=../../utils/BPE

# val or test
split=$1

data_dir=../../datasets/finetuning/VQA-RAD
data=${data_dir}/test.tsv

declare -a Scale=('tiny' 'medium' 'base')

for scale in ${Scale[@]}; do
    if [[ $scale =~ "tiny" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "medium" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "base" ]]; then  
        patch_image_size=384
    fi

    path=../../checkpoints/instruct_biomedgpt_${scale}.pt
    result_path=./results/vqa_rad_unconstrained/${scale}
    mkdir -p $result_path
    selected_cols=0,5,2,3,4

    log_file=${result_path}/${scale}".log"

    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task=vqa_gen \
        --batch-size=64 \
        --log-format=simple --log-interval=10 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --fp16 \
        --beam-search-vqa-eval \
        --ema-eval \
        --beam=${beam_size} \
        --unnormalized \
        --temperature=1.0 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
        > ${log_file} 2>&1
done