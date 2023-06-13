#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=2081
export CUDA_VISIBLE_DEVICES=9
export GPUS_PER_NODE=1

user_dir=../../module
bpe_dir=../../utils/BPE

data=../../datasets/finetuning/MeQSum/test.tsv

declare -a Scale=('tiny' 'medium' 'base')
for scale in ${Scale[@]}; do
    path=../../checkpoints/tuned_checkpoints/MeQSum/${scale}/300_1e-3_0.2/checkpoint_best.pt
    result_path=./results/MeQSum/${scale}
    mkdir -p $result_path
    selected_cols=0,1
    split='test'

    python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task=gigaword \
        --batch-size=32 \
        --log-format=simple --log-interval=10 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --beam=5 \
        --lenpen=0.7 \
        --max-len-b=128 \
        --no-repeat-ngram-size=3 \
        --fp16 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

    python3 eval_rouge.py ${result_path}/test_predict.json
    echo "Complete evaluating biomedgpt_${scale} on MeQSum!"
done
