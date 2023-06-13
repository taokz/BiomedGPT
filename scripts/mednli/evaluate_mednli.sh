#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7087
export CUDA_VISIBLE_DEVICES=6
export GPUS_PER_NODE=1

user_dir=../../module
bpe_dir=../../utils/BPE

declare -a Scale=("base" "medium" "tiny")

for scale in ${Scale[@]}; do
data=../../datasets/finetuning/mednli/mednli_test.tsv
path=../../checkpoints/tuned_checkpoints/mednli/${scale}/200_7e-5_2/checkpoint_best.pt
result_path=./results/mednli/${scale}
mkdir -p $result_path
selected_cols=0,1,2
split='test'
log_file=${result_path}/${scale}.log

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=mnli \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
    > ${log_file} 2>&1
done