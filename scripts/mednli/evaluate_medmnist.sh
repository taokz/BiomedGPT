#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7087

user_dir=../../module
bpe_dir=../../utils/BPE

declare -a DatasetArray=("bloodmnist" "breastmnist" "dermamnist" "pathmnist" "pneumoniamnist")
data_dir=../../datasets/finetuning/MedMNIST

for dataset in ${DatasetArray[@]}; do
    data=${data_dir}/${dataset}_test.tsv
    ans2label_file=${data_dir}/${dataset}_class2label.pkl
    path=../../checkpoints/tuned_checkpoints/MedMNIST/base/${dataset}/25000_1000_5e-5_256/checkpoint_best.pt
    result_path=./results/medmnist/base/${dataset}
    mkdir -p $result_path
    selected_cols=0,2
    log_file=${result_path}/fine_tune.log

    CUDA_VISIBLE_DEVICES=9 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task=image_classify \
        --batch-size=32 \
        --log-format=simple --log-interval=100 \
        --seed=7 \
        --gen-subset=test \
        --results-path=${result_path} \
        --num-workers=0 \
        --fp16 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}" \
        > ${log_file} 2>&1

    echo "complete evaluation on " ${dataset} "dataset" 
done