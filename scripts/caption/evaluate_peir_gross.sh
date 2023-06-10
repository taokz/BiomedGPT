#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1091

user_dir=../../module
bpe_dir=../../utils/BPE

data=../../datasets/finetuning/peir_gross/peir_gross_test.tsv
path=../../checkpoints/tuned_checkpoints/peir_gross/tiny/stage1_checkpoints/100_0.06_600/checkpoint_best.pt

result_path=./results/peir_gross/tiny
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=5,6,7 python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=${MASTER_PORT} ../../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=10 \
    --max-len-b=32 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python metric_caption.py ${data} ${result_path}/test_predict.json 