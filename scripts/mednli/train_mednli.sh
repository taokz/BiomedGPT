#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=3052

declare -a Scale=("base" "medium" "tiny")

for scale in ${Scale[@]}; do
  task=mnli
  log_dir=./mednli_logs/${scale}
  save_dir=../../checkpoints/tuned_checkpoints/mednli/${scale}
  mkdir -p $log_dir $save_dir

  bpe_dir=../../utils/BPE
  user_dir=../../module

  data_dir=../../datasets/finetuning/mednli

  data=${data_dir}/mednli_train.tsv,${data_dir}/mednli_val.tsv
  restore_file=../../checkpoints/biomedgpt_${scale}.pt
  selected_cols=0,1,2

  arch=ofa_${scale}
  criterion=adjust_label_smoothed_cross_entropy
  label_smoothing=0.0
  lr=3e-5
  max_epoch=5
  warmup_ratio=0.06
  # warmup_ratio=0.0
  batch_size=8
  if [[ $scale =~ "tiny" ]]; then
      batch_size=32
  elif [[ $scale =~ "medium" ]]; then
      batch_size=32
  elif [[ $scale =~ "base" ]]; then  
      batch_size=16
  fi
  update_freq=1
  resnet_drop_path_rate=0.0
  encoder_drop_path_rate=0.1
  decoder_drop_path_rate=0.1
  dropout=0.1
  attention_dropout=0.0
  max_src_length=512
  max_tgt_length=30
  num_bins=1000
  prompt_type="src"

  # default
  # for max_epoch in {5,7,10}; do
  #   echo "max_epoch "${max_epoch}
  #   for lr in {1e-4,7e-5,6e-5,5e-5,3e-5}; do
  #     echo "lr "${lr}
  #     for update_freq in {4,2}; do
  for max_epoch in {200,}; do
    echo "max_epoch "${max_epoch}
    for lr in {7e-5,}; do
      echo "lr "${lr}
      for update_freq in {2,}; do
        echo "update_freq "${update_freq}

        log_file=${log_dir}/${max_epoch}"_"${lr}"_"${update_freq}"_trick.log"
        save_path=${save_dir}/${max_epoch}"_"${lr}"_"${update_freq}"_trick"
        mkdir -p $save_path

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python3 -m torch.distributed.launch --nproc_per_node=9 --master_port=${MASTER_PORT} ../../train.py \
            $data \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --reset-optimizer --reset-dataloader --reset-meters \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --label-smoothing=${label_smoothing} \
            --batch-size=${batch_size} \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --resnet-drop-path-rate=${resnet_drop_path_rate} \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay --lr=${lr} \
            --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
            --log-format=simple --log-interval=50 \
            --fixed-validation-seed=7 \
            --keep-best-checkpoints=1 \
            --keep-last-epochs=1 \
            --save-interval=1 --validate-interval=1 \
            --best-checkpoint-metric=acc --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --prompt-type=${prompt_type} \
            --fp16 \
            --fp16-init-scale=16 \
            --fp16-scale-window=512 \
            --num-workers=0 > ${log_file} 2>&1
      done
    done
  done
done