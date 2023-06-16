#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=2051
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export GPUS_PER_NODE=10

declare -a Scale=('tiny' 'medium' 'base')

for scale in ${Scale[@]}; do
  log_dir=./MeQSum_logs/${scale}
  save_dir=../../checkpoints/tuned_checkpoints/MeQSum/${scale}
  mkdir -p $log_dir $save_dir

  bpe_dir=../../utils/BPE
  user_dir=../../module

  data_dir=../../datasets/finetuning/MeQSum
  data=${data_dir}/train.tsv,${data_dir}/val.tsv
  restore_file=../../checkpoints/biomedgpt_${scale}.pt
  selected_cols=0,1

  task=gigaword
  arch=ofa_${scale}
  criterion=adjust_label_smoothed_cross_entropy
  label_smoothing=0.1
  lr=5e-5
  max_epoch=6
  warmup_ratio=0.06
  batch_size=8
  update_freq=4
  resnet_drop_path_rate=0.0
  encoder_drop_path_rate=0.1
  decoder_drop_path_rate=0.1
  dropout=0.1
  attention_dropout=0.0
  # max_src_length=512
  # max_tgt_length=64
  max_src_length=512
  max_tgt_length=128
  num_bins=1000

  for max_epoch in {300,}; do
    echo "max_epoch "${max_epoch}
    for lr in {1e-3,}; do
      echo "lr "${lr}
        for noise_ratio in {0.2,}; do
          echo "noise_ratio "${noise_ratio}

        log_file=${log_dir}/${max_epoch}"_"${lr}"_"${noise_ratio}".log"
        save_path=${save_dir}/${max_epoch}"_"${lr}"_"${noise_ratio}
        mkdir -p $save_path

        python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
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
            --log-format=simple --log-interval=10 \
            --fixed-validation-seed=7 \
            --no-epoch-checkpoints --keep-best-checkpoints=1 \
            --save-interval=1 --validate-interval=1 \
            --save-interval-updates=2500 --validate-interval-updates=2500 \
            --best-checkpoint-metric=rougeL_f1 --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --eval-rouge \
            --eval-print-samples \
            --eval-args='{"beam":10,"lenpen":0.7,"max_len_b":256,"no_repeat_ngram_size":3}' \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --noise-ratio=${noise_ratio} \
            --fp16 \
            --fp16-scale-window=512 \
            --num-workers=0 > ${log_file} 2>&1
      done
    done
  done
  echo "Complete fine-tuning biomedgpt_${scale} on MeQSum!"
done