#!/usr/bin/env

# Number of GPUs per GPU worker
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=127.0.0.1
# The port for communication
export MASTER_PORT=8314
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

data_dir=../../datasets/finetuning/vqa-rad
data=${data_dir}/train.tsv,${data_dir}/val.tsv
ans2label_file=${data_dir}/trainval_ans2label.pkl

declare -a Scale=('tiny' 'medium' 'base')

for scale in ${Scale[@]}; do
    restore_file=../../checkpoints/biomedgpt_${scale}.pt
    selected_cols=0,5,2,3,4

    log_dir=./vqa_rad_logs/${scale}
    save_dir=../../checkpoints/tuned_checkpoints/VQA-RAD/${scale}
    mkdir -p $log_dir $save_dir

    bpe_dir=../../utils/BPE
    user_dir=../../module

    task=vqa_gen
    arch=ofa_${scale}
    criterion=adjust_label_smoothed_cross_entropy
    label_smoothing=0.1
    batch_size=8
    if [[ $scale =~ "tiny" ]]; then
        batch_size=64
        patch_image_size=256
    elif [[ $scale =~ "medium" ]]; then
        batch_size=32
        patch_image_size=256
    elif [[ $scale =~ "base" ]]; then
        batch_size=16
        patch_image_size=384
    fi   
    update_freq=4
    resnet_drop_path_rate=0.0
    encoder_drop_path_rate=0.1
    decoder_drop_path_rate=0.1
    dropout=0.1
    attention_dropout=0.0
    max_src_length=80
    max_object_length=30
    max_tgt_length=40
    num_bins=1000

    uses_ema="--uses-ema"
    store_ema="--store-ema"
    ema_fp32="--ema-fp32"
    ema_decay=0.9999
    ema_start_update=0

    # Specify the inference type in validation after each fine-tuning epoch
    # val_inference_type=allcand
    val_inference_type=beamsearch
    # unconstrained_training_flag="--unconstrained-training"
    unconstrained_training_flag=""

    for max_epoch in {100,}; do
      echo "max_epoch "${max_epoch}
      for warmup_ratio in {0.04,}; do
        echo "warmup_updates "${warmup_updates}  
        for lr in {5e-5,}; do
          echo "lr "${lr}
          echo "patch_image_size "${patch_image_size}

          log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}"_"${unconstrained_training_flag}".log"
          save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}_"${unconstrained_training_flag}"
          mkdir -p $save_path

          CUDA_VISIBLE_DEVICES=0,1,2,4 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
              ${data} \
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
              --weight-decay=0.01 \
              --optimizer=adam \
              --adam-betas="(0.9,0.999)" \
              --adam-eps=1e-08 \
              --clip-norm=1.0 \
              --lr-scheduler=polynomial_decay \
              --lr=${lr} \
              --max-epoch=${max_epoch} \
              --warmup-ratio=${warmup_ratio} \
              --log-format=simple \
              --log-interval=10 \
              --fixed-validation-seed=7 \
              --keep-last-epochs=1 \
              --save-interval=1 --validate-interval=1 \
              --best-checkpoint-metric=vqa_score --maximize-best-checkpoint-metric \
              --max-src-length=${max_src_length} \
              --max-object-length=${max_object_length} \
              --max-tgt-length=${max_tgt_length} \
              --find-unused-parameters \
              --freeze-encoder-embedding \
              --freeze-decoder-embedding \
              ${unconstrained_training_flag} \
              --ans2label-file=${ans2label_file} \
              --valid-batch-size=20 \
              --add-type-embedding \
              --scale-attn \
              --scale-fc \
              --scale-heads \
              --disable-entangle \
              --num-bins=${num_bins} \
              --patch-image-size=${patch_image_size} \
              --prompt-type=prev_output \
              --fp16 \
              --fp16-scale-window=512 \
              --add-object \
              ${uses_ema} \
              ${store_ema} \
              ${ema_fp32} \
              --ema-decay=${ema_decay} \
              --ema-start-update=${ema_start_update} \
              --val-inference-type=${val_inference_type} \
              --num-workers=0 > ${log_file} 2>&1
        done
      done
    done
    echo "complete finetuning with pretrained " ${scale} "model" 
done