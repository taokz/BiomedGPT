#!/usr/bin/env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# Number of GPUs per GPU worker
GPUS_PER_NODE=6
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=127.0.0.1
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

declare -a DatasetArray=("bloodmnist" "breastmnist" "dermamnist" "pathmnist" "pneumoniamnist")

data_dir=../../datasets/finetuning/MedMNIST
for dataset in ${DatasetArray[@]}; do
    data=${data_dir}/${dataset}_train.tsv,${data_dir}/${dataset}_val.tsv
    ans2label_file=${data_dir}/${dataset}_class2label.pkl
    restore_file=../../checkpoints/biomedgpt_base.pt
    selected_cols=0,2

    log_dir=./${dataset}_logs/base
    save_dir=../../checkpoints/tuned_checkpoints/MedMNIST/base/${dataset}
    mkdir -p $log_dir $save_dir

    bpe_dir=../../utils/BPE
    user_dir=../../module

    task=image_classify
    arch=ofa_base
    criterion=adjust_label_smoothed_cross_entropy
    label_smoothing=0.1
    batch_size=32
    update_freq=4
    resnet_drop_path_rate=0.0
    encoder_drop_path_rate=0.1
    decoder_drop_path_rate=0.1
    dropout=0.1
    attention_dropout=0.0
    max_src_length=128
    max_tgt_length=30
    num_bins=1000
    patch_image_size=256

    echo "finetune on " ${dataset} "dataset" 

    for total_num_updates in {25000,}; do
    echo "total_num_updates "${total_num_updates}
    for warmup_updates in {1000,}; do
        echo "warmup_updates "${warmup_updates}  
        for lr in {5e-5,}; do
        echo "lr "${lr}
        for patch_image_size in {256,}; do
            echo "patch_image_size "${patch_image_size}

            log_file=${log_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
            save_path=${save_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}"_"${patch_image_size}
            mkdir -p $save_path

            python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
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
                --total-num-update=${total_num_updates} \
                --warmup-updates=${warmup_updates} \
                --log-format=simple \
                --log-interval=10 \
                --fixed-validation-seed=9 \
                --keep-last-epochs=1 \
                --save-interval=1 --validate-interval=1 \
                --max-update=${total_num_updates} \
                --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
                --max-src-length=${max_src_length} \
                --max-tgt-length=${max_tgt_length} \
                --find-unused-parameters \
                --freeze-encoder-embedding \
                --freeze-decoder-embedding \
                --ans2label-file=${ans2label_file} \
                --valid-batch-size=20 \
                --add-type-embedding \
                --scale-attn \
                --scale-fc \
                --scale-heads \
                --disable-entangle \
                --num-bins=${num_bins} \
                --patch-image-size=${patch_image_size} \
                --fp16 \
                --fp16-scale-window=512 \
                --imagenet-default-mean-and-std \
                --num-workers=0 > ${log_file} 2>&1
        done
        done
    done
    done
    echo "complete finetuning on " ${dataset} "dataset" 
done