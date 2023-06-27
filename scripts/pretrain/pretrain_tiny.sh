#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1051
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export GPUS_PER_NODE=10

bpe_dir=../../utils/BPE
user_dir=../../module

restore_file=../../checkpoints/biomedgpt_tiny.pt

data_dir=../../datasets/pretraining
data=${data_dir}/vision_language.tsv
text_data=${data_dir}/text.tsv
image_data=${data_dir}/image.tsv
detection_data=${data_dir}/detection.tsv

selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,1,2
detection_selected_cols=0,1,2


task=unify_task
arch=ofa_tiny # specify the model scale
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=5e-5
max_epoch=20
warmup_ratio=0.01
batch_size=16
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=30
num_bins=1000
patch_image_size=256
sample_patch_num=196
max_image_size=512

#save_path=./checkpoints
save_path=../../checkpoints/name_by_yourself
log_file=pretrain_tiny.log

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
  $data \
  --text-data=${text_data} \
  --image-data=${image_data} \
  --detection-data=${detection_data} \
  --selected-cols=${selected_cols} \
  --text-selected-cols=${text_selected_cols} \
  --image-selected-cols=${image_selected_cols} \
  --detection-selected-cols=${detection_selected_cols} \
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
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=1 \
  --save-interval=10 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --fp16 \
  --fp16-scale-window=128 \
  --num-workers=0 > ${log_file} 2>&1