#!/usr/bin/env bash
src=en
tgt=de
bedropout=0.5
ARCH=transformer_wmt_en_de
ROOT=/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert

#### MODIFY ######
KD_ALPHA=0.75
DATA_SIG=wmt14_en_de-bert-or-bart
MODEL_SIG=d512_bert_mask_lm_kd_alpha_autobert_1_${KD_ALPHA}
#### MODIFY ######

DATAPATH=$ROOT/data-bin/$DATA_SIG
SAVEDIR=$ROOT/checkpoints/$DATA_SIG/$MODEL_SIG
mkdir -p $SAVEDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=1

LC_ALL=en_US.UTF-8 python $ROOT/fairseq_cli/train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0007 -s $src -t $tgt \
--no-epoch-checkpoints --save-interval-updates 5000 \
--dropout 0.1 --max-tokens 4000 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --label-smoothing 0.1 \
--log-interval 100 --disable-validation \
--update-freq 1 --ddp-backend=no_c10d \
--max-update 200000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--criterion nomaskloss_mask_distillation_loss \
--masking --mask-lm --use-bertinput \
--fp16 --left-pad-source --bert-auto-bertencoder 1 \
--kd-alpha $KD_ALPHA \
--bert-model-name $ROOT/pretrain_models/bert-base-cased-new
# --use-bertinput

# --share-all-embeddings
# --input-mapping 
# --text-filling
# --bart-model-name $ROOT/pretrain_models/bart-base
# --denoising 
