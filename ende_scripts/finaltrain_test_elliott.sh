#!/usr/bin/env bash
src=en
tgt=de
bedropout=0.5
ARCH=transformer_wmt_en_de
ROOT=/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert
#DATAPATH=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bert-nmt/destdir-repro
DATAPATH=$ROOT/data-bin/wmt14_en_de-multi_teacher_tiny
MODEL_SIG=bert_bart_test
SAVEDIR=$ROOT/checkpoints/${src}_${tgt}/$MODEL_SIG
mkdir -p $SAVEDIR

export CUDA_VISIBLE_DEVICES=0
LC_ALL=en_US.UTF-8 python $ROOT/fairseq_cli/train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt \
--no-epoch-checkpoints --save-interval-updates 5000 --reset-optimizer \
--dropout 0.1 --max-tokens 1000 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --label-smoothing 0.1 \
--fp16 \
--update-freq 1 \
--max-update 100000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--criterion new_mask_fill_distillation_loss \
--masking --denoising --disable-validation \
--use-bertinput --kd-alpha 0.9 --input-mapping --mask-lm --text-filling \
--bert-model-name $ROOT/pretrain_models/bert-base-cased-new \
--bart-model-name $ROOT/pretrain_models/bart-base
