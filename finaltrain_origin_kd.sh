#!/usr/bin/env bash
src=en
tgt=de
bedropout=0.5
ARCH=transformer_wmt_en_de_768
#DATAPATH=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bert-nmt/destdir-repro
DATAPATH=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data-bin/wmt14_en_de-multi_teacher_tiny
SAVEDIR=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt-checkpoint/kd_task/test/iwed_${src}_${tgt}_${bedropout}_test
mkdir -p $SAVEDIR
#if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
#then
#    cp /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt-checkpoint/checkpoints/iwed_${src}_${tgt}_${bedropout}_fairseq_base_long_768/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
#fi
#if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
#then
#warmup="--warmup-from-nmt --reset-lr-scheduler"
warmup="--reset-lr-scheduler"
#else
#warmup=""
#fi
#--reset-optimizer
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
LC_ALL=en_US.UTF-8 python /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/fairseq_bertnmt/train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt \
--no-epoch-checkpoints --save-interval-updates 5000 --reset-optimizer \
--dropout 0.3 --max-tokens 1000 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion distillation_loss --max-update 100000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR $warmup --masking --disable-validation \
--kd-alpha 0.9 --label-smoothing 0.1 --input-mapping --origin-kd --left-pad-source \
--update-freq 1 --bert-model-name /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bert-base-cased-new \
--bart-model-name /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bart-base
#|& tee -a $SAVEDIR/training.log
#--fp16
#--share-all-embeddings
# --left-pad-source False \