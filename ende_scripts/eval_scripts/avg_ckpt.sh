root=/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert
ckpt_path=$1

python $root/scripts/average_checkpoints.py --num-update-checkpoints 5 --inputs $ckpt_path --output $ckpt_path/checkpoint_avg5.pt
