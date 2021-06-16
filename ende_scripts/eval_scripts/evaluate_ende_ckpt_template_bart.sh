root=/apdcephfs/share_47076/elliottyan/co-work-projects/fairseq-bert
export PYTHONPATH=$root:$PYTHONPATH

if [ -z "$2" ]; then
	data_signature=wmt14_en_de
else
	data_signature=$2
fi
echo $data_signature
ori_data_signature=wmt14_en_de

#signature=$1
#output_dir=$root/checkpoints/$data_signature/$signature
#ckpt=$output_dir/checkpoint_avg5.pt
ckpt=$1

dirname=$(dirname $ckpt)
signature=$(basename $ckpt)

data_dir=$root/data/$ori_data_signature

result_dir=$root/results/$data_signature/$signature
mkdir -p $result_dir

echo "decoding $ckpt" 
#python $root/fairseq_cli/generate.py \
export CUDA_VISIBLE_DEVICES=1
python $root/generate.py \
    $root/data-bin/$data_signature \
    --results-path $result_dir/test.out \
    --path $ckpt \
    --lenpen 0.6 --left-pad-source --use-bartinput --denoising \
    --bart-model-name $root/pretrain_models/bart-base \
    --beam 4 --remove-bpe

echo "$result_dir"
LC_ALL=en_US.UTF-8 python $root/scripts/extract_generate_output.py \
    --output $result_dir/test.out/generate-test --srclang de --tgtlang en $result_dir/test.out/generate-test.txt

cd $data_dir
ckpt_base="$(basename -- $ckpt)"
bash eval_test14.sh $result_dir/test.out/generate-test.de > $result_dir/eval.result.${ckpt_base}

cat $result_dir/eval.result.${ckpt_base}

