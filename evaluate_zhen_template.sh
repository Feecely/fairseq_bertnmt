root=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt
export PYTHONPATH=$root:$PYTHONPATH
data_signature=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data-bin/wmt14_en_de-multi_teacher_tiny
signature=$1


result_dir=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt-checkpoint/result/
mkdir -p $result_dir/test.out

echo $result_dir

for ckpt in /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt-checkpoint/kd_task/test/iwed_en_de_0.5_test/checkpoint_last.pt; do
	echo "decoding $ckpt"
	export CUDA_VISIBLE_DEVICES=3
	LC_ALL=en_US.UTF-8 python /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/fairseq_bertnmt/generate.py \
	    $data_signature \
	    --results-path $result_dir/test.out \
	    --gen-subset test \
	    --path $ckpt \
	    --lenpen 0.6 --left-pad-source --use-bertinput --masking \
	    --bert-model-name /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/data/bert-base-cased-new \
	    --batch-size 128 -s en -t de \
	    --beam 5 --remove-bpe |& tee  $result_dir/test.out/generate-test.txt

	LC_ALL=en_US.UTF-8 python $root/extract_generate_output.py \
	    --output $result_dir/test.out/generate-test --srclang de --tgtlang en $result_dir/test.out/generate-test.txt
  LC_ALL=en_US.UTF-8 python $root/strip_file.py $result_dir/test.out/generate-test.de
  echo "finish generate"
#	cd $data_dir
  ckpt_base="$(basename -- $ckpt)"
  cd $root
	bash /mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang/bert-nmt/eval_test14.sh $result_dir/test.out/generate-test.de.strip #> $result_dir/${ckpt_base}.eval
done



