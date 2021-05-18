ROOT=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_xyvhuang
DATA_SIG=raw_data/wmt14_en_de-head1w
# DATA_SIG=wmt14_en_de
DATA_DIR=$ROOT/data/$DATA_SIG
OUT_DATA_SIG=wmt14_en_de-multi_teacher_tiny
SAVE_DIR=$ROOT/data-bin/$OUT_DATA_SIG
BERT_DIR=$ROOT/data/bert-base-cased-new
BART_DIR=$ROOT/data/bart-base

rm -rf $SAVE_DIR

# link split sentence and true target
#ln -s $DATA_DIR/encoder.en

INPUT=$DATA_DIR/head1w.train
# INPUT=$DATA_DIR/head1w.train

SOURCE=en
TARGET=de

INPUT_CON=$INPUT.mult.tmp.$SOURCE

export LC_ALL=en_US.UTF-8

#if [ ! -f $INPUT.raw.$SOURCE ]; then
#    ln -sf $INPUT.bert.$SOURCE $INPUT.raw.$SOURCE
#fi
#
#python3 $ROOT/fairseq_bertnmt/mult_teacher/con_tokenizer.py \
#--file_name $INPUT.raw.$SOURCE \
#--output_path $INPUT_CON \
#--bert_tokenizer $BERT_DIR \
#--extra_outs \
#--drop_path $INPUT_CON.drop \
#--bart_tokenizer $BART_DIR
#
## # filter target-side by drop list.
#wc -l $INPUT_CON.drop
#python3 $ROOT/fairseq_bertnmt/mult_teacher/filter_data.py $INPUT.$TARGET $INPUT_CON.drop

ln -sf $INPUT_CON $INPUT.mult.$SOURCE
ln -sf $INPUT.$TARGET.filter $INPUT.mult.$TARGET
ln -sf $INPUT_CON.bert $INPUT.mult.bert.$SOURCE
ln -sf $INPUT_CON.bart $INPUT.mult.bart.$SOURCE

# # link mapping files
ln -sf $INPUT_CON.data_dict.bart-base.map $INPUT.mult.bart.map.$SOURCE
ln -sf $INPUT_CON.data_dict.bert-base-cased-new.map $INPUT.mult.bert.map.$SOURCE

# add --input-mapping to make enable mapping preprocessing.
python3 $ROOT/fairseq_bertnmt/preprocess.py --source-lang en --target-lang de \
--trainpref $INPUT.mult \
--validpref $DATA_DIR/valid \
--testpref $DATA_DIR/test \
--destdir $SAVE_DIR \
--mult-teachers \
--avoid-tokenize-extras \
--workers 1 \
--input-mapping \
--bert-model-name $BERT_DIR \
--bart-model-name $BART_DIR

# --joined-dictionary --nwordssrc 32768 --nwordstgt 32768 \
