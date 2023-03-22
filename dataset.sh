#!/bin/bash
PREP=/home/nbanar/pycharmProjects/OpenNMT-py/preprocess.py


TRAIN_SRC=$1
TRAIN_TGT=$2
VAL_SRC=$3
VAL_TGT=$4
TOK=$5
SAVE=$6
VOC=$7

if [ "${TOK}" == "bpe" ]; then

    python ${PREP} -train_src ${TRAIN_SRC} -train_tgt ${TRAIN_TGT} -valid_src ${VAL_SRC} -valid_tgt ${VAL_TGT} \
                    -save_data  ${SAVE} --src_seq_length 50 --tgt_seq_length 100 --src_vocab "${VOC}"

fi

if [ "${TOK}" == "char" ]; then

    python ${PREP} -train_src ${TRAIN_SRC} -train_tgt ${TRAIN_TGT} -valid_src ${VAL_SRC} -valid_tgt ${VAL_TGT} \
                    -save_data  ${SAVE} \
                    --src_seq_length 450 --tgt_seq_length 500 --src_vocab_size 300 --tgt_vocab_size 300 \
                    --src_vocab "${VOC}"

fi