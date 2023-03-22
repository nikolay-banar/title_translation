MODEL=$1
SRC=$2
TGT=$3
CUDA=$4

ROOT=/home/nbanar/pycharmProjects/data/${SRC}${TGT}
TRAIN=/home/nbanar/pycharmProjects/OpenNMT-py/train.py
PROJECT=/home/nbanar/pycharmProjects/NMT-py


train () {
    CUDA_VISIBLE_DEVICES="${CUDA}" python ${TRAIN} -data "${DATA}" -save_model "${SAVE}" \
     -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
     -encoder_type "${TYPE}" -decoder_type "${TYPE}" -position_encoding \
     -train_steps 50000  -dropout "${DROPOUT}" -batch_size "${BATCH}" -batch_type tokens \
     --log_file "${LOGS}"/"${TOK}".txt \
     -normalization tokens  -accum_count "${AC}" -optim adam -adam_beta2 0.998 --train_from "${CP}" \
     -learning_rate 0.0001 -max_grad_norm 0 --reset_optim all  --early_stopping 2 --keep_checkpoint 1 \
     -label_smoothing 0.1 -valid_steps "${VALID}" -save_checkpoint_steps "${VALID}" -gpu_ranks 0
}



if [[ ${MODEL} == *"char2char"* ]]; then

BATCH=6144
AC=4
TOK=c2c
CP=${PROJECT}/open_models/${SRC}${TGT}/${MODEL}/${SRC}${TGT}_model_transformer_step_100000.pt
DROPOUT=0
TYPE=transformer
VALID=200
echo "simple character level system!"
fi



for file in "${ROOT}"/ft/*
do
DATA=${file}/${TOK}/${SRC}_${TGT}_prep.tok
EXP="${file##*/}"
SAVE=${PROJECT}/open_models/${SRC}${TGT}/ft/${EXP}/${MODEL}/${SRC}${TGT}_model_transformer
LOGS=${PROJECT}/logs/${SRC}${TGT}/ft/${MODEL}/${EXP}

if [ ! -d "${LOGS}" ]; then
  mkdir -p "${LOGS}"
else
  echo "${LOGS} exists"
fi

if [ ! -d "${SAVE}" ]; then
  mkdir -p "${SAVE}"
  train
else
  echo "${SAVE} exists"
fi

done


