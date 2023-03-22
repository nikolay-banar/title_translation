#!/bin/bash

# source language (example: fr)
S=$1
# target language (example: en)
T=$2
# name (example: Tenzil)
PART=$3
CUDA=$4
# path to dl4mt/data

ONMT=/home/nbanar/pycharmProjects/NMT-py

ROOT=/home/nbanar/pycharmProjects/data/"${S}""${T}"



for file in "${ROOT}"/ft/*
do

SRC="${file}"/${PART}/char/"${S}"-"${T}".${S}.tok
TGT="${file}"/${PART}/char/"${S}"-"${T}".${T}.tok
EXP="${file##*/}"
SAVE="${ONMT}"/predictions/${S}${T}/ft/${EXP}/char2charONMTTr4L


for MODEL in "${ONMT}"/open_models/"${S}""${T}"/ft/${EXP}/char2charONMTTr4L/*.pt
do
echo "${MODEL}"
done

if [[ ${PART} == *"test"* ]]; then
if [ ! -f  "${SAVE}"/pred.txt ]; then
mkdir -p "${SAVE}"
touch "${SAVE}"/pred.txt
CUDA_VISIBLE_DEVICES="${CUDA}" python translate.py --model "${MODEL}" --src "${SRC}" \
                                                --tgt "${TGT}" --save "${SAVE}" --test
fi
fi


if [[ ${PART} == *"dev"* ]]; then
if [ ! -f  "${SAVE}"/dev.txt ]; then

mkdir -p "${SAVE}"
touch "${SAVE}"/pred.txt
CUDA_VISIBLE_DEVICES="${CUDA}" python translate.py --model "${MODEL}" --src "${SRC}" \
                                                --tgt "${TGT}" --save "${SAVE}"
fi
fi
done



