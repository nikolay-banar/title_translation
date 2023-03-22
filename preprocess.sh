#!/bin/bash

# source language (example: fr)
S=$1
# target language (example: en)
T=$2
# name (example: Tenzil)

# path to dl4mt/data
APPLY=/home/nbanar/pycharmProjects/subword-nmt/apply_bpe.py

LEARN=/home/nbanar/pycharmProjects/subword-nmt/learn_bpe.py

ONMT=/home/nbanar/pycharmProjects/NMT-py/open_nmt

ROOT=/home/nbanar/pycharmProjects/data/"${S}""${T}"

for i in ${S} ${T}
do
    code="${ROOT}"/train/${i}.bpe
    for file in "${ROOT}"/ft/*
    do

      for part in train dev test
      do
        p="${file}"/"${part}"

        if [ ! -d "${p}"/bpe/ ]; then
          mkdir "${p}"/bpe/
        fi

        if [ ! -f "${p}"/bpe/"${S}"-"${T}"."${i}".bpe ]; then
          python ${APPLY} -c "${code}" < "${p}"/"${S}"-"${T}"."${i}".untok  > "${p}"/bpe/"${S}"-"${T}"."${i}".bpe
        fi

        if [ ! -d "${p}"/char/ ]; then
          mkdir "${p}"/char/
        fi

        if [ ! -f "${p}"/char/"${S}"-"${T}"."${i}".tok ]; then
          python ${ONMT}/preprocess_characters.py --i "${p}"/"${S}"-"${T}"."${i}".untok \
          --o "${p}"/char/"${S}"-"${T}".${i}.tok --tok 1
        fi
      done
    done
done


for file in "${ROOT}"/ft/*
do

if [ ! -d  "${file}"/b2b/ ]; then
  mkdir  "${file}"/b2b/
fi

if [ ! -f  "${file}"/b2b/"${S}"_"${T}"_prep.tok.train.0.pt ]; then
bash ./dataset.sh "${file}"/train/bpe/"${S}"-"${T}"."${S}".bpe \
                  "${file}"/train/bpe/"${S}"-"${T}"."${T}".bpe \
                  "${file}"/dev/bpe/"${S}"-"${T}"."${S}".bpe  \
                  "${file}"/dev/bpe/"${S}"-"${T}"."${T}".bpe \
                  bpe \
                  "${file}"/b2b/"${S}"_"${T}"_prep.tok \
                  "${ROOT}"/b2b/"${S}"_"${T}"_prep.tok.vocab.pt
fi

if [ ! -d  "${file}"/c2c/ ]; then
  mkdir  "${file}"/c2c/
fi

if [ ! -f  "${file}"/c2c/"${S}"_"${T}"_prep.tok.train.0.pt ]; then
bash ./dataset.sh "${file}"/train/char/"${S}"-"${T}"."${S}".tok \
                  "${file}"/train/char/"${S}"-"${T}"."${T}".tok \
                  "${file}"/dev/char/"${S}"-"${T}"."${S}".tok \
                  "${file}"/dev/char/"${S}"-"${T}"."${T}".tok \
                  char \
                  "${file}"/c2c/"${S}"_"${T}"_prep.tok \
                  "${ROOT}"/c2c/"${S}"_"${T}"_prep.tok.vocab.pt
fi



done

