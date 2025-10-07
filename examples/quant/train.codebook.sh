#!/bin/bash

BASEDIR=./
path_2_data=./bin_data
lang_list=./langlist.txt

nslices=16
ncodes=10000
# we searched among {0.1, 0.5, 0.7, 0.9}, and found 0.9 and 0.5 the best for Indonesian- and English-bridge respectively
contchance=0.9

model=codebook.slice$nslices.ncodes$ncodes.passcont$contchance

FAIRSEQ_MODEL=$BASEDIR/model/$model

lang_pairs="id-ms,ms-id,id-tl,tl-id,id-jv,jv-id"  # For English-bridge, change to "en-ms,ms-en,en-tl,tl-en,en-jv,jv-en"
valid_lang_pairs=""

LANS="id jv ms tl"  #  For English-bridge, change to "en jv ms tl"

for slan in $LANS; do
        for tlan in $LANS; do
                if [[ $slan != $tlan ]]; then
                        echo $slan $tlan
                        if [ -z "$valid_lang_pairs" ]; then
                                valid_lang_pairs=$slan"-"$tlan
                        else
                                valid_lang_pairs=$valid_lang_pairs","$slan"-"$tlan
                        fi
                fi
        done
done

echo $lang_pairs
echo $valid_lang_pairs

mkdir $FAIRSEQ_MODEL -p

# Effective batch size is 2048 * 4 (GPUs) * 2 (update freq)
CUDA_VISIBLE_DEVICES=0 fairseq-train $path_2_data \
  --user-dir examples/quant/codebook_transformer_src \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --encoder-layers 6 --decoder-layers 6 \
  --encoder-attention-heads 16 --decoder-attention-heads 16 \
  --encoder-embed-dim 512 --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
  --arch codebook_transformer  \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 5.0 \
  --max-tokens 512 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --valid-lang-pairs "$valid_lang_pairs" \
  --criterion codebook_label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --update-freq 4 \
  --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --keep-best-checkpoints 5 \
  --codebook-loss-weight 1.0 --commitment-loss-weight 1.001 \
  --codebook-slices $nslices --codebook-size $ncodes \
  --alternate-cont-chance $contchance \
  --seed 222 --log-format simple --log-interval 1000 --fp16 \
  --max-update 500000 \
  --save-dir $FAIRSEQ_MODEL 2>&1 | tee -a $FAIRSEQ_MODEL/train.log