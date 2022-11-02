#!/bin/bash

BASEDIR=/path/to/working/dir
path_2_data=/path/to/binarized/data
lang_list=/path/to/langlist

model=adversarial.langid

FAIRSEQ_MODEL=$BASEDIR/model/$model

language_classifier_one_vs_rest=43 # For English-bridge, change to 24 (both assuming our binarized data)
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

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $path_2_data \
  --user-dir examples/quant/codebook_transformer_src \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --encoder-layers 6 --decoder-layers 6 \
  --encoder-attention-heads 16 --decoder-attention-heads 16 \
  --encoder-embed-dim 512 --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
  --arch language_classification_transformer \
  --task multilingual_translation_adversarial_language_classifier \
  --sampling-method "temperature" \
  --sampling-temperature 5.0 \
  --max-tokens $bsz \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion language_classification_cross_entropy --label-smoothing 0.1 \
  --valid-lang-pairs "$valid_lang_pairs" \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --update-freq $updates \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --num-language-to-classify 124 --actual-vocab-size 165034 --report-accuracy \
  --language-classifier-one-vs-rest 43 \
  --ddp-backend=legacy_ddp \
  --seed 222 --log-format simple --log-interval 1000 --fp16 \
  --left-pad-source False \
  --finetune-from-model $OUT_DIR/mm100_small.trimmed/model.pt \
  --save-dir $OUT_DIR/$model 2>&1 | tee $OUT_DIR/$model/train.log
