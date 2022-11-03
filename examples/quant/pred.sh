#!/bin/bash

BASEDIR=/path/to/working/dir
path_2_data=/path/to/binarized/data
lang_list=/path/to/langlist

name=$1 # model name
ckpt=$2 # checkpoint name
GPU=$3  # GPU

FAIRSEQ_MODEL=$BASEDIR/model/$name

OUTDIR=$BASEDIR/data/$name/$ckpt/
mkdir -p $OUTDIR

echo "outdir" $OUTDIR

LANS="id jv ms tl"  # For English-bridge, change to "en jv ms tl"

# Point to dictionary and language pair of pretrained model
ln -s $BASEDIR/model/mm100_small.trimmed/dict.txt $BASEDIR/model/$name/
ln -s $BASEDIR/model/mm100_small.trimmed/language_pairs.txt $BASEDIR/model/$name/

for sl in $LANS; do
for tl in $LANS; do
        if [[ $sl != $tl ]]; then

        CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
            $path_2_data \
            --user-dir examples/quant/codebook_transformer_src \
            --batch-size 1024 \
            --remove-bpe 'sentencepiece' \
            --task translation_multi_simple_epoch \
            --path $FAIRSEQ_MODEL/$ckpt.pt \
            --fixed-dictionary $FAIRSEQ_MODEL/dict.txt \
            -s $sl -t $tl \
            --beam 5 \
            --lang-dict $lang_list \
            --lang-pairs $FAIRSEQ_MODEL/language_pairs.txt \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test \
            --dataset-impl mmap \
            --left-pad-source False \
            --distributed-world-size 1 --distributed-no-spawn --fp16 > $OUTDIR/$sl-$tl.pred.log

        grep ^H $OUTDIR/$sl-$tl.pred.log | cut -f3- > $OUTDIR/$sl-$tl.sys
        grep ^T $OUTDIR/$sl-$tl.pred.log | cut -f2- > $OUTDIR/$sl-$tl.ref

        cat $OUTDIR/$sl-$tl.sys | sacrebleu $OUTDIR/$sl-$tl.ref -tok spm > $OUTDIR/$sl-$tl.spbleu
        cat $OUTDIR/$sl-$tl.sys | sacrebleu $OUTDIR/$sl-$tl.ref -m chrf --chrf-word-order 2 > $OUTDIR/$sl-$tl.chrf

        fi
done
done
