# Learning an Artificial Language for Knowledge-Sharing in Multilingual Translation
[Paper]() to appear in WMT 2022.

## Data and Preprocessing
The [training data](https://data.statmt.org/wmt21/multilingual-task/small_task2_filt_v2.tar.gz) come from 
Small Track 2 of WMT 2021's [Large Scale Multilingual Machine Translation Task](https://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html).

The dev and test data are [FLoRes-101](https://github.com/facebookresearch/flores/tree/main/previous_releases/flores101)
dev and devtest sets.

For faster reproducibility of the experiments, we share the [binarized data](https://bwsyncandshare.kit.edu/s/WC9yZXTJNgPFWSS).

We also share the binarized [oracle data](https://bwsyncandshare.kit.edu/s/HwKKfGJSBfLXrBm/download/zeroshot_oracle.tar.xz) 
for zero-shot directions (used in Section 7.1 of our paper) .

## Training
We initialize from a pretrained M2M-124 model ([Goyal et al. 2022](https://aclanthology.org/2022.tacl-1.30.pdf)).

For faster training, we trimmed the vocabulary (original size 256K) by excluding all entries 
that did not appear in the training data.

We share the trimmed [model](https://bwsyncandshare.kit.edu/s/5ZXZxxjAyQws2MJ/download/model.pt) and [vocabulary](https://bwsyncandshare.kit.edu/s/W3Tadsmm6ir6JHW/download/trimmed_dict.txt). 

## Transformer with Codebook

### Trained Models
| Data              | Model                                                                                  | Scores               | 
|-------------------|----------------------------------------------------------------------------------------|----------------------|
| Indonesian-bridge | [ckpt](https://bwsyncandshare.kit.edu/s/zbokT24kps8qEZP/download/checkpoint_last_5.pt) | Row (3.3) of Table 3 |
| English-bridge    | [ckpt](https://bwsyncandshare.kit.edu/s/TAGNsdsPjTYtJ8m/download/checkpoint_last_5.pt) | Row (6.3) of Table 3 |

### Training
```bash
#!/bin/bash

BASEDIR=/path/to/working/dir
path_2_data=/path/to/binarized/data
lang_list=/path/to/langlist

nslices=16
ncodes=10000
# we searched among {0.1, 0.5, 0.7, 0.9}, and found 0.9 and 0.5 the best for Indonesian- and English-bridge respectively
contchance=0.9

model=small.en.pretrained.slice$nslices.ncodes$ncodes.toggle.noquanttok.passcont$contchance

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
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $path_2_data \
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
  --max-tokens 2048 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --valid-lang-pairs "$valid_lang_pairs" \
  --criterion encoder_similarity_label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --update-freq 2 \
  --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --keep-best-checkpoints 5 \
  --codebook-loss-weight 1.0 --commitment-loss-weight 1.001 \
  --codebook-slices $nslices --codebook-size $ncodes \
  --alternate-cont-chance $contchance \
  --seed 222 --log-format simple --log-interval 1000 --fp16 \
  --finetune-from-model $BASEDIR/model/mm100_small.trimmed/model.pt \
  --max-update 500000 \
  --save-dir $FAIRSEQ_MODEL 2>&1 | tee -a $FAIRSEQ_MODEL/train.log
```

### Inference
```bash
name="codebook.id" # model directory name
ckpt="checkpoint_last_5" # checkpoint name
GPU=0  # GPU
bash ./pred.sh $name $ckpt $GPU 
```

## Models we compared to  

### Language-Independent Objective
Approach first described in [Pham et al. (2019)](https://arxiv.org/pdf/1906.08584.pdf); [Arivazhagan et al., (2019)](https://arxiv.org/pdf/1903.07091.pdf)

#### Training
```bash
bash ./train.langind.sh
```

#### Inference
Same as above

### Adversarial language classifier
Approach first described in [Arivazhagan et al., (2019)](https://arxiv.org/pdf/1903.07091.pdf)

#### Before Training
These parameters should be set differently depending on the training data: 
* `--num-language-to-classify`: total number of languages
* `--language-classifier-one-vs-rest`: if non-zero, language classification will be binary (this class vs the rest). We used binary classification since it gave better performance in initial experiments.  
* `--actual-vocab-size`: vocabulary size, excluding language tags

Before training, please replace `$FAIRSEQ_DIR/fairseq/trainer.py` by `$FAIRSEQ_DIR/fairseq/trainer_adv.py`.
The differences to `train.py` are:
* It builds a separate optimizer for the adversarial classifier.
* It scales num_updates, so that the optimizer schedules are unaffected by the additional steps from the alternating training.

#### Training
```bash
bash ./train.adversarial.sh
```

#### Inference
Same as above

## Contact
Please feel free to open an issue or mail to danni.liu@kit.edu. :)