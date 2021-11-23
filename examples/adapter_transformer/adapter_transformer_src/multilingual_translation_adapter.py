# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
import torch
from fairseq import metrics


@register_task("multilingual_translation_adapter")
class MultilingualTranslationTaskAdapter(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument('--encoder-adapter', action='store_true', 
                help='Use adapters in encoder, freeze all other parameters.')
        parser.add_argument('--decoder-adapter', action='store_true',
                help='Use adapters in decoder, freeze all other parameters.')
        parser.add_argument('--drop-adapters-for-inference', action='store_true',
                help='Drop adapter modules at inference time.')
        parser.add_argument('--encoder-drop-residual', type=int,
                help='drop residual after self-attention in this encoder layer',)


    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)

        self.encoder_adapter = args.encoder_adapter
        self.decoder_adapter = args.decoder_adapter

#        if args.drop_adapters_for_inference is None:

#        self.drop_adapter_for_inference = args.drop_adapter_for_inference
#        self.src_langs, self.tgt_langs = set(), set()

#        for pair in args.lang_pairs:
#            src, tgt = pair.split("-")[0], pair.split("-")[1]
#            self.src_langs.add(src)
#            self.tgt_langs.add(tgt)
        
#            self.src_langs = list(self.src_langs)
#            self.tgt_langs = list(self.tgt_langs)

#            print("*************Unique src and tgt lang************************")
#            print("src", self.src_langs)
#            print("tgt", self.tgt_langs)

    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz and similarity loss"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

            if any("similarity_loss" in log for log in logging_outputs):
                similarity_loss = sum(log.get("similarity_loss", 0) for log in logging_outputs) / nsentences
                metrics.log_scalar("similarity_loss", similarity_loss, priority=200, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)


    def _get_src_tgt_lang_idx(self, lang_pair):
        # add index
        src, tgt = lang_pair.split("-")

        src_lang_idx, tgt_lang_idx = None, None

        if self.encoder_adapter:
            src_lang_idx = self.src_lang_idx_dict[src]
        if self.decoder_adapter:
            tgt_lang_idx = self.tgt_lang_idx_dict[tgt]

        return src_lang_idx, tgt_lang_idx


#    @property
#    def src_lang_idx_dict(self):
#        return {lang: lang_idx for lang_idx, lang in enumerate(self.src_langs)}

#    @property
#    def tgt_lang_idx_dict(self):
#        return {lang: lang_idx for lang_idx, lang in enumerate(self.tgt_langs)}
