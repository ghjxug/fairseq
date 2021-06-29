# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
import torch


@register_task("multilingual_translation_adapter")
class MultilingualTranslationTaskAdapter(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument('--encoder-adapter', action='store_true', help='Use adapters in encoder')
        parser.add_argument('--decoder-adapter', action='store_true', help='Use adapters in decoder')
        parser.add_argument('--drop-adapters-for-inference', action='store_true',
                help='Keep adapter modules at inference time')

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


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        # self.datasets["train"] is SampledMultiDataset
        # What's the current language?
#        if self.encoder_adapter:
#            model.encoder.set_lang_idx(0)

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    
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
