# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MultilingualDatasetManagerReverse(MultilingualDatasetManager):
    def __init__(self, args, lang_pairs, langs, dicts, sampling_method):
        super().__init__()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
            action=FileContentsAction,
        )

    def load_into_concat_dataset(self, split, datasets, data_param_list):
        if self.args.lang_tok_replacing_bos_eos:
            # TODO: to investigate why TransformEosLangPairDataset doesn't work with ConcatDataset
            return SampledMultiDataset(
                OrderedDict(datasets),
                sampling_ratios=None,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=None,
                split=split,
            )
        return ConcatDataset([d for _, d in datasets])


