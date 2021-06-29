# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.data import data_utils as fairseq_data_utils


class SampledMultiDatasetReverse(SampledMultiDataset):
    def collater(self, samples, **extra_args):
        super().collater(samples, **extra_args)

        prev_input_tokens = fairseq_data_utils.collate_tokens(
            [s["net_input"]["src_tokens"] for s in samples],
            self.pad_index,
            self.eos_index,
            left_pad=False,
            move_eos_to_beginning=self.move_eos_to_beginning,
        )

        batch["prev_input_tokens"] = prev_input_tokens
