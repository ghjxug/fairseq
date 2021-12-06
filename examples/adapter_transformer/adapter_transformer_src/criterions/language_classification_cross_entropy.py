# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)
import torch
import math
from fairseq import utils
import torch.nn.functional as F


@register_criterion("language_classification_cross_entropy")
class LanguageClassificationCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    class LabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
        def __init__(
                self,
                task,
                sentence_avg,
                label_smoothing,
                ignore_prefix_size=0,
                report_accuracy=False,
        ):
            super().__init__(
                task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # 1) forward pass for src -> tgt
        net_output = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        loss, nll_loss = self.compute_encoder_classification_loss(sample["net_input"], net_output)

        ####################################################
        # sample has the following keys:
        # id
        # nsentences
        # ntokens   --> number of total target tokens
        # net_input
        # src_tokens    --> source token indices
        # src_lengths   --> source length tensor (for each instance)
        # prev_output_tokens --> target (indices) shifted left
        # target --> target token (indices)
        ####################################################

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "classification_loss": 0,  # classification_loss.data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def compute_encoder_classification_loss(self, net_input, net_output, reduce=True):
        # get language tags
        # net_input["src_tokens"] is B x T
        src_lang_target = net_input["src_tokens"][:, 0] - 256000   # B  TODO: resolve hard-coding
        # assert src_lang_target >= 0
        encoder_classification_out = net_output[1]["classification_out"]
        max_len, bsz, _ = encoder_classification_out.shape
        # print("SRC Batch size:", src_lang_target.shape)
        # print("lang classifier:", encoder_classification_out.shape)

        # print("pred shape 1", F.log_softmax(src_lang_pred.float(), dim=-1).shape)
        # print("pred shape 2", model.get_normalized_probs(src_lang_pred, log_probs=True).shape) <-- this eats the 1st dim
        lprobs = F.log_softmax(encoder_classification_out.float(), dim=-1)  # softmax
        # print("===BEFORE", src_lang_target.shape)
        target = src_lang_target.repeat(max_len, 1) #.t()   # B --> T x B
        # print(target)
        # print("===AFTER", target.shape)

        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()

        lprobs, target = lprobs.view(-1, lprobs.size(-1)), target.view(-1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def add_args(parser):
        super(
            LanguageClassificationCrossEntropyCriterion,
            LanguageClassificationCrossEntropyCriterion,
        ).add_args(parser)
        # fmt: off

        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

        parser.add_argument(
            "--similarity-regularization-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="weight for similarity regularization, 0 means no similarity regularization",
        )

