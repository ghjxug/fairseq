# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from .multilingual_translation_adapter import MultilingualTranslationTaskAdapter
import torch
from fairseq import metrics


@register_task("multilingual_translation_adversarial_language_classifier")
class MultilingualTranslationAdversarial(MultilingualTranslationTaskAdapter):
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        assert args.left_pad_source is False
        assert args.encoder_langtok == "src"

        self.language_classifier_steps = args.language_classifier_steps
        self.language_classifier_one_vs_rest = args.language_classifier_one_vs_rest
        self.vocab_size = args.actual_vocab_size

    def build_model(self, args):
        model = super().build_model(args)
        print("Encoder type", type(model.encoder))
        print("Decoder type", type(model.decoder))

        return model

    def classification_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, vocab_size=256000
    ):
        # Improve classifier to distinguish source languages
        # Based on train_step from FairseqTask
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample,
                                                          classification_step=True,
                                                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                                                          vocab_size=vocab_size)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def translation_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, vocab_size=256000
    ):
        # Improve translation quality and trick classifier using reversed gradients
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample,
                                                          classification_step=False,
                                                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                                                          vocab_size=vocab_size)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    #

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # Alternate
        if update_num % self.language_classifier_steps == 0:
            return self.translation_step(sample, model, criterion, optimizer, update_num, ignore_grad, vocab_size=self.vocab_size)

        else:
            return self.classification_step(sample, model, criterion, optimizer, update_num, ignore_grad, vocab_size=self.vocab_size)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample,
                                                          classification_step=True,
                                                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                                                          vocab_size=self.vocab_size)
        return loss, sample_size, logging_output

    # def reduce_metrics(self, logging_outputs, criterion):
    #     super().reduce_metrics(logging_outputs, criterion)
    #     criterion.__class__.reduce_metrics(logging_outputs)

    # def _freeze_non_classifier(self, model):
    #     for name, param in model.named_parameters():
    #         # Freeze everything apart from encoder language classifier
    #         if "encoder.language_classifier" in name:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    #
    # def _unfreeze(self, model):
    #     for name, param in model.named_parameters():
    #         # Freeze everything apart from encoder language classifier
    #         if "encoder.language_classifier" in name:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
    #         # print(name, param.requires_grad)
    # #
    # def _freeze_classifier(self, model):
    #     for name, param in model.named_parameters():
    #         if "encoder.language_classifier" in name:
    #             param.requires_grad = False
    #             print("freeze", param)
    #
    # def _unfreeze_classifier(self, model):
    #     for name, param in model.named_parameters():
    #         if "encoder.language_classifier" in name:
    #             param.requires_grad = True
    #             print("unfreeze", param)
