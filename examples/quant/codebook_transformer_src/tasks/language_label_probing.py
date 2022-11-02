# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from .multilingual_translation_similarity import MultilingualTranslationSimilarityTask

@register_task("language_probing")
class LanguageProbing(MultilingualTranslationSimilarityTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        MultilingualTranslationSimilarityTask.add_args(parser)

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)

    def build_model(self, args):
        model = super().build_model(args)
        print(type(model.encoder))
        print(type(model.decoder))
        for name, param in model.named_parameters():
            # Freeze everything apart from encoder language classifier
            if "encoder.language_classifier" not in name:
                print("*** Freezing", name)
                param.requires_grad = False

        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        criterion.__class__.reduce_metrics(logging_outputs)
