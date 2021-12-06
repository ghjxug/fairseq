# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)

from fairseq.models.transformer import TransformerModel
from ..modules.adapter_transformer_layer import (
    AdapterTransformerEncoderLayer,
    AdapterTransformerDecoderLayer,
)
from ..models.adapter_transformer import AdapterTransformerModel, AdapterTransformerEncoder
from ..modules.classifier import ClassificationLayer


@register_model("language_probing_transformer")
class LanguageProbingTransformerModel(AdapterTransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser = AdapterTransformerModel.add_args(parser)
        parser.add_argument(
            "--classifier-middle-layer-size",
            default=256,
            type=int,
            help="TBA",
        )
        parser.add_argument(
            "--num-language-to-classify",
            required=True,
            type=int,
            help="TBA",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LanguageProbingTransformerEncoder(args, src_dict, embed_tokens)

    def load_state_dict(
            self,
            state_dict,
            strict=True,
            model_cfg=None,
            args=None,
    ):
        # Setting strict to False due to newly added parameters
        return super().load_state_dict(state_dict, strict=False)


class LanguageProbingTransformerEncoder(AdapterTransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        print(args.encoder_embed_dim,
              args.classifier_middle_layer_size,
              args.num_language_to_classify)

        self.language_classifier = ClassificationLayer(args=args,
                                                       input_dim=args.encoder_embed_dim,
                                                       middle_dim=args.classifier_middle_layer_size,
                                                       output_dim=args.num_language_to_classify,
                                                       )

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        enc_out_dict = super().forward_scriptable(src_tokens,
                                             src_lengths,
                                             return_all_hiddens,
                                             token_embeddings)

        enc_out = enc_out_dict["encoder_out"][0]   # T x B x C
        lang_classifier_out = self.language_classifier(enc_out)     # T x B x num_lan
        # print("***************************", lang_classifier_out.shape)


        enc_out_dict["classification_out"] = lang_classifier_out

        return enc_out_dict


@register_model_architecture(
    "language_probing_transformer", "language_probing_transformer"
)
def language_probing_transformer_architecture(args):
    base_architecture(args)
