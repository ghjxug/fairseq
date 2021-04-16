# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)

from fairseq.models.transformer import TransformerModel
from ..modules.residual_drop_transformer_layer import ResidualDropTransformerEncoderLayer

@register_model("residual_drop_transformer")
class ResidualDropTransformerModel(TransformerModel):
    """TODO: A variant of Transformer as is in "xxx"
    (https://arxiv.org/abs/2009.13102).
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)   #TODO:
        parser.add_argument(
            '--encoder-drop-residual',
            type=int,
            help='drop residual after self-attention in this encoder layer',
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ResidualDropTransformerEncoder(args, src_dict, embed_tokens)


class ResidualDropTransformerEncoder(TransformerEncoder):
    """Residual drop (https://) implemented in
    TransformerEncoder.
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layers = nn.ModuleList(
            [self._build_encoder_layer(args, idx, args.encoder_drop_residual) for idx in range(args.encoder_layers)]
        )
    
    def _build_encoder_layer(self, args, layer_idx, encoder_drop_residual_at_layer=None):
        drop_residual_after_att = (layer_idx == encoder_drop_residual_at_layer)
        return ResidualDropTransformerEncoderLayer(args, drop_residual_after_att)


@register_model_architecture(
    "residual_drop_transformer", "residual_drop_transformer"
)
def residual_drop_transformer_architecture(args):
    base_architecture(args)
