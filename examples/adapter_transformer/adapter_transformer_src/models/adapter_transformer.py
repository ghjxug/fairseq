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
from ..modules.adapter_transformer_layer import (
        AdapterTransformerEncoderLayer, 
        AdapterTransformerDecoderLayer,
        )


@register_model("adapter_transformer")
class AdapterTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--bottleneck-dim",
            default=256,
            type=int,
            help="bottleneck size of adapter",
        )
        parser.add_argument(
            "--num-src-lang",
            default=1,
            type=int,
            help="number of unique adapters",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if args.encoder_adapter:
            return AdapterTransformerEncoder(args, src_dict, embed_tokens)
        else:
            return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.decoder_adapter:
            return AdapterTransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),
            )
        else:
            return AdapterTransformerEncoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),    
            )

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        #TODO: see if there's a better place to set this 
        return super().load_state_dict(state_dict, strict=False)


    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens = True,
        features_only = False,
        alignment_layer = None,
        alignment_heads = None,
    ):
#        print("SRC=================", src_tokens)

        return super().forward(src_tokens, src_lengths, prev_output_tokens,
                return_all_hiddens, features_only, alignment_layer, alignment_heads)


class AdapterTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList(
            [AdapterTransformerEncoderLayer(args) for idx in range(args.encoder_layers)]
        )

        for name, child in (self.named_children()):
            # Freeze everything other than the adapter
            for p_name, param in child.named_parameters():
                param.requires_grad = False

        for l in self.layers:
            l.activate_adapters()

#    def set_lang_idx(self, lang_idx):
#        for i, layer in enumerate(self.layers):
#            layer.set_lang_idx(lang_idx)


class AdapterTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.layers = nn.ModuleList(
            [AdapterTransformerDecoderLayer(args) for idx in range(args.decoder_layers)]
        )

        for name, child in (self.named_children()):
#            # Freeze everything other than the adapter
            for p_name, param in child.named_parameters():
                param.requires_grad = False

        for l in self.layers:
            l.activate_adapters()


@register_model_architecture(
    "adapter_transformer", "adapter_transformer"
)
def adapter_transformer_architecture(args):
    base_architecture(args)
