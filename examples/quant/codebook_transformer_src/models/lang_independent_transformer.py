# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    base_architecture,
)
from fairseq.models.transformer import TransformerModel


@register_model("lang_independent_transformer")
class LangIndependentTransformer(TransformerModel):
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        # Setting strict to False due to newly added params
        return super().load_state_dict(state_dict, strict=True)

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        x, extra = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        # Additionally return encoder output
        extra["encoder_out"] = encoder_out["encoder_out"]
        extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        if "classification_out" in encoder_out:
            extra["classification_out"] = encoder_out["classification_out"]

        return x, extra

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LangIndependentTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            )


class LangIndependentTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        x, extra = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        # Additionally return encoder output
        extra["encoder_out"] = encoder_out["encoder_out"]
        extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        if "classification_out" in encoder_out:
            extra["classification_out"] = encoder_out["classification_out"]

        return x, extra


@register_model_architecture(
    "lang_independent_transformer", "lang_independent_transformer"
)
def lang_independent_transformer_architecture(args):
    base_architecture(args)
