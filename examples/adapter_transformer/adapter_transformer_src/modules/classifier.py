# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules import LayerNorm

from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor


class ClassificationLayer(nn.Module):
    def __init__(self, args, input_dim, middle_dim, output_dim):
        super(ClassificationLayer, self).__init__()
        self.fc_1 = nn.Linear(input_dim, middle_dim)
        self.fc_2 = nn.Linear(middle_dim, output_dim)
        self.dropout = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = F.relu(self.fc_1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x
