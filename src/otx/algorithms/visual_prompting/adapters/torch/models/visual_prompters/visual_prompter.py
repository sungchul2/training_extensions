"""Visual prompter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn


class VisualPrompter(nn.Module):
    def __init__(self, type: str, *args, **kwargs):
        super().__init__()
        self.prompter = self.build_prompter(type)(*args, **kwargs)
        
    def build_prompter(self, type: str):
        if type.lower() == "sam":
            from otx.algorithms.visual_prompting.adapters.torch.models.visual_prompters.sam import SAM
            return SAM
