"""Visual prompter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class VisualPrompter:
    def __new__(cls, type: str, *args, **kwargs):
        if type.lower() == "sam":
            from otx.algorithms.visual_prompting.adapters.torch.models.visual_prompters.sam import SAM
            return SAM(*args, **kwargs)
