"""Visual prompter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


def build_prompter(type: str):
    if type.lower() == "sam":
        from otx.algorithms.visual_prompting.adapters.torch.models.visual_prompters.sam import SAM
        return SAM


class VisualPrompter:
    def __new__(cls, type: str, *args, **kwargs):
        return build_prompter(type)(*args, **kwargs)
