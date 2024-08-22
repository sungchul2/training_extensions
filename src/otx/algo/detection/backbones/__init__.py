# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom backbone implementations for detection task."""

from .backbone_factory import BackboneFactory
from .csp_darknet import CSPDarknet
from .presnet import PResNet

__all__ = ["BackboneFactory", "CSPDarknet", "PResNet"]
