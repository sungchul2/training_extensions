# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base assigner."""

from abc import ABCMeta, abstractmethod

from otx.algo.common.utils.structures import AssignResult


class BaseAssigner(metaclass=ABCMeta):
    """Base class for assigner."""

    @abstractmethod
    def assign(self, *args, **kwargs) -> AssignResult:
        """Assign gt to priors."""
