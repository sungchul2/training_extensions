"""Unet architecture from Habana Model-References, dropblock, and MONAI.

https://github.com/HabanaAI/Model-References/tree/master
https://github.com/miguelvr/dropblock/tree/master
https://github.com/Project-MONAI/MONAI/tree/dev
"""

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.ops import resize
from enum import Enum
from otx.algorithms.segmentation.adapters.mmseg.models.segmentors.otx_encoder_decoder import OTXEncoderDecoder
from mmseg.models.builder import SEGMENTORS
import warnings
from torch.nn.modules.loss import _Loss
from typing import Callable, Optional, Union, Sequence


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}


def get_norm(name, out_channels):
    if "groupnorm" in name:
        return nn.GroupNorm(32, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim, bias=False):
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_drop_block():
    return LinearScheduler(
        DropBlock3D(block_size=5, drop_prob=0.0),
        start_value=0.0,
        stop_value=0.1,
        nr_steps=10000,
    )


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block()

    def forward(self, data):
        out = self.conv(data)
        if self.use_drop_block:
            self.drop_block.step()
            out = self.drop_block(out)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class ResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ResidBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block()
            self.skip_drop_block = get_drop_block()
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            self.downsample = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
            self.norm_res = get_norm(kwargs["norm"], out_channels)

    def forward(self, input_data):
        residual = input_data
        out = self.conv1(input_data)
        out = self.conv2(out)
        if self.use_drop_block:
            out = self.drop_block(out)
        out = self.norm(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.use_drop_block:
                residual = self.skip_drop_block(residual)
            residual = self.norm_res(residual)
        out = self.lrelu(out + residual)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm, dim):
        super(AttentionLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=3, stride=1, dim=dim)
        self.norm = get_norm(norm, out_channels)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, kwargs["dim"])
        self.conv_block = ConvBlock(2 * out_channels, out_channels, kernel_size, 1, **kwargs)
        self.attention = kwargs["attention"]
        if self.attention:
            att_out, norm, dim = out_channels // 2, kwargs["norm"], kwargs["dim"]
            self.conv_o = AttentionLayer(out_channels, att_out, norm, dim)
            self.conv_s = AttentionLayer(out_channels, att_out, norm, dim)
            self.psi = AttentionLayer(att_out, 1, norm, dim)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        if self.attention:
            out_a = self.conv_o(out)
            skip_a = self.conv_s(skip_data)
            psi_a = self.psi(self.relu(out_a + skip_a))
            attention = self.sigmoid(psi_a)
            skip_data = skip_data * attention
        out = torch.cat((out, skip_data), dim=1)
        out = self.conv_block(out)
        return out


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, input_data):
        return self.conv(input_data)
    
    
##########################
########## Loss ##########
##########################
def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class LossReduction(StrEnum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

    
class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added to the intersection and union components of the inter-over-union calculation to smooth results
    respectively, these values should be small. The `include_background` class attribute can be set to False for
    an instance of DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be
    background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                # input = torch.softmax(input, 1)
                # [Habana]
                input = F.softmax(input, 1)


        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target_split = target.split([1, target.shape[1]-1], dim=1)
                target = target_split[1]
                input_split = input.split([1, input.shape[1]-1], dim=1)
                input = input_split[1]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
    
def softmax_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls = input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target

    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss


def sigmoid_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # see also https://github.com/pytorch/pytorch/blob/v1.9.0/aten/src/ATen/native/Loss.cpp#L231
    max_val = (-input).clamp(min=0)
    loss: torch.Tensor = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    # sigmoid(-i) if t==1; sigmoid(i) if t==0 <=>
    # 1-sigmoid(i) if t==1; sigmoid(i) if t==0 <=>
    # 1-p if t==1; p if t==0 <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss
    
    
class FocalLoss(_Loss):
    """
    FocalLoss is an extension of BCEWithLogitsLoss that down-weights loss from
    high confidence correct predictions.

    Reimplementation of the Focal Loss described in:

        - ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole-volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018

    Example:
        >>> import torch
        >>> from monai.losses import FocalLoss
        >>> from torch.nn import BCEWithLogitsLoss
        >>> shape = B, N, *DIMS = 2, 3, 5, 7, 11
        >>> input = torch.rand(*shape)
        >>> target = torch.rand(*shape)
        >>> # Demonstrate equivalence to BCE when gamma=0
        >>> fl_g0_criterion = FocalLoss(reduction='none', gamma=0)
        >>> fl_g0_loss = fl_g0_criterion(input, target)
        >>> bce_criterion = BCEWithLogitsLoss(reduction='none')
        >>> bce_loss = bce_criterion(input, target)
        >>> assert torch.allclose(fl_g0_loss, bce_loss)
        >>> # Demonstrate "focus" by setting gamma > 0.
        >>> fl_g2_criterion = FocalLoss(reduction='none', gamma=2)
        >>> fl_g2_loss = fl_g2_criterion(input, target)
        >>> # Mark easy and hard cases
        >>> is_easy = (target > 0.7) & (input > 0.7)
        >>> is_hard = (target > 0.7) & (input < 0.3)
        >>> easy_loss_g0 = fl_g0_loss[is_easy].mean()
        >>> hard_loss_g0 = fl_g0_loss[is_hard].mean()
        >>> easy_loss_g2 = fl_g2_loss[is_easy].mean()
        >>> hard_loss_g2 = fl_g2_loss[is_hard].mean()
        >>> # Gamma > 0 causes the loss function to "focus" on the hard
        >>> # cases.  IE, easy cases are downweighted, so hard cases
        >>> # receive a higher proportion of the loss.
        >>> hard_to_easy_ratio_g2 = hard_loss_g2 / easy_loss_g2
        >>> hard_to_easy_ratio_g0 = hard_loss_g0 / easy_loss_g0
        >>> assert hard_to_easy_ratio_g2 > hard_to_easy_ratio_g0
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        use_softmax: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the loss calculation.
                If False, `alpha` is invalid when using softmax.
            to_onehot_y: whether to convert the label `y` into the one-hot format. Defaults to False.
            gamma: value of the exponent gamma in the definition of the Focal loss. Defaults to 2.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss.
                The value should be in [0, 1]. Defaults to None.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            use_softmax: whether to use softmax to transform the original logits into probabilities.
                If True, softmax is used. If False, sigmoid is used. Defaults to False.

        Example:
            >>> import torch
            >>> from monai.losses import FocalLoss
            >>> pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
            >>> grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
            >>> fl = FocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.use_softmax = use_softmax
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                The input should be the original logits since it will be transformed by
                a sigmoid/softmax in the forward function.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            ValueError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
            ValueError: When ``self.weight`` is a sequence and the length is not equal to the
                number of classes.
            ValueError: When ``self.weight`` is/contains a value that is less than 0.

        """
        n_pred_ch = input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            self.class_weight = self.class_weight.to(loss)
            broadcast_dims = [-1] + [1] * len(target.shape[2:])
            self.class_weight = self.class_weight.view(broadcast_dims)
            loss = self.class_weight * loss

        if self.reduction == LossReduction.SUM.value:
            # Previously there was a mean over the last dimension, which did not
            # return a compatible BCE loss. To maintain backwards compatible
            # behavior we have a flag that performs this extra step, disable or
            # parameterize if necessary. (Or justify why the mean should be there)
            average_spatial_dims = True
            if average_spatial_dims:
                loss = loss.mean(dim=list(range(2, len(target.shape))))
            loss = loss.sum()
        elif self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.NONE.value:
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return loss
    
    
class Loss(nn.Module):
    def __init__(self, focal):
        super(Loss, self).__init__()
        self.dice = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, batch=True)
        self.focal = FocalLoss(gamma=2.0)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        self.use_focal = focal

    def forward(self, y_pred, y_true):
        loss = self.dice(y_pred, y_true)
        if self.use_focal:
            loss += self.focal(y_pred, y_true)
        else:
            loss += self.cross_entropy(y_pred, y_true[:, 0].long())
        return loss


# affinity='disabled',
# amp=False,
# attention=False,
# augment=True,
# batch_size=64,
# benchmark=False,
# blend='gaussian',
# bucket_cap_mb=130,
# ckpt_path=None,
# data='/local/data/brats/01_2d',
# data2d_dim=3,
# deep_supervision=True,
# dim=2,
# drop_block=False,
# exec_mode='train',
# factor=0.3,
# focal=False,
# fold=0,
# framework='pytorch-lightning',
# gpus=0,
# gradient_clip=False,
# gradient_clip_norm=12,
# gradient_clip_val=0,
# habana_loader=True,
# hpus=8,
# inference_mode='graphs',
# is_autocast=True,
# learning_rate=0.001,
# logname='res_log',
# lr_patience=70,
# max_epochs=10000,
# measurement_type='throughput',
# min_epochs=30,
# momentum=0.99,
# negative_slope=0.01,
# nfolds=5,
# norm='instance',
# num_workers=8,
# nvol=1,
# optimizer='fusedadamw',
# overlap=0.5,
# oversampling=0.33,
# patience=100,
# profile=False,
# profile_steps='90:95',
# progress_bar_refresh_rate=25,
# residual=False,
# results='results/unet2d_hpu8_bf16_bs64',
# resume_training=False,
# run_lazy_mode=True,
# save_ckpt=False,
# save_preds=False,
# scheduler='none',
# seed=123,
# set_aug_seed=False,
# skip_first_n_eval=0,
# steps=None,
# sync_batchnorm=False,
# task='1',
# test_batches=0,
# train_batches=0,
# tta=False,
# use_torch_compile=False,
# val_batch_size=64,
# warmup=5,
# weight_decay=0.0001


@SEGMENTORS.register_module()
class HabanaUNet(nn.Module):
    def __init__(
        self,
        num_classes,
        kernels=[3, 3, 3, 3], # manually set
        strides=[1, 1, 1, 1], # manually set
        in_channels=3,
        normalization_layer="instance",
        negative_slope=0.01,
        deep_supervision=True,
        attention=False,
        drop_block=False,
        residual=False,
        dimension=2,
        **kwargs
    ):
        super(HabanaUNet, self).__init__()
        self.dim = dimension
        self.num_classes = num_classes
        self.attention = attention
        self.residual = residual
        self.negative_slope = negative_slope
        self.deep_supervision = deep_supervision
        self.norm = normalization_layer + f"norm{dimension}d"
        self.filters = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(len(strides))]

        down_block = ResidBlock if self.residual else ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            in_channels=in_channels,
            out_channels=self.filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
            drop_block=drop_block,
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
            drop_block=drop_block,
        )
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block(decoder_level=0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        
        self.apply(self.initialize_weights)
        
        self.loss = Loss(focal=False)

    def forward(self, img, img_metas, **kwargs):
        if not kwargs.get("return_loss", True):
            img = img[0]
        out = self.input_block(img)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)
        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[2:-1][::-1]):
                out.append(self.deep_supervision_heads[i](decoder_out))
                
        if kwargs.get("return_loss", True):
            # train
            losses = dict(loss=self.compute_loss(out, kwargs["gt_semantic_seg"]))
            return losses
        else:
            # test
            resize_shape = img_metas[0][0]['img_shape'][:2]
            seg_logit = out[:, :, :resize_shape[0], :resize_shape[1]]
            size = img_metas[0][0]['ori_shape'][:2]

            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=False,
                warning=False)

            if self.num_classes == 1:
                seg_logit = F.sigmoid(seg_logit)
                seg_pred = (seg_logit > 0.3).to(seg_logit).squeeze(1)
            else:
                seg_logit = F.softmax(seg_logit, dim=1)
                seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            seg_pred = list(seg_pred)
            
            return seg_pred
        
    def compute_loss(self, preds, label):
        if self.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)
    
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)

        outputs = dict(
            loss=losses["loss"],
            log_vars={k: v.detach().cpu() for k, v in losses.items()},
            num_samples=len(data_batch['img_metas']))

        return outputs

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride, drop_block=False):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            drop_block=drop_block,
            kernel_size=kernel_size,
            in_channels=in_channels,
            attention=self.attention,
            out_channels=out_channels,
            negative_slope=self.negative_slope,
        )
        
    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.num_classes, dim=self.dim)
    
    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)])

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block, drop_block=False):
        layers = []
        for i, (in_channel, out_channel, kernel, stride) in enumerate(zip(in_channels, out_channels, kernels, strides)):
            use_drop_block = drop_block and len(in_channels) - i <= 2
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride, use_drop_block)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
