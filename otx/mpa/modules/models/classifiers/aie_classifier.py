# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import VisionTransformer
from mmcls.models.builder import BACKBONES, CLASSIFIERS, HEADS, NECKS
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

from otx.mpa.modules.models.heads.custom_cls_head import CustomLinearClsHead

from .sam_classifier import SAMImageClassifier


class ResMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return out


@HEADS.register_module()
class ResMLPClassifier(CustomLinearClsHead):
    def __init__(self, num_classes, in_channels, *args, **kwargs) -> None:
        super().__init__(num_classes, in_channels, *args, **kwargs)
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, 256)
        self.block1 = ResMLPBlock(256, 256)
        self.block2 = ResMLPBlock(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def fc(self, x):
        x = x.view(-1, self.in_channels)
        x = F.relu(self.fc1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc2(x)
        return x
    

@NECKS.register_module()
class CLIPProjection(BaseModule):
    def __init__(self, in_channels, out_channels, init_cfg=None):
        super(CLIPProjection, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = in_channels ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(in_channels, out_channels))

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[-1]
            out = inputs @ self.proj
        elif isinstance(inputs, torch.Tensor):
            out = inputs @ self.proj
        else:
            raise TypeError('`CLIPProjection` neck inputs should be tuple or torch.tensor')
        return (out, )


@BACKBONES.register_module()
class CLIPVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, norm_layer = build_norm_layer(kwargs['norm_cfg'], self.embed_dims, postfix=1)
        self.add_module('pre_norm', norm_layer)
        # self.proj = nn.Parameter((self.embed_dims ** -0.5) * torch.randn(self.embed_dims, 768))

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + VisionTransformer.resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)


@CLASSIFIERS.register_module()
class MMCLSVisionTransformerSAMImageClassifier(SAMImageClassifier):
    def extract_feat(self, img):
        x = super().extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        if self.with_neck:
            cls_token = x
        else:
            _, cls_token = x
        return cls_token


@CLASSIFIERS.register_module()
class FrozenMMCLSVisionTransformerSAMImageClassifier(MMCLSVisionTransformerSAMImageClassifier):
    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class CLIPVisionTransformerSAMImageClassifier(SAMImageClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = clip.load("ViT-L/14@336px", "cpu")[0].visual
        self.backbone.to(device)

    def extract_feat(self, img):
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class FrozenCLIPVisionTransformerSAMImageClassifier(CLIPVisionTransformerSAMImageClassifier):
    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class SAMImageClassifierTrainOnlyHead(SAMImageClassifier):
    def extract_feat(self, img):
        return img
