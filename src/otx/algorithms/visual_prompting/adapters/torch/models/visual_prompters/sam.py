"""Segment Anything Model for zero-shot inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.torch.datasets.pipelines import (
    ResizeLongestSide,
)
from otx.algorithms.visual_prompting.adapters.torch.models.backbones import (
    build_tiny_vit,
    build_vit,
)
from otx.algorithms.visual_prompting.adapters.torch.models.decoders import (
    SAMMaskDecoder,
)
from otx.algorithms.visual_prompting.adapters.torch.models.encoders import (
    SAMPromptEncoder,
)


class SAM(nn.Module):
    """Segment Anything Model for zero-shot inference."""
    CKPT_PATHS = {
        "tiny_vit": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    def __init__(
        self,
        backbone: str,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        device: str = "cuda"
    ) -> None:
        super().__init__()
        assert backbone.lower() in ["tiny_vit", "vit_b", "vit_l", "vit_h"]
        self.backbone = backbone.lower()
        self.image_size = 1024
        self.prompt_embed_dim = 256
        self.vit_patch_size = 16
        self.image_embedding_size = self.image_size // self.vit_patch_size
        self.image_format = "RGB"
        self.device = device
        self.mask_threshold = 0.

        self.set_image_encoder()
        self.set_prompt_encoder()
        self.set_mask_decoder()
        self.load_checkpoint()

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.eval()
        self.to(self.device)

        self.transform = ResizeLongestSide(self.img_size)
        self.reset_image()

    def set_image_encoder(self) -> None:
        if self.backbone == "tiny_vit":
            self.image_encoder = build_tiny_vit(self.image_size)
        elif "vit" in self.backbone:
            self.image_encoder = build_vit(self.backbone, self.image_size)
        else:
            raise NotImplementedError(
                (f"{self.backbone} for image encoder of SAM is not implemented yet. "
                 f"Use vit_b, l, or h.")
            )

    def set_prompt_encoder(self) -> None:
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=256,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

    def set_mask_decoder(self) -> None:
        self.mask_decoder = SAMMaskDecoder(
            num_multimask_outputs=3,
            transformer_cfg=dict(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def load_checkpoint(self) -> None:
        state_dict = torch.hub.load_state_dict_from_url(str(self.CKPT_PATHS[self.backbone]))
        if self.backbone == "tiny_vit":
            for key in ["image_encoder.norm_head.weight", "image_encoder.norm_head.bias", "image_encoder.head.weight", "image_encoder.head.bias"]:
                state_dict.pop(key)
        self.load_state_dict(state_dict)
        print(f"[*] Complete to load checkpoint from {self.CKPT_PATHS[self.backbone]}.")

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x