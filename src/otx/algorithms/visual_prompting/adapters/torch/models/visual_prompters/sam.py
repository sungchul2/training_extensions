"""Segment Anything Model for zero-shot inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional, Tuple

import numpy as np
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
        "tiny_vit": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
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
        self.mask_threshold = 0.

        self.set_image_encoder()
        self.set_prompt_encoder()
        self.set_mask_decoder()
        self.load_checkpoint()

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.eval()
        self.to(device)

        self.transform = ResizeLongestSide(self.image_size)
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

    def set_image(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        image_format: str = "RGB",
        cal_image=True
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
            image (np.ndarray): The image for calculating masks. Expects an
                image in HWC uint8 format, with pixel values in [0, 255].
            image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image, self.transform.target_length)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Transform the mask to the form expected by the model
        input_mask_torch = None
        if mask is not None:
            input_mask = self.transform.apply_image(mask)
            input_mask_torch = torch.as_tensor(input_mask, device=self.device)
            input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_mask = self.set_torch_image(input_image_torch, image.shape[:2], transformed_mask=input_mask_torch)
        return input_mask

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
        transformed_mask: torch.Tensor = None,
        cal_image=True
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
            transformed_image (torch.Tensor): The input image, with shape
                1x3xHxW, which has been transformed with ResizeLongestSide.
            original_image_size (tuple(int, int)): The size of the image
                before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.image_encoder.img_size}."
        
        if cal_image:
            self.reset_image()
            self.original_size = original_image_size
            self.input_size = tuple(transformed_image.shape[-2:])
            input_image = self.preprocess(transformed_image)
            self.features = self.image_encoder(input_image)
            self.is_image_set = True

        if transformed_mask is not None:
            input_mask = self.preprocess(transformed_mask)  # pad to 1024
            return input_mask

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        attn_sim = None,
        target_embedding = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
            point_coords (np.ndarray or None): A Nx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
            point_labels (np.ndarray or None): A length N array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            box (np.ndarray or None): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form 1xHxW, where
                for SAM, H=W=256.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
            (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (np.ndarray): An array of length C containing the model's
                predictions for the quality of each mask.
            (np.ndarray): An array of shape CxHxW, where C is the number
                of masks and H=W=256. These low resolution logits can be passed to
                a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks, high_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
            attn_sim=attn_sim,
            target_embedding=target_embedding,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        high_res_masks = high_res_masks[0]

        return masks, iou_predictions, low_res_masks, high_res_masks

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        attn_sim = None,
        target_embedding = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
            point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
            point_labels (torch.Tensor or None): A BxN array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            boxes (np.ndarray or None): A Bx4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
            (torch.Tensor): The output masks in BxCxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (torch.Tensor): An array of shape BxC containing the model's
                predictions for the quality of each mask.
            (torch.Tensor): An array of shape BxCxHxW, where C is the number
                of masks and H=W=256. These low res logits can be passed to
                a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        # Upscale the masks to the original image resolution
        high_res_masks = self.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = high_res_masks > self.mask_threshold  # 0.0
            return masks, iou_predictions, low_res_masks, high_res_masks 
        else:
            return high_res_masks, iou_predictions, low_res_masks, high_res_masks 

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
