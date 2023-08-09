"""Zero-shot inferencer for visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.utils import shuffle
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.torch.datasets.pipelines.sam_transforms import (
    ResizeLongestSide,
)


class SamPredictor:
    """Predictor for SAM.

    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    """
    
    def __init__(
        self,
        sam_model: nn.Module,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
            sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

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
        if image_format != self.model.image_format:
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
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        
        if cal_image:
            self.reset_image()
            self.original_size = original_image_size
            self.input_size = tuple(transformed_image.shape[-2:])
            input_image = self.model.preprocess(transformed_image)
            self.features = self.model.image_encoder(input_image)
            self.is_image_set = True

        if transformed_mask is not None:
            input_mask = self.model.preprocess(transformed_mask)  # pad to 1024
            return input_mask

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
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        # Upscale the masks to the original image resolution
        high_res_masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = high_res_masks > self.model.mask_threshold  # 0.0
            return masks, iou_predictions, low_res_masks, high_res_masks 
        else:
            return high_res_masks, iou_predictions, low_res_masks, high_res_masks 
        

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
        

class ZeroShotLearningInferencer:
    """Inferencer for PerSAM zero-shot learning."""
    def __init__(
        self,
        backbone: str,
        use_attn_sim: bool = False,
        device: str = "cuda"
    ):
        self.predictor = SamPredictor(PerSAM(backbone=backbone, device=device))
        self.use_attn_sim = use_attn_sim
        self.threshold_ref: float = 0.3
        
        self._initialize_reference()

    def _initialize_reference(self):
        self.target_feats: List[torch.Tensor] = []
        self.target_embeddings: List[torch.Tensor] = []
        self.ref_logit = None
        self.ref_masks = defaultdict(list)

    def _generate_ref_feature_mask(self, ref_feat: torch.Tensor, ref_mask: torch.Tensor) -> Tuple[torch.Tensor]:
        """Generate reference features.

        It should be run whenever the masks are predicted to get reference features corresponding to the masks.
        
        Args:
            ref_feat (torch.Tensor):
            ref_mask (torch.Tensor):

        Returns:
            (torch.Tensor): 
            (torch.Tensor):
        """
        # Post-process ref_mask
        ref_mask = F.interpolate(ref_mask.unsqueeze(0).unsqueeze(0), size=self.predictor.input_size, mode="bilinear").squeeze()
        ref_mask = self.predictor.model.preprocess_mask(ref_mask)
        ref_mask = F.interpolate(ref_mask.unsqueeze(0).unsqueeze(0), size=ref_feat.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        target_feat = ref_feat[ref_mask > self.threshold_ref]
        target_embedding = target_feat.mean(0).unsqueeze(0)    
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)
        
        return target_feat, target_embedding

    def _generate_prompt_info(self, test_feat: torch.Tensor, test_image: np.ndarray, topk: int = 0) -> Tuple[List]:
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        
        num_classes = len(self.target_feats)
        # Positive-negative location prior
        topk_xys = []
        topk_labels = []
        attn_sims = []
        for ref_feat in self.target_feats:
            sim= ref_feat @ test_feat
            sim = sim.reshape(1, 1, h, w)
            sim = self.predictor.model.postprocess_masks(
                            sim,
                            input_size=self.predictor.input_size,
                            original_size=self.predictor.original_size).squeeze()

            threshold = 0.85 * sim.max() if num_classes > 1 else 0.65
            topk_xy_i, topk_label_i = self._point_selection(sim, test_image, topk=topk, threshold=threshold)
            topk_xys.append(topk_xy_i)
            topk_labels.append(topk_label_i)

            if self.use_attn_sim:
                # Obtain the target guidance for cross-attention layers
                sim = (sim - sim.mean()) / torch.std(sim)
                sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear")
                attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
                attn_sims.append(attn_sim)

        return topk_xys, topk_labels, attn_sims

    def _point_selection(
        self,
        mask_sim: torch.Tensor,
        test_image: np.ndarray,
        topk: int = 1,
        threshold: float = 0.8
    ) -> Tuple[Dict, Dict]:
        # Top-1 point selection
        h, w = mask_sim.shape
        topk_xy = {}
        topk_label = {}
        last_xy = None
        last_xy = mask_sim.flatten(0).topk(1, largest=False)[1]
        last_x = (last_xy // w).unsqueeze(0)
        last_y = (last_xy - last_x * w)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_xy = last_xy.cpu().numpy()

        if topk > 0:
            # Top-last point selection
            topk_xy = mask_sim.flatten(0).topk(topk)[1]
            topk_x = (topk_xy // w).unsqueeze(0)
            topk_y = (topk_xy - topk_x * w)
            topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
            topk_label[0] = np.array([1] * topk)
            topk_xy[0] = topk_xy.cpu().numpy()
            topk_xy[0].append([last_xy[0][0], last_xy[0][1]])
            topk_label[0].append(0)

        else:
            sim_points = (mask_sim >= threshold)
            np_xy = torch.nonzero(sim_points*mask_sim)
            np_xy = shuffle(np_xy.cpu().detach().numpy(), random_state=0)

            h, w, _ = test_image.shape
            max_len = h
            if max_len < w:
                max_len = w

            ratio = self.predictor.input_size / max_len
            h = int(h * ratio)
            w = int(w * ratio)
            n_w = w // 16
            for i in range(len(np_xy)):
                x = np_xy[i][1]
                y = np_xy[i][0]
                key = int((int(y*ratio)//16)*n_w) + int(x*ratio)//16
                if key not in topk_xy.keys():
                    topk_xy[key] = [[x,y]]
                    topk_label[key] = [1]
                elif len(topk_xy[key]) < 1:
                    topk_xy[key].append([x,y])
                    topk_label[key].append(1)

            for i in topk_label.keys():
                topk_xy[i].append([last_xy[0][0], last_xy[0][1]])
                topk_label[i].append(0)
            
        return topk_xy, topk_label

    def _predict_mask(
        self,
        prompt_points: torch.Tensor,
        prompt_labels: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # First-step prediction
        if attention is not None and embedding is not None:
            masks, scores, logits, _ = self.predictor.predict(
                point_coords=prompt_points, 
                point_labels=prompt_labels, 
                multimask_output=False,
                attn_sim=attention,  # Target-guided Attention
                target_embedding=embedding  # Target-semantic Prompting
            )
        else:
            masks, scores, logits, _ = self.predictor.predict(
                point_coords=prompt_points, 
                point_labels=prompt_labels, 
                multimask_output=False
            )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = self.predictor.predict(
                    point_coords=prompt_points,
                    point_labels=prompt_labels,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        
        best_idx = np.argmax(scores)
        
        return masks[best_idx]

    def _infer_reference_prediction(
        self,
        images: np.ndarray,
        prompts: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reference prediction.

        Reference prediction using given prompts. These results will be saved at `results_reference` in result dict.
        
        Args:
            images (np.ndarray): Reference image to be predicted by using given prompts.
            prompts (list): List of multi class prompts.
            params (dict): Parameters for reference prediction.
                - reset_feat : If True, reset reference features using a new given image and prompts, defaults to False.
                - do_ref_seg : If True, referring segmentation using the source image will be executed, defaults to False.
        """
        if params.get("reset_feat", False):
            print(f"[*] Reinitialize reference image & feature.")
            self._initialize_reference()

        self.predictor.set_image(images)
        for prompt in prompts:
            if not (prompt.get("point_coords", None) is not None and prompt.get("point_labels", None) is not None):
                print((
                    f"If point_coords is set, point_labels must be also set: "
                    f"point_coords: {prompt.get('point_coords', None)}\n"
                    f"point_labels: {prompt.get('point_labels', None)}"
                ))
                continue

            if not (prompt.get("point_coords", None) is not None or prompt.get("box", None) is not None):
                print((
                    f"There should be at least one of point_coords and box: "
                    f"point_coords: {prompt.get('point_coords', None)}\n"
                    f"box: {prompt.get('box')}"
                ))
                continue

            input_prompts = {
                "point_coords": prompt.get("point_coords", None),
                "point_labels": prompt.get("point_labels", None),
                "box": prompt.get("box", None),
                "mask_input": self.ref_logit
            }
            masks, scores, logits, _ = self.predictor.predict(**input_prompts, multimask_output=True)
            best_idx = np.argmax(scores)

            ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
            ref_mask = torch.tensor(masks[best_idx], dtype=torch.float32)
            target_feat, target_embedding = self._generate_ref_feature_mask(ref_feat, ref_mask)

            self.target_feats.append(target_feat)
            self.target_embeddings.append(target_embedding)

            self.ref_logit = logits[best_idx][None] if params.get("use_logit", False) else None
            self.ref_masks["results_reference"].append(masks[best_idx])

        if params.get("do_ref_seg", True):
            self.ref_masks["results_referring"] += self._infer_referring_segmentation([images])[0]
        return self.ref_masks

    def _infer_referring_segmentation(self, images: List[np.ndarray]) -> List[List[torch.Tensor]]:
        """Referring segmentation using reference features.
        
        Args:
            images (list):
        """
        total_best_masks = []
        for image in images:
            self.predictor.set_image(image)
            test_feat = self.predictor.features.squeeze()
            topk_xys, topk_labels, attn_sims = self._generate_prompt_info(test_feat, image)
            best_masks = []
            for i in range(len(topk_xys)):
                best_mask = None
                for j in topk_xys[i].keys():
                    inputs = dict(
                        prompt_points=np.array(topk_xys[i].get(j)),
                        prompt_labels=np.array(topk_labels[i].get(j)),
                    )
                    if best_mask is not None and best_mask[inputs["prompt_points"][0][1], inputs["prompt_points"][0][0]] > 0:
                        # if given prompt value in best_mask is already assigned
                        continue

                    if self.use_attn_sim:
                        inputs.update(dict(
                            attention=attn_sims[i],
                            embedding=self.target_embeddings[i],
                        ))
                    mask = self._predict_mask(**inputs)
                    if best_mask is None:
                        best_mask = mask
                    else:
                        best_mask += mask
                best_masks.append(best_mask)
            total_best_masks.append(best_masks)
        return total_best_masks

    def infer(self, images: Union[List[Any], np.ndarray, Image.Image], params: Dict[str, Any], prompts: Optional[Dict[str, Any]] = None) -> Any:
        """Inference for zero-shot learning using PerSAM.
        
        This inference supports three inference types:
            1. Reference prediction
                - Basic visual prompting inference to given reference image(s) using given prompts
            2. Referring segmentation
                - Inference to other given test image(s) using reference feature
        """
        if isinstance(images, Image.Image):
            images = np.array(images, dtype=np.uint8)
        if isinstance(images, list) and any(not isinstance(image, np.ndarray) for image in images):
            images = [np.array(image) for image in images]

        if params.get("type") == "ref_pred":
            assert isinstance(images, np.ndarray), f"images must be np.ndarray, given {type(images)}."
            return self._infer_reference_prediction(images, prompts, params)
        
        elif params.get("type") == "refer_seg":
            return self._infer_referring_segmentation(images)
