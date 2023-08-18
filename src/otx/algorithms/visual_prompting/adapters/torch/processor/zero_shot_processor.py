"""Zero-shot learning processor for visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.nn import functional as F

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.torch.models.visual_prompters import (
    VisualPrompter,
)

logger = get_logger()

DEFAULT_SETTINGS = dict(
    reference=dict(
        reset_feat=False,
        do_target_seg=False,
        use_logit=False,
        target_params=dict(
            return_score=False,
        ),
    ),
    target=dict(
        mode="point_selection",
        return_score=False,
    ),
)


class ZeroShotLearningProcessor:
    """Processor for SAM zero-shot learning.
    
    Args:
        backbone (str): Backbone network to be used between tiny_vit, vit_b, vit_l, and vit_h.
        use_attn_sim (bool): Whether using attention similarity used at PerSAM, defaults to False.
        default_threshold_reference (float): Threshold used for reference features, defaults to 0.3.
        default_threshold_target (float): Threshold used for target point selection, defaults to 0.65.
        default_reference_path (str): Path to save and load reference features, defaults to reference_features.pth in the same dirctory with processor.
        device (str): Device setting, defaults to cuda.
    """
    def __init__(
        self,
        backbone: str,
        use_attn_sim: bool = False,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        default_reference_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_features.pth"),
        device: str = "cuda"
    ) -> None:
        self.model = VisualPrompter(type="sam", backbone=backbone, device=device)
        self.use_attn_sim = use_attn_sim
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target
        self.default_reference_path = default_reference_path
        self.threshold_target: float
        
        self._initialize_reference()

    def _initialize_reference(self) -> None:
        """Initialize reference information."""
        self.reference_feats: List[torch.Tensor] = []
        self.reference_embeddings: List[torch.Tensor] = []
        self.reference_logit = None

    def _generate_masked_features(self, feats: torch.Tensor, masks: torch.Tensor, threshold_mask: float) -> Tuple[torch.Tensor, ...]:
        """Generate masked features.
        
        Args:
            feats (torch.Tensor): Raw reference features. It will be filtered with masks.
            masks (torch.Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (torch.Tensor): Masked features.
            (torch.Tensor): Masked embeddings used for semantic prompting.
        """
        # Post-process masks
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=self.model.input_size, mode="bilinear").squeeze()
        masks = self.model.preprocess_mask(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_embedding = masked_feat.mean(0).unsqueeze(0)    
        masked_feat = masked_embedding / masked_embedding.norm(dim=-1, keepdim=True)
        masked_embedding = masked_embedding.unsqueeze(0)
        
        return masked_feat, masked_embedding

    def _point_selection_feature_matching(self, target_feat: torch.Tensor, target_image: np.ndarray, return_score: bool, topk: int = 0, manual_ref_feats: Optional[List[torch.Tensor]] = None) -> List[np.ndarray]:
        """Generate points, labels, and attention similarity which can be used for zero-shot inference.
        
        Args:
            target_feat (torch.Tensor): Target feature.
            target_image (np.ndarray): Target image.
            return_score (bool): Whether return prediction score for each instance or the final mask by threshold, defaults to False.
            manual_ref_feats (list, optional): Not to use all of generated reference features, defaults to None.
            
        Returns:
            (np.ndarray): Predicted mask along with similarity score.
        """
        # Cosine similarity
        c_feat, h_feat, w_feat = target_feat.shape
        h_img, w_img, _ = target_image.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)
        
        predicted_masks = []
        used_reference_features = self.reference_feats if manual_ref_feats is None else [manual_ref_feats]
        for reference_feats in used_reference_features:
            num_classes = len(reference_feats)
            # Positive-negative location prior
            predicted_mask = np.zeros((num_classes+1, h_img, w_img), dtype=np.float32)
            for i, ref_feat in enumerate(reference_feats):
                if ref_feat is None:
                    # empty class
                    continue

                sim = ref_feat @ target_feat
                sim = sim.reshape(1, 1, h_feat, w_feat)
                sim = self.model.postprocess_masks(
                    sim,
                    input_size=self.model.input_size,
                    original_size=self.model.original_size).squeeze()

                threshold = 0.85 * sim.max() if num_classes > 1 else self.default_threshold_target
                topk_points, topk_labels = self._point_selection(sim, h_img, w_img, topk=topk, threshold=threshold)
                for j in topk_points.keys():
                    prompt_points: List = []
                    prompt_labels: List = []
                    flag_fg: bool = False
                    for point, label in zip(topk_points.get(j), topk_labels.get(j)):
                        if label == 1 and predicted_mask[i+1][point[1], point[0]] > 0:
                        # Filter already assigned foreground prompts
                            continue

                        prompt_points.append(point)
                        prompt_labels.append(label)
                        if label == 1:
                            flag_fg = True

                    if not flag_fg:
                        # Skip if not using foreground prompts
                        continue

                    inputs = {
                        "prompt_points": np.array(prompt_points),
                        "prompt_labels": np.array(prompt_labels),
                    }
                    if self.use_attn_sim:
                        # Obtain the target guidance for cross-attention layers
                        attn_sim = (sim - sim.mean()) / sim.std()
                        attn_sim = F.interpolate(attn_sim.unsqueeze(0).unsqueeze(0), size=(h_feat, w_feat), mode="bilinear")
                        attn_sim = attn_sim.sigmoid_().unsqueeze(0).flatten(3)
                        inputs.update(dict(
                            attention=attn_sim,
                            embedding=self.reference_embeddings[i],
                        ))

                    mask = self._predict_mask(**inputs)
                    if return_score:
                        predicted_mask[i+1][mask] = np.mean([sim.detach().cpu().numpy()[point[1], point[0]] for point, label in zip(prompt_points, prompt_labels) if label == 1])
                    else:
                        predicted_mask[i+1] += mask.astype(np.float32)
                predicted_mask[i+1] = np.clip(predicted_mask[i+1], 0, 1)
            predicted_masks.append(predicted_mask)
        return predicted_masks

    def _point_selection(
        self,
        mask_sim: torch.Tensor,
        height: int,
        width: int,
        topk: int = 1,
        threshold: float = 0.8
    ) -> Tuple[Dict, ...]:
        """Select point used as point prompts."""
        # TODO (sungchul): refactoring

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
            # TODO (sungchul): sort coords by similarity score like top-k, not shuffling
            sim_points = (mask_sim >= threshold)
            np_xy = torch.nonzero(sim_points*mask_sim)
            np_xy = shuffle(np_xy.cpu().detach().numpy(), random_state=0)

            max_len = height
            if max_len < width:
                max_len = width

            ratio = self.model.image_size / max_len
            height = int(height * ratio)
            width = int(width * ratio)
            n_w = width // 16
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
    ) -> np.ndarray:
        """Predict target masks.
        
        Args:
            prompt_points (torch.Tensor): Selected points as point prompts from similarity map.
            prompt_labels (torch.Tensor): Labels that are set in foreground or background.
            attention (torch.Tensor, optional): Target-guided attention used at PerSAM.
            embedding (torch.Tensor, optional): Target-semantic Prompting used at PerSAM.

        Return:
            (np.ndarray): Predicted mask.
        """
        # First-step prediction
        if attention is not None and embedding is not None:
            masks, scores, logits, _ = self.model.predict(
                point_coords=prompt_points, 
                point_labels=prompt_labels, 
                multimask_output=False,
                attn_sim=attention,  # Target-guided Attention
                reference_embedding=embedding  # Target-semantic Prompting
            )
        else:
            masks, scores, logits, _ = self.model.predict(
                point_coords=prompt_points, 
                point_labels=prompt_labels, 
                multimask_output=False
            )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = self.model.predict(
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
        masks, scores, logits, _ = self.model.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        
        best_idx = np.argmax(scores)
        
        return masks[best_idx]

    def _auto_generation_feature_matching(self, target_feat: torch.Tensor, target_image: np.ndarray, return_score: bool, manual_ref_feats: Optional[List[torch.Tensor]] = None) -> List[np.ndarray]:
        """Predict target masks using auto generation.
        
        Args:
            target_feat (torch.Tensor): Target feature.
            target_image (np.ndarray): Target image.
            return_score (bool): Whether return prediction score for each instance or the final mask by threshold, defaults to False.
            manual_ref_feats (list, optional): Not to use all of generated reference features, defaults to None.
            
        Returns:
            (np.ndarray): Predicted mask along with similarity score.
        """
        auto_gen_masks = self.model.auto_generator.generate(target_image)
        predicted_masks = []
        used_reference_features = self.reference_feats if manual_ref_feats is None else [manual_ref_feats]
        for reference_feats in used_reference_features:
            num_classes = len(reference_feats)
            predicted_mask = np.zeros((num_classes+1,) + target_image.shape[:2], dtype=np.float32)
            for i, ref_feat in enumerate(reference_feats):
                for auto_gen_mask in auto_gen_masks:
                    target_mask = torch.tensor(auto_gen_mask["segmentation"], dtype=torch.float32)
                    masked_target_feat, _ = self._generate_masked_features(target_feat, target_mask, 0.5)
                    if masked_target_feat is None:
                        continue

                    masked_target_feat = masked_target_feat.permute(1, 0)
                    sim = ref_feat @ masked_target_feat
                    if return_score:
                        predicted_mask[i+1][auto_gen_mask["segmentation"]] = sim.detach().cpu().numpy()[0,0]
                    else:
                        if sim >= self.default_threshold_target:
                            predicted_mask[i+1] += auto_gen_mask["segmentation"].astype(np.float32)
                predicted_mask[i+1] = np.clip(predicted_mask[i+1], 0, 1)
            predicted_mask.append(predicted_mask)
        return predicted_masks

    def _preprocess_prompts(
        self,
        prompts: List[Dict[str, Any]],
        height: int,
        width: int,
        num_sample: int = 5,
        convert_mask: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Preprocess prompts.

        This function proceeds such below thigs:
            1. Gather prompts which have the same labels
            2. If there are polygon prompts, convert them to a mask and randomly sample points
            3. If there are box prompts, the key `point_coords` is changed to `box`
        
        Args:
            prompts (list): Given prompts to be processed.
            height (int): Image height.
            width (int): Image width.
            num_sample (int): The number of points to be sampled in a mask generated from given polygon, defaults to 5.
            convert_mask (bool): Whether converting polygon to mask, defaults to False.
            
        Returns:
            (dict): Processed and arranged prompts using label information as keys.
                processed_prompts = {
                    0: { # background
                        "point_coords": np.ndarray(),
                        "point_labels": np.ndarray(),
                        "box": np.ndarray(),
                    },
                    1: {
                        "point_coords": np.ndarray(),
                        "point_labels": np.ndarray(),
                        "box": np.ndarray(),
                    },
                    2: {
                        "point_coords": np.ndarray(),
                        "point_labels": np.ndarray(),
                        "box": np.ndarray(),
                    }
                }
        """
        processed_prompts: Dict[str, Dict[str, Any]] = {}
        for prompt in prompts:
            if prompt.get("label", 0) not in processed_prompts:
                # initialize
                processed_prompts[prompt.get("label", 0)] = defaultdict(dict)

            if prompt.get("type") == "point":
                self._update_value(processed_prompts[prompt.get("label", 0)], "point_coords", prompt.get("point_coords"))
                self._update_value(processed_prompts[prompt.get("label", 0)], "point_labels", prompt.get("point_labels"))

            elif prompt.get("type") == "polygon":
                polygon = prompt.get("point_coords")
                if convert_mask:
                    # convert polygon to mask
                    contour = [[int(point[0]), int(point[1])] for point in polygon]
                    gt_mask = np.zeros((height, width), dtype=np.uint8)
                    gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)

                    # randomly sample points from generated mask
                    ys, xs, _ = np.nonzero(gt_mask)
                else:
                    ys, xs = polygon[:,1], polygon[:,0]

                rand_idx = np.random.permutation(len(ys))[:num_sample]
                _point_coords = []
                _point_labels = []
                for x, y in zip(xs[rand_idx], ys[rand_idx]):
                    _point_coords.append([x, y])
                    _point_labels.append(prompt.get("point_labels")[0])

                self._update_value(processed_prompts[prompt.get("label", 0)], "point_coords", np.array(_point_coords))
                self._update_value(processed_prompts[prompt.get("label", 0)], "point_labels", np.array(_point_labels))

            elif prompt.get("type") == "box":
                self._update_value(processed_prompts[prompt.get("label", 0)], "box", prompt.get("point_coords"))

        processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x[0]))
        return processed_prompts

    def _update_value(self, target: Dict[str, Any], key: str, value: np.ndarray) -> None:
        """Update numpy value to target dictionary."""
        if key in target:
            target[key] = np.concatenate((target[key], value))
        else:
            target[key] = value

    def _merge_prompts(self, label: int, input_prompts: Dict[str, Any], processed_prompts: Dict[str, Any], use_only_background: bool = True) -> Dict[str, Any]:
        """Merge target prompt and other prompts.

        Merge a foreground prompt and other prompts (background or prompts with other classes).
        
        Args:
            label (int): Label information. Background is 0 and other foregrounds are >= 0.
            input_prompts (dict): A foreground prompt to be merged with other prompts.
            processed_prompts (dict): The whole class-wise prompts processed at _preprocess_prompts.
            use_only_background (bool): Whether merging only background prompt, defaults to True. It is applied to only point_coords.
        """
        merged_input_prompts = deepcopy(input_prompts)
        for other_label, other_input_prompts in processed_prompts.items():
            if other_label == label:
                continue
            if (use_only_background and other_label == 0) or (not use_only_background):
                # only add point (and scribble) prompts
                # use_only_background=True -> background prompts are only added as background
                # use_only_background=False -> other prompts are added as background
                if "point_coords" in other_input_prompts:
                    # point, scribble
                    self._update_value(merged_input_prompts, "point_coords", other_input_prompts.get("point_coords"))
                    self._update_value(merged_input_prompts, "point_labels", np.zeros_like(other_input_prompts.get("point_labels")))

        return merged_input_prompts

    def _infer_visual_prompt(
        self,
        images: List[np.ndarray],
        prompts: List[List[Dict[str, Any]]],
    ) -> Dict[str, np.ndarray]:
        """Base visual prompt prediction.
        
        Args:
            images (list): List of images to be predicted by using given prompts.
            prompts (list): List of multi class prompts. Both background prompt and foreground prompt have similar format,
                but foreground prompt has label information, too. This given prompts will be processed at _preprocess_prompts.
                Foreground prompts have `1` as point_labels and background prompts have `0` as point_labels.

                [Example]
                prompts = [
                    [
                        {
                            "type": "point",
                            "point_coords": np.array([[100, 200]]),
                            "point_labels": np.array([1]),
                            "label": 1
                        }, # foreground which has label 1 and type is point
                        {
                            "type": "point",
                            "point_coords": np.array([[200, 200]]),
                            "point_labels": np.array([0]),
                        }, # background
                        {
                            "type": "polygon",
                            "point_coords": np.array([[100, 300], [102, 298], ...]), # polygon of a scribble
                            "point_labels": np.array([1]),
                            "label": 2
                        }, # foreground which has label 2 and type is polygon (scribble)
                        {
                            "type": "box",
                            "point_coords": np.array([[100, 300], [400, 400]]), # (x1, y1), (x2, y2)
                            "point_labels": np.array([1]),
                            "label": 1
                        }, # foreground which has label 1 but different prompt
                    ], # for the first reference image
                    [...], # for the second reference image
                    ...
                ]

        Returns:
            (dict): Base visual prompting results.
        """
        results_visual_prompts = []
        for image, prompt in zip(images, prompts):
            height, width, _ = image.shape
            self.model.set_image(image)

            processed_prompts = self._preprocess_prompts(prompt, height, width)
            num_classes = max(processed_prompts.keys()) + 1
            results_visual_prompt = np.zeros((1, height, width))
            for label in range(num_classes):
                if label == 0:
                    # background
                    continue

                if label not in processed_prompts:
                    # for empty class
                    results_visual_prompt = np.concatenate((results_visual_prompt, np.zeros((1, height, width))), axis=0)
                    continue

                input_prompts = processed_prompts.get(label)
                merged_input_prompts = self._merge_prompts(label, input_prompts, processed_prompts)
                merged_input_prompts.update({"mask_input": self.reference_logit})
                masks, scores, _, _ = self.model.predict(**merged_input_prompts, multimask_output=True)
                best_idx = np.argmax(scores)

                results_visual_prompt = np.concatenate((results_visual_prompt, masks[best_idx][None]), axis=0)
            results_visual_prompts.append(results_visual_prompt)
        return {"results_visual_prompt": results_visual_prompts}

    def _infer_reference_prediction(
        self,
        images: List[np.ndarray],
        prompts: List[List[Dict[str, Any]]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reference prediction.

        Reference prediction using given prompts. These results will be saved at `results_reference` in result dict.
        
        Args:
            images (list): List of reference images to be predicted by using given prompts. Currently, single reference is only supported.
            prompts (list): List of multi class prompts. Both background prompt and foreground prompt have similar format,
                but foreground prompt has label information, too. This given prompts will be processed at _preprocess_prompts.
                Foreground prompts have `1` as point_labels and background prompts have `0` as point_labels.

                [Example]
                prompts = [
                    [
                        {
                            "type": "point",
                            "point_coords": np.array([[100, 200]]),
                            "point_labels": np.array([1]),
                            "label": 1
                        }, # foreground which has label 1 and type is point
                        {
                            "type": "point",
                            "point_coords": np.array([[200, 200]]),
                            "point_labels": np.array([0]),
                        }, # background
                        {
                            "type": "polygon",
                            "point_coords": np.array([[100, 300], [102, 298], ...]), # polygon of a scribble
                            "point_labels": np.array([1]),
                            "label": 2
                        }, # foreground which has label 2 and type is polygon (scribble)
                        {
                            "type": "box",
                            "point_coords": np.array([[100, 300], [400, 400]]), # (x1, y1), (x2, y2)
                            "point_labels": np.array([1]),
                            "label": 1
                        }, # foreground which has label 1 but different prompt
                    ], # for the first reference image
                    [...], # for the second reference image
                    ...
                ]

            params (dict): Parameters for reference prediction.
                - reset_feat : Reset reference features using a new given image and prompts, defaults to False.
                - do_target_seg : Referring segmentation to targets using the source image will be executed, defaults to False.
                - use_logit: Use previous logits at the next inference, defaults to False.

        Returns:
            (dict): Reference prediction results.
        """
        if params.get("reset_feat", DEFAULT_SETTINGS["reference"]["reset_feat"]):
            logger.info(f"[*] Reinitialize reference image & feature.")
            self._initialize_reference()

        reference_results = defaultdict(list)
        for image, prompt in zip(images, prompts):
            height, width, _ = image.shape
            self.model.set_image(image)
            processed_prompts = self._preprocess_prompts(prompt, height, width)
            num_classes = max(processed_prompts.keys()) + 1

            reference_feats = []
            reference_embeddings = []
            results_reference = []
            for label in range(num_classes):
                if label == 0:
                    # background
                    continue

                if label not in processed_prompts:
                    # for empty class
                    reference_feats.append(None)
                    reference_embeddings.append(None)
                    results_reference.append(np.zeros((height, width)))
                    continue

                input_prompts = processed_prompts.get(label)
                merged_input_prompts = self._merge_prompts(label, input_prompts, processed_prompts)
                merged_input_prompts.update({"mask_input": self.reference_logit})
                masks, scores, logits, _ = self.model.predict(**merged_input_prompts, multimask_output=True)
                best_idx = np.argmax(scores)

                ref_feat = self.model.features.squeeze().permute(1, 2, 0)
                ref_mask = torch.tensor(masks[best_idx], dtype=torch.float32)
                reference_feat = None
                default_threshold_reference = self.default_threshold_reference
                while reference_feat is None:
                    logger.info(f"[*] default_threshold_reference : {default_threshold_reference}")
                    reference_feat, reference_embedding = self._generate_masked_features(ref_feat, ref_mask, self.default_threshold_reference)
                    default_threshold_reference -= 0.1

                reference_feats.append(reference_feat)
                reference_embeddings.append(reference_embedding)
                results_reference.append(masks[best_idx])
                if params.get("use_logit", DEFAULT_SETTINGS["reference"]["use_logit"]):
                    self.reference_logit = logits[best_idx][None]

            self.reference_feats.append(reference_feats)
            self.reference_embeddings.append(reference_embeddings)
            reference_results["results_reference"].append(
                np.concatenate((
                    np.zeros((1, height, width)), np.stack(results_reference, axis=0)
                ), axis=0)
            )

            self.model.reset_image()

        if params.get("do_target_seg", DEFAULT_SETTINGS["reference"]["do_target_seg"]):
            for image, manual_ref_feats in zip(images, self.reference_feats):
                reference_results["results_target"] += self._infer_target_segmentation(
                    [image], params.get("target_params", DEFAULT_SETTINGS["reference"]["target_params"]), manual_ref_feats)

        return reference_results

    def _infer_target_segmentation(self, images: List[np.ndarray], params: Dict[str, Any], manual_ref_feats: Optional[List[torch.Tensor]] = None) -> List[List[np.ndarray]]:
        """Referring segmentation to targets using reference features.
        
        Args:
            images (list): List of target images.
            params (dict): Parameters for target segmentation.
            manual_ref_feats (list, optional): Not to use all of generated reference features, defaults to None.

        Returns:
            (list): List of results of each target image. Each result is a mask with CxHxW shape.
        """
        if len(params) == 0:
            logger.info((
                f"[*] Use default settings: "
                f"{DEFAULT_SETTINGS['target']}"
            ))
        mode = params.get("mode", DEFAULT_SETTINGS["target"]["mode"])
        return_score = params.get("return_score", DEFAULT_SETTINGS["target"]["return_score"])
        total_predicted_masks = []
        for image in images:
            self.model.set_image(image)
            target_feat = self.model.features.squeeze()
            if mode == "auto_generation":
                predicted_masks = self._auto_generation_feature_matching(target_feat.permute(1, 2, 0), image, return_score, manual_ref_feats=manual_ref_feats)
            elif mode == "point_selection":
                predicted_masks = self._point_selection_feature_matching(target_feat, image, return_score, manual_ref_feats=manual_ref_feats)
            else:
                continue
            total_predicted_masks.append(predicted_masks)
        return total_predicted_masks

    def __inspect_image_format(self, images: List[Union[np.ndarray, Image.Image]]) -> List[np.ndarray]:
        assert isinstance(images, (tuple, list)), f"images must be wrapped by iterable instances, list or tuple, given {type(images)}."
        assert any(isinstance(image, (np.ndarray, Image.Image)) for image in images), f"images must be set of np.ndarray or Image.Image."
        if any(not isinstance(image, np.ndarray) for image in images):
            images = [np.array(image, dtype=np.uint8) for image in images]
        return images

    def infer(self, images: List[Union[np.ndarray, Image.Image]], params: Dict[str, Any], prompts: Optional[List[List[Dict[str, Any]]]] = None) -> Any:
        """Inference for zero-shot learning.
        
        This inference supports two inference types:
            1. Base visual prompting
                - Basic visual prompting inference to given image using given prompts
            2. Referring segmentation (target segmentation)
                - Inference to other given test image(s) using reference feature
        """
        images = self.__inspect_image_format(images)
        if params.get("type") == "base":
            return self._infer_visual_prompt(images, prompts)
        
        elif params.get("type") == "target":
            return self._infer_target_segmentation(images, params)

    def learn(self, images: List[Union[np.ndarray, Image.Image]], params: Dict[str, Any], prompts: Optional[List[List[Dict[str, Any]]]] = None) -> Any:
        """Learn reference features for zero-shot learning."""
        images = self.__inspect_image_format(images)
        if params.get("type") == "reference":
            return self._infer_reference_prediction(images, prompts, params)
        else:
            raise ValueError(f"type for `learn` must be `reference`. Current: {params.get('type')}")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save reference features and embeddings.
        
        Args:
            path (str, Path, optional): Path to save reference features and embeddings, defaults to None.
        """
        assert len(self.reference_feats) > 0 and len(self.reference_embeddings) > 0, "reference features must be generated before saving."
        if path is None:
            path = self.default_reference_path
        
        _, extension = os.path.splitext(path)
        if extension in (".pth", ".pt"):
            features = {"reference_feats": self.reference_feats, "reference_embeddings": self.reference_embeddings}
            torch.save(features, path)
        else:
            raise ValueError(f"{extension} is not supported to save features and embeddings. Use .pth/.pt")

        logger.info(f"Saved at {path}.")

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load reference features and embeddings.
        
        Args:
            path (str, Path, optional): Path to load reference features and embeddings, defaults to None.
        """
        if path is None:
            path = self.default_reference_path

        _, extension = os.path.splitext(path)
        if extension in (".pth", ".pt"):
            features = torch.load(path)
            for k, v in features.items():
                if len(getattr(self, k)) > 0:
                    logger.warning(f"Because {k} is already set, it will be overwritten with loaded {k}.")
                setattr(self, k, v)
        else:
            raise ValueError(f"{extension} is not supported to load features and embeddings. Use .pth/.pt")

        logger.info(f"Loaded at {path}.")
