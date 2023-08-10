"""Zero-shot learning processor for visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.torch.models.visual_prompters import SAM


class ZeroShotLearningProcessor:
    """Processor for SAM zero-shot learning."""
    def __init__(
        self,
        backbone: str,
        use_attn_sim: bool = False,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.6,
        device: str = "cuda"
    ):
        self.model = SAM(backbone=backbone, device=device)
        self.use_attn_sim = use_attn_sim
        self.default_threshold_reference = default_threshold_reference  # immutable
        self.default_threshold_target = default_threshold_target
        self.threshold_target: float
        
        self._initialize_reference()

    def _initialize_reference(self) -> None:
        self.target_feats: List[torch.Tensor] = []
        self.target_embeddings: List[torch.Tensor] = []
        self.ref_logit = None

    def _generate_ref_feature_mask(self, ref_feat: torch.Tensor, ref_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
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
        ref_mask = F.interpolate(ref_mask.unsqueeze(0).unsqueeze(0), size=self.model.input_size, mode="bilinear").squeeze()
        ref_mask = self.model.preprocess_mask(ref_mask)
        ref_mask = F.interpolate(ref_mask.unsqueeze(0).unsqueeze(0), size=ref_feat.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        target_feat = ref_feat[ref_mask > self.default_threshold_reference]
        target_embedding = target_feat.mean(0).unsqueeze(0)    
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)
        
        return target_feat, target_embedding

    def _generate_prompt_info(self, test_feat: torch.Tensor, test_image: np.ndarray, topk: int = 0) -> Tuple[List, ...]:
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
            sim = self.model.postprocess_masks(
                sim,
                input_size=self.model.input_size,
                original_size=self.model.original_size).squeeze()

            # threshold = 0.85 * sim.max() if num_classes > 1 else 0.65
            topk_xy_i, topk_label_i = self._point_selection(sim, test_image, topk=topk, threshold=self.threshold_target)
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
    ) -> Tuple[Dict, ...]:
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

            ratio = self.model.image_size / max_len
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
            masks, scores, logits, _ = self.model.predict(
                point_coords=prompt_points, 
                point_labels=prompt_labels, 
                multimask_output=False,
                attn_sim=attention,  # Target-guided Attention
                target_embedding=embedding  # Target-semantic Prompting
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
                - reset_feat : Reset reference features using a new given image and prompts, defaults to True.
                - do_ref_seg : Referring segmentation to targets using the source image will be executed, defaults to True.
        """
        if params.get("reset_feat", True):
            print(f"[*] Reinitialize reference image & feature.")
            self._initialize_reference()

        self.model.set_image(images)
        ref_masks = defaultdict(list)
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
            masks, scores, logits, _ = self.model.predict(**input_prompts, multimask_output=True)
            best_idx = np.argmax(scores)

            ref_feat = self.model.features.squeeze().permute(1, 2, 0)
            ref_mask = torch.tensor(masks[best_idx], dtype=torch.float32)
            target_feat, target_embedding = self._generate_ref_feature_mask(ref_feat, ref_mask)

            self.target_feats.append(target_feat)
            self.target_embeddings.append(target_embedding)

            self.ref_logit = logits[best_idx][None] if params.get("use_logit", False) else None
            ref_masks["results_reference"].append(masks[best_idx])

        if params.get("do_ref_seg", True):
            ref_masks["results_target"] += self._infer_target_segmentation([images])[0]
        return ref_masks

    def _infer_target_segmentation(self, images: List[np.ndarray]) -> List[List[torch.Tensor]]:
        """Referring segmentation to targets using reference features.
        
        Args:
            images (list):
        """
        total_best_masks = []
        for image in images:
            self.model.set_image(image)
            test_feat = self.model.features.squeeze()
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
        """Inference for zero-shot learning using PerSAM inference logic.
        
        This inference supports three inference types:
            1. Reference prediction
                - Basic visual prompting inference to given reference image(s) using given prompts
            2. Referring segmentation (target segmentation)
                - Inference to other given test image(s) using reference feature
        """
        self.threshold_target = params.get("threshold", self.default_threshold_target)

        if params.get("type") == "reference":
            assert isinstance(images, (np.ndarray, Image.Image)), f"images must be np.ndarray or Image.Image, given {type(images)}."
            if isinstance(images, Image.Image):
                images = np.array(images, dtype=np.uint8)
            return self._infer_reference_prediction(images, prompts, params)
        
        elif params.get("type") == "target":
            assert isinstance(images, (tuple, list)), f"images must be wrapped by iterable instances, list or tuple, given {type(images)}."
            assert any(isinstance(image, (np.ndarray, Image.Image)) for image in images), f"images must be set of np.ndarray or Image.Image."
            if any(not isinstance(image, np.ndarray) for image in images):
                images = [np.array(image) for image in images]
            return self._infer_target_segmentation(images)
