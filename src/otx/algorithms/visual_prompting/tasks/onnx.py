import os
from typing import Tuple

import cv2
import numpy as np
import onnxruntime


class SAMPreprocessor:
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length
        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def forward(self, image: np.ndarray, is_image: bool):
        image = self.apply_resize(image)
        if is_image:
            image = self.apply_normalize(image)
        image = self.apply_pad(image)
        return image
        
    def apply_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize input image to target shape.

        Args:
            image (np.ndarray): 

        Returns:
            np.ndarray: Resized image to target shape.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        h, w, _ = image.shape
        target_size = self.get_preprocess_shape(h, w)
        image = cv2.resize(image, target_size, 0, 0, interpolation=cv2.INTER_LINEAR)
        return image

    def apply_normalize(self, image: np.ndarray) -> np.ndarray:
        # Normalize colors
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def apply_pad(self, image: np.ndarray) -> np.ndarray:
        # Pad
        h, w = image.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        image = np.pad(image, ((0, padh), (0, padw)))
        return image

    def get_preprocess_shape(self, oldh: int, oldw: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Args:
            oldh (int): Original height.
            oldw (int): Original width.

        Returns:
            Tuple[int, int]: Target shape as (width, height) to be used at cv2.resize.
        """
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (neww, newh)
    

class SAMEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model # encoder

    def forward(self, image: torch.Tensor):
        # image : BxCxHxW
        image_embeddings = self.model(image)
        return image_embeddings


class SAMDecoder(SamOnnxModel): ...


class PerSAMFeatureExtractor(nn.Module):
    def forward(self, image_embedding: torch.Tensor, ref_mask: torch.Tensor):
        ref_feat = image_embedding.squeeze().permute(1, 2, 0)

        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)
        return target_embedding, target_feat


class PerSAMSimCoordGetter(nn.Module):
    def __init__(self, target_embedding):
        super().__init__()
        self.target_embedding = target_embedding # mean feature
        self.target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

    def forward(self, image_embeddings, input_size, orig_im_size):
        attn_sim, point_coords, point_labels = self.get_attn_sim_coords(image_embeddings, input_size, orig_im_size)
        return attn_sim, point_coords, point_labels, self.target_embedding.unsqueeze(0)

    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = torch.tensor([1] * topk)
            
        # Top-last point selection
        last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = torch.tensor([0] * topk)
        
        return topk_xy, topk_label, last_xy, last_label

    def get_attn_sim_coords(self, image_embeddings, input_size, original_size):
        test_feat = image_embeddings.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat
        
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(sim, input_size=tuple(input_size), original_size=tuple(original_size)).squeeze()
        
        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection(sim, topk=1)
        topk_xy = torch.cat([topk_xy_i, last_xy_i], dim=0).unsqueeze(0)
        topk_label = torch.cat([topk_label_i, last_label_i], dim=0).unsqueeze(0)
        
        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
        return attn_sim, topk_xy, topk_label
    

class PerSAMDecoder(SamOnnxModel):
    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
        attn_sim: torch.Tensor,
        target_embedding: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks


class ONNXVisualPromptingInferencer:
    """Inferencer for PerSAM."""

    def __init__(self):
        self.sam_preprocessor_image = SAMPreprocessor(1024, is_image=True)
        self.sam_preprocessor_mask = SAMPreprocessor(1024, is_image=False)

        self.sam_encoder_session = onnxruntime.InferenceSession("./onnx/image_encoder.onnx")
        self.persam_decoder_session = onnxruntime.InferenceSession("./onnx/persam_decoder.onnx")

        self.exp = 215 # expand mask value to
        self.topk = 1 # choose topk points
        self.center = False # whether prompt with center
        self.large = False # whether choose largest mask for prompting after stage 1
        self.box_prompt = False # whether use box prompt
        self.threshold = 10 # the threshold for bounding box expansion

    def set_images(
        self,
        dataset: Dict[str, Any],
        ref_image: Optional[np.ndarray] = None,
        ref_mask: Optional[np.ndarray] = None
    ):
        """Set images for inference.
        
        Args:
            dataset (str): ...
            filename (str): File path. If the file is an image (ends with .png or .jpg),
                this file is set to reference image and other files in the same directory will be test images.
                If the file is a video, the first frame is set to reference image and subsequence frames are test images.
        """
        assert dataset.get("type").lower() in ["davis", "potato"]
        self.test_images = []
        self.test_masks = []
        self.ref_image = None
        self.ref_mask = None
        if dataset.get("type").lower() == "davis":
            test_dataset = DAVISTestDataset("../data/DAVIS", imset="2017/val.txt")
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
            for data in test_loader:
                class_name = data["info"]["name"][0]
                if class_name != dataset.get("class_name", "bike-packing"):
                    continue
                images = data["rgb"].cpu().numpy()
                masks = data["gt"][0].cpu().numpy()
                self.num_images = images.shape[1]
                self.num_obj = len(data["info"]["labels"][0])
                for i in range(self.num_images):
                    if i == 0 and (ref_image is None and ref_mask is None):
                        self.ref_image = images[0, i]
                        self.ref_mask = masks[:, i]
                    else:
                        self.test_images.append(images[0, i])
                        self.test_masks.append(masks[:, i])

        if dataset.get("type").lower() == "potato":
            for i, path in enumerate(sorted(os.listdir("../data/potato"))):
                img = cv2.imread(f"../data/potato/{path}")
                if i == 0 and (ref_image is None and ref_mask is None):
                    self.ref_image = img
                else:
                    self.test_images.append(img)

        if self.ref_image is None:
            self.ref_image = ref_image
            self.ref_mask = ref_mask
        self.original_size = np.array(self.ref_image.shape[:2])

    def resize_and_crop(self, soft_prediction: np.ndarray, original_size: np.ndarray, image_size: int) -> np.ndarray:
        """Resize and crop soft prediction.
    
        Args:
            soft_prediction (np.ndarray): Predicted soft prediction with HxW shape.
            original_size (np.ndarray): The original image size.
    
        Returns:
            final_soft_prediction (np.ndarray): Resized and cropped soft prediction for the original image.
        """
        resized_soft_prediction = cv2.resize(
            soft_prediction, (image_size, image_size), 0, 0, interpolation=cv2.INTER_LINEAR
        )
    
        prepadded_size = self.get_padded_size(original_size, image_size).astype(np.int64)
        resized_cropped_soft_prediction = resized_soft_prediction[..., : prepadded_size[0], : prepadded_size[1]]
    
        original_size = original_size.astype(np.int64)
        h, w = original_size
        final_soft_prediction = cv2.resize(
            resized_cropped_soft_prediction, (w, h), 0, 0, interpolation=cv2.INTER_LINEAR
        )
        return final_soft_prediction
    
    def get_padded_size(self, original_size: np.ndarray, longest_side: int) -> np.ndarray:
        """Get padded size from original size and longest side of the image.
    
        Args:
            original_size (np.ndarray): The original image size with shape Bx2.
            longest_side (int): The size of the longest side.
    
        Returns:
            transformed_size (np.ndarray): The transformed image size with shape Bx2.
        """
        original_size = original_size.astype(np.float32)
        scale = longest_side / np.max(original_size)
        transformed_size = scale * original_size
        transformed_size = np.floor(transformed_size + 0.5).astype(np.int64)
        return transformed_size
    
    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten().argsort()[-topk:]
        topk_x = (topk_xy // h)[None]
        topk_y = (topk_xy - topk_x * h)
        topk_xy = np.concatenate((topk_y, topk_x), axis=0).transpose(1, 0)
        topk_label = np.array([1] * topk)
            
        # Top-last point selection
        last_xy = mask_sim.flatten().argsort()[:topk]
        last_x = (last_xy // h)[None]
        last_y = (last_xy - last_x * h)
        last_xy = np.concatenate((last_y, last_x), axis=0).transpose(1, 0)
        last_label = np.array([0] * topk)
        
        return topk_xy, topk_label, last_xy, last_label
    
    def sigmoid(self, x):
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def forward(self, mode: str):
        if mode == "video":
            return self.forward_video()
        elif mode == "image":
            return self.forward_image()

    def forward_video(self):
        # TODO: torch -> numpy
        processed_first_frame_image = self.sam_preprocessor_image(torch.tensor(self.ref_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)).numpy()
        processed_first_frame_mask = self.sam_preprocessor_image(torch.tensor(self.ref_mask * self.exp, dtype=torch.float32).unsqueeze(1)).numpy()
        image_embedding = self.sam_encoder_session.run(None, {"image": processed_first_frame_image})[0]
        image_embedding = image_embedding[0].transpose(1, 2, 0)

        fore_feat_list = []
        # Foreground features
        input_boxes = []
        for ref_mask in self.ref_mask:
            input_boxes.append(ref_mask)
    
        for obj in range(self.num_obj):
            print("Processing Object", obj)
            obj_mask = processed_first_frame_mask[obj, 0]
            obj_mask = cv2.resize(obj_mask, image_embedding.shape[0:2], interpolation=cv2.INTER_LINEAR)
    
            fore_feat = image_embedding[obj_mask > 0]
            if fore_feat.shape[0] == 0:
                fore_feat_list.append(fore_feat.mean(0))
                print("Find a small object in", name, "Object", obj)
                continue
    
            fore_feat_mean = fore_feat.mean(0)
            fore_feat_max = fore_feat.max(0)
            fore_feat = (fore_feat_max / 2 + fore_feat_mean / 2)[None]
            fore_feat = fore_feat / np.linalg.norm(fore_feat, axis=-1, keepdims=True)
            fore_feat_list.append(fore_feat)

        outputs = []
        for i in range (self.num_images-1):
            current_img = self.test_images[i]
            # TODO: torch -> numpy
            current_img = self.sam_preprocessor_image(torch.tensor(current_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)).numpy()
            
            # pred masks
            image_embeddings = self.sam_encoder_session.run(None, {"image": current_img})[0]
            test_feat = image_embeddings[0]
            C, htest, wtest = test_feat.shape
        
            test_feat = test_feat / np.linalg.norm(test_feat, axis=0, keepdims=True)
            test_feat = test_feat.reshape(C, htest * wtest)
            
            concat_mask = np.zeros((1, self.ref_mask.shape[1], self.ref_mask.shape[2]), dtype=np.uint8)
            for j in range(min(len(fore_feat_list), len(input_boxes))):
                # Cosine similarity
                fore_feat = fore_feat_list[j]
                sim = fore_feat @ test_feat  # 1, h*w
                sim = sim.reshape(htest, wtest)
                sim = cv2.resize(sim, (htest * 4, wtest * 4), interpolation=cv2.INTER_LINEAR)
                mask_sim = self.resize_and_crop(sim, original_size=self.original_size, image_size=1024)
        
                # Top-k point selection
                w, h = mask_sim.shape
        
                topk_xy_i, topk_label_i, _, _ = self.point_selection(mask_sim, topk=self.topk)
                topk_xy = topk_xy_i[None].astype(np.float32)
                topk_label = topk_label_i[None].astype(np.float32)

                
                # # use clustering
                # sim_points = (mask_sim >= 0.3)
                # np_xy = np.argwhere(sim_points*mask_sim)
                # np_xy = shuffle(np_xy, random_state=0)
                # xy_cnt = len(np_xy)
                
                # eps = 1 
                # dbs = DBSCAN(eps=eps, min_samples=1)
                # dbs.fit(np_xy)
                # labels = dbs.labels_
                # topk_xy = np.array([[0,0]]*len(set(labels)), dtype=np.float32)
                # for i in range(np.size(labels)):
                #     topk_xy[labels[i]]=[np_xy[i][1],np_xy[i][0]]
                # if xy_cnt > 0:
                #     topk_label = np.array([1] * len(set(labels)), dtype=np.float32).reshape(1, -1)

                # topk_xy = topk_xy.reshape(1, -1, 2)
                # topk_label = topk_label.reshape(1, -1)
                
        
                if self.center:
                    topk_label = np.concatenate([topk_label, [1]], axis=0)
                
                if self.box_prompt:
                    center, input_box_ = get_box_prompt(input_boxes[j], self.threshold)
                    if center:
                        topk_xy = np.concatenate((topk_xy, center), axis=0)
    
                    # TODO
                    masks, scores, logits, _ = predictor.predict(
                                point_coords=topk_xy,
                                point_labels=topk_label,
                                box=input_box_[None, :],
                                multimask_output=True,
                                attn_sim=attn_sim,
                                target_embedding=fore_feat.unsqueeze(0))
                else:
                    persam_decoder_inputs = {
                        "image_embeddings": image_embeddings,
                        "point_coords": topk_xy,
                        "point_labels": topk_label,
                        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                        "has_mask_input": np.array([[0]], dtype=np.float32),
                        "orig_im_size": self.original_size,
                    }
                    persam_decoder_outputs = [out.name for out in self.persam_decoder_session.get_outputs()]
                    masks, scores, logits = self.persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
    
                if self.large:
                    masks_ = masks 
                    mask_num = np.array([np.sum(masks_[0]), np.sum(masks_[1]), np.sum(masks_[2])], dtype=np.uint8)
                    ic_index = np.argmax(mask_num, axis=0).astype(np.uint8)
                else:
                    ic_index = 0
    
                persam_decoder_inputs = {
                    "image_embeddings": image_embeddings,
                    "point_coords": topk_xy,
                    "point_labels": topk_label,
                    "mask_input": logits[:, ic_index: ic_index + 1, :, :],
                    "has_mask_input": np.array([[1]], dtype=np.float32),
                    "orig_im_size": self.original_size,
                }
                persam_decoder_outputs = [out.name for out in self.persam_decoder_session.get_outputs()]
                masks, scores, logits = self.persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
                ic_index = np.argmax(scores)
        
                # box refine
                y, x = np.nonzero(masks[0, ic_index])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([[[x_min, y_min], [x_max, y_max]]], dtype=np.float32)
    
                topk_xy = np.concatenate((topk_xy, input_box), axis=1)
                topk_label = np.concatenate((topk_label, np.array([[2, 3]], dtype=np.float32)), axis=1)
                persam_decoder_inputs = {
                    "image_embeddings": image_embeddings,
                    "point_coords": topk_xy,
                    "point_labels": topk_label,
                    "mask_input": logits[:, ic_index: ic_index + 1, :, :],
                    "has_mask_input": np.array([[1]], dtype=np.float32),
                    "orig_im_size": self.original_size,
                }
                persam_decoder_outputs = [out.name for out in self.persam_decoder_session.get_outputs()]
                masks, scores, logits = self.persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
        
                ic_index = np.argmax(scores)
                masks = masks[0]
                concat_mask = np.concatenate((concat_mask, masks[-1].reshape(1, masks.shape[1], masks.shape[2])), axis=0)
                
            current_mask_pred = np.argmax(concat_mask, axis=0).astype(np.uint8)
            outputs.append(current_mask_pred)
        return outputs

    def forward_image(self):
        # TODO: torch -> numpy
        processed_first_frame_image = self.sam_preprocessor_image(torch.tensor(self.ref_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)).numpy()
        processed_first_frame_mask = self.sam_preprocessor_mask(torch.tensor(self.ref_mask, dtype=torch.float32).unsqueeze(1)).numpy()
        image_embedding = self.sam_encoder_session.run(None, {"image": processed_first_frame_image})[0]
        image_embedding = image_embedding[0].transpose(1, 2, 0)
    
        fore_feat_list = []
        fore_embedding_list = []
        # Foreground features
        input_boxes = []
        for ref_mask in self.ref_mask:
            input_boxes.append(ref_mask)
    
        for obj in range(self.num_obj):
            print("Processing Object", obj)
            obj_mask = processed_first_frame_mask[obj, 0]
            obj_mask = cv2.resize(obj_mask, image_embedding.shape[0:2], interpolation=cv2.INTER_LINEAR)
    
            target_feat = image_embedding[obj_mask > 0]
            target_embedding = target_feat.mean(0)[None]
            target_feat = target_embedding / np.linalg.norm(target_embedding, axis=-1, keepdims=True)
            target_embedding = target_embedding.unsqueeze(0)

            fore_feat_list.append(target_feat)
            fore_embedding_list.append(target_embedding)
    
        # process other frames
        onnx_outputs = []
        for i in range(self.num_images):
            current_img = self.test_images[i]
            # TODO: torch -> numpy
            current_img = self.sam_preprocessor_image(torch.tensor(current_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)).numpy()
            
            # pred masks
            image_embeddings = self.sam_encoder_session.run(None, {"image": current_img})[0]
            test_feat = image_embeddings[0]
            C, htest, wtest = test_feat.shape
        
            test_feat = test_feat / np.linalg.norm(test_feat, axis=0, keepdims=True)
            test_feat = test_feat.reshape(C, htest * wtest)
            
            concat_mask = np.zeros((1, first_frame_mask.shape[1], first_frame_mask.shape[2]), dtype=np.uint8)
            for j in range(min(len(fore_feat_list), len(input_boxes))):
                # Cosine similarity
                fore_feat = fore_feat_list[j]
                sim = fore_feat @ test_feat  # 1, h*w
                sim = sim.reshape(htest, wtest)
                sim = cv2.resize(sim, (h * 4, w * 4), interpolation=cv2.INTER_LINEAR)
                mask_sim = self.resize_and_crop(sim, original_size=original_size, image_size=1024)
        
                # Top-k point selection
                w, h = mask_sim.shape
                topk_xy_i, topk_label_i, last_xy_i, last_label_i = self.point_selection(mask_sim, topk=topk)
                topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)[None].astype(np.float32)
                topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)[None].astype(np.float32)
        
                # Obtain the target guidance for cross-attention layers
                sim = (sim - sim.mean()) / sim.std()
                sim = cv2.resize(sim, (64, 64), interpolation=cv2.INTER_LINEAR)
                attn_sim = sigmoid(sim).flatten()[None, None, None]
                target_embedding = fore_feat[None]
        
                if self.center:
                    topk_label = np.concatenate([topk_label, [1]], axis=0)
                
                if self.box_prompt:
                    center, input_box_ = self.get_box_prompt(input_boxes[j], self.threshold)
                    if center:
                        topk_xy = np.concatenate((topk_xy, center), axis=0)
    
                    # TODO
                    masks, scores, logits, _ = predictor.predict(
                                point_coords=topk_xy,
                                point_labels=topk_label,
                                box=input_box_[None, :],
                                multimask_output=True,
                                attn_sim=attn_sim,
                                target_embedding=fore_feat.unsqueeze(0))
                else:
                    persam_decoder_inputs = {
                        "image_embeddings": image_embeddings,
                        "point_coords": topk_xy,
                        "point_labels": topk_label,
                        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                        "has_mask_input": np.array([[0]], dtype=np.float32),
                        "orig_im_size": original_size,
                        "attn_sim": attn_sim,
                        "target_embedding": target_embedding,
                    }
                    persam_decoder_outputs = [out.name for out in persam_decoder_session.get_outputs()]
                    masks, scores, logits = persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
    
                if large:
                    masks_ = masks 
                    mask_num = np.array([np.sum(masks_[0]), np.sum(masks_[1]), np.sum(masks_[2])], dtype=np.uint8)
                    ic_index = np.argmax(mask_num, axis=0).astype(np.uint8)
                else:
                    ic_index = 0
    
                persam_decoder_inputs = {
                    "image_embeddings": image_embeddings,
                    "point_coords": topk_xy,
                    "point_labels": topk_label,
                    "mask_input": logits[:, ic_index: ic_index + 1, :, :],
                    "has_mask_input": np.array([[1]], dtype=np.float32),
                    "orig_im_size": original_size,
                    "attn_sim": attn_sim,
                    "target_embedding": target_embedding,
                }
                persam_decoder_outputs = [out.name for out in persam_decoder_session.get_outputs()]
                masks, scores, logits = persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
                ic_index = np.argmax(scores)
        
                # box refine
                y, x = np.nonzero(masks[0,ic_index])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([[[x_min, y_min], [x_max, y_max]]], dtype=np.float32)
    
                topk_xy = np.concatenate((topk_xy, input_box), axis=1)
                topk_label = np.concatenate((topk_label, np.array([[2, 3]], dtype=np.float32)), axis=1)
                persam_decoder_inputs = {
                    "image_embeddings": image_embeddings,
                    "point_coords": topk_xy,
                    "point_labels": topk_label,
                    "mask_input": logits[:, ic_index: ic_index + 1, :, :],
                    "has_mask_input": np.array([[1]], dtype=np.float32),
                    "orig_im_size": original_size,
                    "attn_sim": attn_sim,
                    "target_embedding": target_embedding,
                }
                persam_decoder_outputs = [out.name for out in persam_decoder_session.get_outputs()]
                masks, scores, logits = persam_decoder_session.run(persam_decoder_outputs, persam_decoder_inputs)
        
                ic_index = np.argmax(scores)
                masks = masks[0]
                concat_mask = np.concatenate((concat_mask, masks[-1].reshape(1, masks.shape[1], masks.shape[2])), axis=0)
                
            current_mask_pred = np.argmax(concat_mask, axis=0).astype(np.uint8)
            # output = Image.fromarray(current_mask_pred)
            # output.putpalette(palette)
            # output.save(save_path + '{:05d}.png'.format(i))
            onnx_outputs.append([current_img, msk[:,i], current_mask_pred])
        
            if box_prompt:
                cur_labels = np.unique(current_mask_pred)
                cur_labels = cur_labels[cur_labels!=0]
                input_boxes = all_to_onehot(current_mask_pred, cur_labels)
            
        print(f"Finish predict video: {name}")


if __name__ == "__main__":
    inferencer = ONNXVisualPromptingInferencer(mode="video")
    inferencer.set_images()
