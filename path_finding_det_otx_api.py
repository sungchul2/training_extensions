"""YOLOX training using OTX API.

https://github.com/Megvii-BaseDetection/YOLOX/tree/main
"""

import random
import sys
import time
from datetime import timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from datumaro import Bbox, Image
from torchvision import tv_tensors

import otx
from otx.core.config.data import DataModuleConfig, SubsetConfig
from otx.core.data.dataset.detection import \
    OTXDetectionDataset as _OTXDetectionDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import (DetBatchDataEntity,
                                            DetBatchPredEntity, DetDataEntity)
from otx.core.data.module import OTXDataModule
from otx.core.data.transform_libs.torchvision import (PadtoSquare,
                                                      ResizetoLongestEdge)
from otx.core.model.entity.detection import OTXDetectionModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine

sys.path.append("/local_ssd3/sungchul/workspace/src/YOLOX")
from yolox.data.data_augment import random_affine
from yolox.data.datasets.mosaicdetection import get_mosaic_coordinate
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import adjust_box_anns, postprocess



class MosaicDetection(_OTXDetectionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = (640, 640)
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
    
    def _get_item_impl(self, index: int) -> DetDataEntity | None:
        if self.dm_subset.name == "train":
            input_h, input_w = self.input_dim[0], self.input_dim[1]
            
            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
            
            # 3 additional image indices
            indices = [index] + [random.randint(0, self.__len__() - 1) for _ in range(3)]
            
            mosaic_labels = []
            for i_mosaic, index in enumerate(indices):
                item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
                img = item.media_as(Image)
                ignored_labels: list[int] = []  # This should be assigned form item
                img_data, img_shape = self._get_img_data_and_shape(img)
                if i_mosaic == 0:
                    ori_shape = img_shape
                
                h0, w0 = img_shape  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img_data, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1
                
                bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
                _bboxes = (
                    np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
                    if len(bbox_anns) > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
                _labels = np.expand_dims(np.array([ann.label for ann in bbox_anns]), axis=1)
                bboxes_labels = np.concatenate((_bboxes, _labels), axis=1)

                labels = bboxes_labels.copy()
                # Normalized xywh to pixel xyxy format
                if bboxes_labels.size > 0:
                    labels[:, 0] = scale * bboxes_labels[:, 0] + padw
                    labels[:, 1] = scale * bboxes_labels[:, 1] + padh
                    labels[:, 2] = scale * bboxes_labels[:, 2] + padw
                    labels[:, 3] = scale * bboxes_labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )
            
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
                
            entity = DetDataEntity(
                image=mosaic_img,
                img_info=ImageInfo(
                    img_idx=index,
                    img_shape=mosaic_img.shape[:2],
                    ori_shape=ori_shape,
                    image_color_channel=self.image_color_channel,
                    ignored_labels=ignored_labels,
                ),
                bboxes=tv_tensors.BoundingBoxes(
                    mosaic_labels[:,:4],
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=mosaic_img.shape[:2],
                    dtype=torch.float32,
                ),
                labels=torch.as_tensor(mosaic_labels[:,-1].squeeze(), dtype=torch.int64),
            )
            
            return self._apply_transforms(entity)

        return super()._get_item_impl(index) # val, test
            
    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        # cp_labels = []
        # while len(cp_labels) == 0:
        #     cp_index = random.randint(0, self.__len__() - 1)
        #     cp_labels = self._dataset.load_anno(cp_index)
        # img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        cp_index = random.randint(0, self.__len__() - 1)
        item = self.dm_subset.get(id=self.ids[cp_index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img, img_shape = self._get_img_data_and_shape(img)
        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
        _bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        _labels = np.expand_dims(np.array([ann.label for ann in bbox_anns]), axis=1)
        cp_labels = np.concatenate((_bboxes, _labels), axis=1)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
            
            
otx.core.data.dataset.detection.OTXDetectionDataset = MosaicDetection


if __name__ == "__main__":
    class OTXYOLOXModel(OTXDetectionModel):
        def _create_model(self) -> nn.Module:
            """Create a PyTorch model for this class."""
            def init_yolo(M):
                for m in M.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eps = 1e-3
                        m.momentum = 0.03

            in_channels = [256, 512, 1024]
            self.depth = 0.33
            self.width = 0.50
            self.act = "silu"
            self.max_labels = 120
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            model = YOLOX(backbone, head)
            model.apply(init_yolo)
            model.head.initialize_biases(1e-2)
            
            weights = torch.hub.load_state_dict_from_url(
                "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
                map_location="cpu")
            for key in [
                "head.cls_preds.0.weight",
                "head.cls_preds.0.bias",
                "head.cls_preds.1.weight",
                "head.cls_preds.1.bias",
                "head.cls_preds.2.weight",
                "head.cls_preds.2.bias"
            ]:
                if key in weights["model"]:
                    weights["model"].pop(key)
            model.load_state_dict(weights["model"], strict=False)
            model.train()
            return model

        def _customize_inputs(self, inputs: DetBatchDataEntity):
            """Customize OTX input batch data entity if needed for your model."""
            def xyxy2cxcywh(bboxes):
                bboxes_cxcywh = bboxes.clone()
                bboxes_cxcywh[:, 2] = bboxes_cxcywh[:, 2] - bboxes_cxcywh[:, 0]
                bboxes_cxcywh[:, 3] = bboxes_cxcywh[:, 3] - bboxes_cxcywh[:, 1]
                bboxes_cxcywh[:, 0] = bboxes_cxcywh[:, 0] + bboxes_cxcywh[:, 2] * 0.5
                bboxes_cxcywh[:, 1] = bboxes_cxcywh[:, 1] + bboxes_cxcywh[:, 3] * 0.5
                return tv_tensors.BoundingBoxes(bboxes_cxcywh, format="CXCYWH", canvas_size=bboxes.canvas_size)
            targets = [torch.hstack((labels.unsqueeze(-1), xyxy2cxcywh(bboxes))) for bboxes, labels in zip(inputs.bboxes, inputs.labels)]
            padded_targets = torch.zeros((len(targets), self.max_labels, 5), device=inputs.images.device)
            for i, target in enumerate(targets):
                padded_targets[i,range(len(target))[: self.max_labels]] = target[: self.max_labels]
            return {"x": inputs.images, "targets": padded_targets}

        def _customize_outputs(self, outputs, inputs):
            """Customize OTX output batch data entity if needed for model."""
            if self.training:
                return {"loss" if k == "total_loss" else k: v for k, v in outputs.items() if "loss" in k}

            postprocessed_outputs = postprocess(outputs, self.num_classes, conf_thre=0.01, nms_thre=0.65)

            scores = []
            bboxes = []
            labels = []
            for i, output in enumerate(postprocessed_outputs):
                if output is None:
                    scores.append(torch.tensor([0.], device=inputs.images[i].device))
                    bboxes.append(tv_tensors.BoundingBoxes(
                        torch.tensor([0., 0., 0., 0.]),
                        format="XYXY",
                        canvas_size=inputs.imgs_info[i].img_shape,
                        device=inputs.images[i].device))
                    labels.append(inputs.labels[i][0].unsqueeze(0))
                    continue
                # output = (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                scores.append(output[:,4] * output[:,5])
                bboxes.append(
                    tv_tensors.BoundingBoxes(
                        output[:,:4],
                        format="XYXY",
                        canvas_size=inputs.imgs_info[i].img_shape,
                    ),
                )
                labels.append(output[:,6].to(torch.int64))

            return DetBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
            )

    start = time.time()
    transform = v2.Compose([
        # ResizetoLongestEdge(size=640, antialias=True),
        # PadtoSquare(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = v2.Compose([
        ResizetoLongestEdge(size=640, antialias=True),
        PadtoSquare(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 8 # 64

    train_data_cfg = SubsetConfig(batch_size=batch_size, subset_name='train', transforms=transform)#, num_workers=0) # TODO : reset num_workers
    val_data_cfg = SubsetConfig(batch_size=batch_size, subset_name='val', transforms=transform_test)#, num_workers=0)
    test_data_cfg = SubsetConfig(batch_size=batch_size, subset_name='test', transforms=transform_test)#, num_workers=0)
    datamodule_cfg = DataModuleConfig(
        data_format='coco',
        data_root="/local_ssd2/sungchul/workspace/data/wgisd_coco/",
        train_subset=train_data_cfg,
        val_subset=val_data_cfg,
        test_subset=test_data_cfg)

    model = OTXYOLOXModel(num_classes=5)
    datamodule = OTXDataModule(task=OTXTaskType.DETECTION, config=datamodule_cfg)

    engine = Engine(
        task=OTXTaskType.DETECTION,
        work_dir="./det_otx_api",
        datamodule=datamodule,
        model=model,
        # optimizer=get_optimizer(model=model, batch_size=batch_size)
        # optimizer: list[OptimizerCallable] | OptimizerCallable | None = None,
        # scheduler: list[LRSchedulerCallable] | LRSchedulerCallable | None = None,
    )
    
    results = engine.train(max_epochs=96, seed=42, deterministic=True)
    
    dt = timedelta(seconds=time.time() - start)
    print(f"Elapsed time: {dt}")
