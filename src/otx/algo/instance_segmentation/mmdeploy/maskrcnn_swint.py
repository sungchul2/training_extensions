"""MMDployment config of MaskRCNN-SwinT-FP16 model for Instance-Seg Task.

reference: https://github.com/open-mmlab/mmdeploy/
"""

_base_ = ["./base_instance_segmentation.py"]


# NOTE: Its necessary to use opset11 as squeeze>=opset13 does not work in
# mmdeploy::mmcv::ops::roi_align::roi_align_default.
# Refer to src/otx/algorithms/common/adapters/mmdeploy/ops/custom_ops.py::squeeze__default for future rewrite.
ir_config = dict(
    output_names=["boxes", "labels", "masks"],
    opset_version=11,
)

scale_ir_input = False
