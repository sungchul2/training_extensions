"""MMDeploy config of ATSS model for Detection Task.

reference: https://github.com/open-mmlab/mmdeploy/
"""

_base_ = ["./base_detection.py"]

ir_config = dict(
    output_names=["boxes", "labels"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 736, 992]))],
)
