"""Export the Omnidata depth and normal networks to ONNX at 512x512, for the TensorRT engines
of scripts/build_trt_engines.sh. Run from the repository root with the step-4 checkpoints in
pretrained_models/; writes the .onnx files next to them.
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vigs'))
from midas.omnidata import OmnidataModel   # noqa: E402


def export_onnx(model, dummy, filename):
    model = model.eval().cuda()

    torch.onnx.export(
        model,
        dummy,
        filename,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    print(f"saved {filename}")



if __name__ == "__main__":
    dummy = torch.randn(1, 3, 512, 512).cuda()

    depth_model = OmnidataModel(
        'depth',
        'pretrained_models/omnidata_dpt_depth_v2.ckpt',
        device='cuda'
    ).model
    export_onnx(depth_model, dummy, "pretrained_models/omnidata_depth_512.onnx")

    normal_model = OmnidataModel(
        'normal',
        'pretrained_models/omnidata_dpt_normal_v2.ckpt',
        device='cuda'
    ).model
    export_onnx(normal_model, dummy, "pretrained_models/omnidata_normal_512.onnx")
