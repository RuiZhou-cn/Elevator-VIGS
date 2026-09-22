"""Export the DroidNet feature encoder and ConvGRU update module to ONNX, for the TensorRT
engines of scripts/build_trt_engines.sh. Run from the repository root; writes into
pretrained_models/. ONNX is exported in fp32 and converted to fp16 at engine build time.
"""
import os
import sys
from collections import OrderedDict

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vigs'))
from modules.droid_net import DroidNet   # noqa: E402


if __name__ == "__main__":
    droidnet = DroidNet()
    weights = "pretrained_models/droid.pth"
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])
    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
    droidnet.load_state_dict(state_dict)
    droidnet.to("cuda:0").eval()

    # Feature encoder (fnet), dynamic H/W axes: the engine serves every input resolution the
    # runtime uses.
    dummy = torch.randn(1, 1, 3, 344, 616, device="cuda", dtype=torch.float32)
    torch.onnx.export(
        droidnet.fnet,
        dummy,
        "pretrained_models/droidnet_fnet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {3: "H", 4: "W"},      # N, T, C, H, W
            "output": {3: "H_out", 4: "W_out"}
        },
        opset_version=18,
    )

    # ConvGRU update module. Its graph aggregation (self.agg) does not export, so the update is
    # split: the GRU trunk runs in TensorRT and the aggregation in PyTorch afterwards.
    num = 28
    B, C, H, W = 1, 128, 43, 77

    net  = torch.randn(B, num, 128, H, W, device="cuda", dtype=torch.float32)
    inp  = torch.randn(B, num, 128, H, W, device="cuda", dtype=torch.float32)
    corr = torch.randn(B, num, 196, H, W, device="cuda", dtype=torch.float32)
    flow = torch.randn(B, num,   4, H, W, device="cuda", dtype=torch.float32)

    torch.onnx.export(
        droidnet.update,
        (net, inp, corr, flow),
        "pretrained_models/update_module_partial.onnx",
        opset_version=17,
        input_names=["net", "inp", "corr", "flow"],
        output_names=["net_out", "delta", "weight"],
        dynamic_axes={
            "net":  {1: "num", 3: "H", 4: "W"},
            "inp":  {1: "num", 3: "H", 4: "W"},
            "corr": {1: "num", 3: "H", 4: "W"},
            "flow": {1: "num", 3: "H", 4: "W"},

            "net_out": {1: "num", 3: "H", 4: "W"},
            "delta":   {1: "num", 3: "H", 4: "W"},
            "weight":  {1: "num", 3: "H", 4: "W"},
        }
    )
