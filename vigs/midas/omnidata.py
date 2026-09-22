"""The Omnidata depth and normal networks (DPT on a ViT-B/ResNet-50 hybrid) as the tracker
uses them: the depth prior of every keyframe, the normal prior of the Gaussian map, and the
geometric cue of the ride detector. `OmnidataModel(task, checkpoint)` loads one task's weights;
calling it on a normalized (1, 3, H, W) tensor returns the network output."""
from pathlib import Path

import torch

from midas.dpt_depth import DPTDepthModel


class OmnidataModel:
    backbone = "vitb_rn50_384"
    channel_dict = {"depth": 1, "normal": 3}
    ckpt_dict = {
        "depth": "omnidata_dpt_depth_v2.ckpt",
        "normal": "omnidata_dpt_normal_v2.ckpt",
    }

    def __init__(self, task="depth", model_path=None, device="cuda:0"):
        if model_path is None:
            model_path = Path.cwd() / "pretrained_models" / self.ckpt_dict[task]

        self.model_path = model_path
        self.task = task
        self.channel = self.channel_dict[task]
        self.device = device

        self.model = DPTDepthModel(backbone=self.backbone, num_channels=self.channel)

        try:
            checkpoint = torch.load(self.model_path, map_location=device)
        except Exception:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        assert "state_dict" in checkpoint, "No state_dict found in checkpoint"

        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            # remove the "model." prefix
            state_dict[k[len("model.") :]] = v
        self.model.load_state_dict(state_dict)
        self.model.to(device)

    def __call__(self, im_tensor):
        with torch.no_grad():
            return self.model(im_tensor)
