"""The TensorRT engines of the tracker, each with a PyTorch fallback: the Omnidata depth and
normal networks, the DROID feature encoder, and the ConvGRU trunk of the update operator (its
graph aggregation stays in PyTorch). `try_create` returns None when an engine file is missing,
and the caller keeps the PyTorch module."""
import torch
from modules.trt_base import TrtRunner


class OmnidataTrtRunner:
    """TensorRT-accelerated Omnidata depth and normal estimator.

    Runs two separate engines (depth + normal) on the same input image.
    Falls back to the PyTorch OmnidataModel if the .engine files are missing.
    """

    def __init__(self, normal=True):
        self.depth_runner  = TrtRunner("pretrained_models/omnidata_depth_512_simplified_fp16.engine")
        # the normal prior feeds the Gaussian map only: a tracking run never loads its engine
        self.normal_runner = TrtRunner("pretrained_models/omnidata_normal_512_simplified_fp16.engine") if normal else None

    @classmethod
    def try_create(cls, normal=True):
        try:
            return cls(normal=normal)
        except Exception as e:
            print(f"[TRT] OmnidataTrtRunner not available ({e}), falling back to PyTorch.")
            return None

    def __call__(self, image):
        """(depth, normal); normal is None when the runner was created without it."""
        with torch.cuda.amp.autocast(enabled=False):
            image_f32 = image.to(torch.float32)
            depth  = self.depth_runner.run({"input": image_f32})[0]
            normal = self.normal_runner.run({"input": image_f32})[0] if self.normal_runner is not None else None
            return depth, normal


class FeatureEncoderTrtRunner:
    """TensorRT-accelerated DroidNet feature encoder (fnet).

    Falls back to the PyTorch fnet if the .engine file is missing.
    """

    def __init__(self):
        self.runner = TrtRunner("pretrained_models/droidnet_fnet_fp16.engine")

    @classmethod
    def try_create(cls):
        try:
            return cls()
        except Exception as e:
            print(f"[TRT] FeatureEncoderTrtRunner not available ({e}), falling back to PyTorch.")
            return None

    def __call__(self, image):
        with torch.cuda.amp.autocast(enabled=False):
            return self.runner.run({"input": image.to(torch.float32)})[0]


class UpdateModuleTRTRunner:
    """TensorRT-accelerated DroidNet update operator (partial).

    The graph-aggregation step (self.agg) cannot be exported to TRT, so the
    update operator is split: the GRU trunk runs in TRT, then self.agg runs
    in PyTorch on the TRT output.

    Falls back to the full PyTorch update operator if the .engine file is missing.
    """

    def __init__(self, net, pgba=False):
        engine = ("pretrained_models/update_module_partial_pgba_fp16.engine" if pgba
                  else "pretrained_models/update_module_partial_fp16.engine")
        self.runner = TrtRunner(engine)
        self.agg = net.update.agg

    @classmethod
    def try_create(cls, net, pgba=False):
        try:
            return cls(net, pgba=pgba)
        except Exception as e:
            print(f"[TRT] UpdateModuleTRTRunner not available ({e}), falling back to PyTorch.")
            return None

    def flow_probe(self, net, inp, corr):
        """(delta, weight) of one GRU step with zero flow, no graph aggregation: the motion
        filter's per-frame flow-magnitude probe (net.update(net, inp, corr) in PyTorch)."""
        flow = torch.zeros(net.shape[0], net.shape[1], 4, net.shape[-2], net.shape[-1],
                           device=net.device, dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            _, delta, weight = self.runner.run({"net": net.to(torch.float32), "inp": inp.to(torch.float32),
                                                "corr": corr.to(torch.float32), "flow": flow})
        return delta, weight

    def __call__(self, net, inp, corr, flow=None, ii=None):
        # Only the call form with flow and ii/jj is implemented; that is the form
        # the factor graph uses.
        if flow is not None:
            net  = net.to(torch.float32)
            inp  = inp.to(torch.float32)
            corr = corr.to(torch.float32)
            flow = flow.to(torch.float32)
            with torch.cuda.amp.autocast(enabled=False):
                net, delta, weight = self.runner.run({"net": net, "inp": inp, "corr": corr, "flow": flow})
            eta, upmask = self.agg(net, ii.to(net.device))
            net     = net.to(torch.float16)
            delta   = delta.to(torch.float16)
            weight  = weight.to(torch.float16)
            upmask  = upmask.to(torch.float16)
            return net, delta, weight, eta, upmask
