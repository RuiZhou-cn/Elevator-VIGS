"""Per-frame front of the tracker: keyframe selection, feature encoding and the depth prior.

Every incoming frame is scored by the optical-flow probe; a keyframe is created on enough
motion or after a time limit. That limit drops from 3 s to `Elevator.kf_dt` while the ride
detector is armed (App. C.4). The filter also samples the two cues of the armed window
(`elevator.detect.ArmedWindowGate`, Sec. 3.4) once per `sample_dt` of video.
"""
import torch
import threading
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import geom.projective_ops as pops

from modules.corr import CorrBlock
from torchvision import transforms
from midas.omnidata import OmnidataModel
from modules.trt_runner import FeatureEncoderTrtRunner, OmnidataTrtRunner
from elevator.detect import ArmedWindowGate


class MotionFilter:
    """ This class is used to filter incoming frames, extract features, and extract depth and normal """

    def __init__(self, net, video, config, disable_mono, device="cuda:0", update_op=None):

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        # the frontend's TensorRT update trunk, shared for the per-frame flow probe (same
        # thread, same engine profile: 1 edge is inside its shape range)
        self.update_trt = update_op if hasattr(update_op, "flow_probe") else None
        self.disable_mono = disable_mono
        self.video = video
        self.thresh = config["Tracking"]["motion_filter"]["thresh"]
        self.init_thresh = config["Tracking"]["motion_filter"]["init_thresh"] if "init_thresh" in config["Tracking"]["motion_filter"] else self.thresh
        self.device = device

        self.omni_dep = None
        self._omni_lock = threading.Lock()   # ensure_omni_dep is also called from the gate's loader thread
        self.omni_normal = None

        # Rides need denser keyframes (App. C.4): sparse edges span the whole ride and lose the
        # rise. The window comes from the online detector (ride_hint / elev_armed), never from
        # config; elevator handling is always on, so an elevator-free sequence runs the same code.
        self._elev_kf_dt = float(config["Elevator"]["kf_dt"])
        # The armed window of Sec. 3.4: the two cues per sampled frame -> causal ARMED/DISARMED,
        # published on `video` for the frontend's ride FSM. The geometric cue shares this
        # filter's Omnidata depth model.
        self._armed_gate = ArmedWindowGate(video, device, config["Elevator"]["detect"]["armed"])
        video.armed_gate = self._armed_gate         # the frontend waits on it before reading events

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        self.feature_encoder_trt = FeatureEncoderTrtRunner.try_create()
        self.omnidata_trt = OmnidataTrtRunner.try_create(normal=video.dense_outputs)

    def ensure_omni_dep(self):
        """This filter's own Omnidata depth model for the keyframe prior (PyTorch path only;
        with TensorRT engines this method is never called).

        Deliberately NOT shared with the geometric cue's scorer, which loads its own copy: the two
        run on different threads and CUDA streams, and one module driven by two concurrent
        forwards returns corrupted depth -- a poisoned prior for the BA and a poisoned rho_sp for
        the armed window. The second copy costs 470 MB of weights.
        """
        with self._omni_lock:
            if self.omni_dep is None:
                self.omni_dep = OmnidataModel('depth', 'pretrained_models/omnidata_dpt_depth_v2.ckpt', device="cuda:0")
            return self.omni_dep

    @torch.cuda.amp.autocast(enabled=True)
    def context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def feature_encoder(self, image):
        """ features for correlation volume """
        if self.feature_encoder_trt is not None:
            return self.feature_encoder_trt(image).squeeze(0).clone()  # have to clone, otherwise strange behavior, very few keyframe
        else:
            return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def prior_extractor(self, im_tensor):
        """Monocular depth + normal priors for one frame, resampled to the input resolution."""
        if self.disable_mono:
            return None, None
        input_size = im_tensor.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize((512, 512), antialias=True)])
        im_tensor = trans_totensor(im_tensor).cuda()
        # The normal prior feeds the Gaussian map only (video.dense_outputs): a tracking run
        # skips that second DPT forward, and never loads its weights.
        if self.omnidata_trt is not None:
            depth, normal = self.omnidata_trt(im_tensor)
        else:
            depth = self.ensure_omni_dep()(im_tensor)
            normal = None
            if self.video.dense_outputs:
                if self.omni_normal is None:
                    self.omni_normal = OmnidataModel('normal', 'pretrained_models/omnidata_dpt_normal_v2.ckpt', device="cuda:0")
                normal = self.omni_normal(im_tensor)

        depth = depth[None] * 50
        depth = F.interpolate(depth, input_size, mode='bicubic')
        depth = depth.float().squeeze()
        if normal is not None:
            normal = normal * 2.0 - 1.0
            normal = F.interpolate(normal, input_size, mode='bicubic')
            normal = normal.float().squeeze()

        return depth, normal

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, t, tstamp, image, intrinsics=None, pose=None):
        """ main update operation - run on every frame in video """
        # one upload of the reader's RGB frame, shared by the cue scorer and the networks
        image = image.to(self.device)
        # armed-window sample (~3.4 Hz), the two cues scored on the worker thread
        self._armed_gate.sample(tstamp, image[0])
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8
        intrinsics[:,:4] /= 8.0
        # [2,1,0] turns the reader's RGB into BGR for the networks -- the base system does the
        # same and every reported number carries it, so it stays. Then normalize.
        inputs = image[None,:, [2,1,0]] / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV) # same in droid-slam, just following it
        inputs_for_prior = inputs
        # extract features
        gmap = self.feature_encoder(inputs)

        # always add first frame to the depth video
        if self.video.counter.value == 0:
            depth, normal = self.prior_extractor(inputs_for_prior[0])
            net, inp = self.context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(t, image[0], pose, 1.0, depth, normal, intrinsics, gmap, net[0], inp[0], tstamp)
        # only add new keyframe if there is enough motion or time difference is too long
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration (one GPU sync for the read)
            if self.update_trt is not None:
                delta, weight = self.update_trt.flow_probe(self.net[None], self.inp[None], corr)
            else:
                _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            _flow_mag = delta.norm(dim=-1).mean().item()

            # NOTE: can consider before initialization use larger thresh ( use init_thresh)
            # Cap the KF gap so IMU pre-integration stays short: 3 s normally, ~1 s once the
            # detector arms (ride_hint) or the armed window is open (elev_armed; scored on a
            # worker thread, so it lags by the one sample in flight). k_1 is not forced here:
            # the ~1 s in-ride cadence already puts a keyframe inside the arrival rest.
            _in_ride = self.video.ride_hint or self.video.elev_armed
            _maxdt = self._elev_kf_dt if _in_ride else 3
            _last_kf_t = self.video.kf_stamps[self.video.counter.value-1]
            if _flow_mag > self.thresh or (tstamp - _last_kf_t) > _maxdt:
                net, inp = self.context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                depth, normal = self.prior_extractor(inputs_for_prior[0])
                self.video.append(t, image[0], pose, None, depth, normal, intrinsics, gmap, net[0], inp[0], tstamp)
