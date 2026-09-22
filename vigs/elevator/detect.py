"""The ride detector of Sec. 3.4: is the robot inside an elevator, and has it departed?

Two independent layers, in the order a ride trips them. Neither ever feeds the estimator --
together they only answer WHEN, and hand that window to `elevator.transport`.

* the ARMED WINDOW (`CueScorer` / `ArmedWindowFSM` / `ArmedWindowGate`), sampled by the motion
  filter every ~0.29 s of video: the semantic cue P_elev of Eq. (10), a frozen zero-shot SigLIP2
  scored against the negative pool, together with the geometric cue rho_sp of Eq. (11), the
  depth spread of the Omnidata relative depth. Both are EMA-smoothed and must fire together to
  open the armed window. It publishes its transitions on the shared DepthVideo; `armed_at`
  replays them.
* the RIDE FSM (`RideDetector`), stepped by the tracking frontend at IMU rate inside that armed
  window: three states over the open-loop world-vertical velocity v_z,

    DISARMED --window opens--> ARMED --|v_z| >= depart_v held--> RIDING     (departure)
    RIDING --v_z back in band--> ARMED                                      (arrival)
    ARMED --walked leave_dist from the last arrival--> DISARMED
    ARMED --departure with depart_dh_max sideways--> DISARMED   (departure refused)
    RIDING --rise stays under min_rise_m--> DISARMED            (ride discarded)

  where v_z is the open integral of the world-vertical acceleration a_z, low-pass filtered and
  with its constant error c removed (Sec. 3.4, App. C.4). It is only thresholded, and pinned to
  zero whenever the body is at rest, which bounds its drift to a single ride. The two
  rejections are what an elevator cannot do: carry a robot that was moving sideways when it
  departed (a rider stands still as the elevator leaves; the check runs BEFORE the ride is
  committed, so a refused departure never touches the estimator), or stop less than a floor
  from where it departed. A discarded ride is published on `rejected`, never on `rides`, and
  the visit closes as after leaving the elevator: the armed window must drop and rise again
  before another can open. The sideways test is NOT repeated during a ride: inside an elevator
  the horizontal estimate is the quantity the ride corrupts (vision sees a static box while the
  IMU accelerates), and on real rides it wanders metres with the rider standing still --
  one station recording slides 5 m in the 2 s after the departure, Mall-H 1-3 m in the
  transient after activation -- so a sideways veto there threw out true rides.

The armed window is load-bearing, not an optimisation: an ungated FSM produces catastrophic
false rides.
"""
import queue
import threading
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# =============================================================================================
# The two cues and the armed window -- keyframe rate, run by the motion filter
# =============================================================================================
SIGLIP_MODEL = "google/siglip2-base-patch16-256"
# The elevator classes M of Eq. (10): the generic prompts are averaged into ONE class
# (paraphrase mean), each variant is its own class. All compete against the negative pool N.
# Printed verbatim as Table 7 of the paper -- keep the two in step.
PROMPTS_ELEVATOR_GENERIC = [
    "a photo taken from inside an elevator cabin",
    "the interior of an elevator with metal walls and a button panel",
    "inside a small elevator, the doors are closed",
]
PROMPTS_ELEVATOR_VARIANTS = [
    "inside an elevator with mirrored walls",
    "the interior of a glass panoramic elevator",
    "inside a wood-paneled elevator cabin",
    "an elevator button panel with floor number buttons",
    # Glass elevators score far below p_arm without these prompts.
    "the view out through the glass wall of an elevator",
    "looking out of a glass elevator while it moves",
    "inside a panoramic lift looking out over a shopping mall atrium",
]
PROMPTS_NEGATIVE = [
    "a long empty corridor in a building",
    "an office room with desks and chairs",
    "a staircase inside a building",
    "the lobby of a building",
    "a plain blank wall",
    "a wooden door in a room",
    "a city street outdoors",
    "a laboratory room with equipment",
    "a glass-walled meeting room in an office",
    "the atrium of a modern building with glass railings",
    "a public restroom with a large mirror",
    "a wood-paneled sauna room",
    # absorber block (keep last): the near-misses seen from OUTSIDE the elevator
    "closed elevator doors seen from the hallway",
    "an elevator entrance with a call button, seen from outside",
    "open elevator doors seen from the hallway",
    "a closed door seen up close",
    "a fire door on a stairwell landing",
]


class CueScorer:
    """The two cues of Sec. 3.4 on a (3,H,W) uint8 RGB tensor: `semantic_cue` is the elevator
    probability P_elev of Eq. (10) from a frozen zero-shot SigLIP2, and `geometric_cue` is the
    depth spread rho_sp of Eq. (11) from the Omnidata relative depth, small inside an enclosed
    elevator. Models load lazily on first use; pass `omni` to share the motion filter's depth
    model."""

    def __init__(self, device="cuda:0", omni=None):
        self.device = device
        self.omni = omni
        self._loaded = False

    def _load(self):
        import torch
        from midas.omnidata import OmnidataModel
        from transformers import AutoModel, AutoProcessor

        shared = self.omni is not None
        if not shared:
            self.omni = OmnidataModel("depth", "pretrained_models/omnidata_dpt_depth_v2.ckpt",
                                      device=self.device)
        self.omni.model.eval()
        # cache first: from_pretrained otherwise asks the Hub for the current revision on
        # every run, a network round trip (or its 10 s timeout) that would sit inside the
        # timed frame loop. The download happens once, on a cache miss.
        # Loaded at float32 and cast afterwards: transformers instantiates a model under
        # torch.set_default_dtype(<dtype>), PROCESS-wide, so a float16 load on this thread
        # would turn the main thread's concurrent tensor construction Half (the inertial
        # init's linspace). float32 is the default already, so nothing changes.
        try:
            proc = AutoProcessor.from_pretrained(SIGLIP_MODEL, local_files_only=True)
            self.sig = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32,
                                                 local_files_only=True)
        except OSError:
            proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
            self.sig = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float32)
        self.sig = self.sig.to(self.device).half().eval()
        prompts = PROMPTS_ELEVATOR_GENERIC + PROMPTS_ELEVATOR_VARIANTS + PROMPTS_NEGATIVE
        tok = proc(text=prompts, padding="max_length", max_length=64,
                   return_tensors="pt").to(self.device)
        with torch.no_grad():
            temb = self.sig.get_text_features(**tok)
            temb = temb if torch.is_tensor(temb) else temb.pooler_output
            temb = (temb / temb.norm(dim=-1, keepdim=True)).float()
            ng = len(PROMPTS_ELEVATOR_GENERIC)
            gen = temb[:ng].mean(0, keepdim=True)
            gen = gen / gen.norm(dim=-1, keepdim=True)
            self.temb = torch.cat([gen, temb[ng:]], 0)  # [generic, variants..., negs...]
            self._n_elev = 1 + len(PROMPTS_ELEVATOR_VARIANTS)
            self._scale = self.sig.logit_scale.exp().float()
            self._bias = self.sig.logit_bias.float()
        # the text tower (538 MiB) is dead once the prompts are embedded
        del self.sig.text_model
        torch.cuda.empty_cache()
        # preprocessing mirrored from the processor, so it follows SIGLIP_MODEL
        ip = proc.image_processor.to_dict()
        self._sig_hw = (ip["size"]["height"], ip["size"]["width"])
        self._sig_mean = torch.tensor(ip["image_mean"], device=self.device).view(1, 3, 1, 1)
        self._sig_std = torch.tensor(ip["image_std"], device=self.device).view(1, 3, 1, 1)
        self._qs = torch.tensor([0.05, 0.95], device=self.device)   # q_05 / q_95 of Eq. (11)
        self._loaded = True

    def _to_gpu(self, img_rgb):
        if not self._loaded:
            self._load()
        return img_rgb.to(self.device, non_blocking=True).float()[None] / 255.0

    def semantic_cue(self, img_rgb):
        """The semantic cue of Eq. (10): P_elev = max over the elevator classes M of
        sigmoid(l_m - logsumexp over N of l_n), each class scored against the whole negative
        pool, the best one wins."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            v = F.interpolate(self._to_gpu(img_rgb), self._sig_hw, mode="bilinear",
                              align_corners=False, antialias=True).clamp(0, 1)
            v = ((v - self._sig_mean) / self._sig_std).half()
            iemb = self.sig.get_image_features(pixel_values=v)
            iemb = iemb if torch.is_tensor(iemb) else iemb.pooler_output
            iemb = iemb.float()
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            logits = ((iemb @ self.temb.T) * self._scale + self._bias).squeeze(0)
            neg_lse = torch.logsumexp(logits[self._n_elev:], 0)
            return float(torch.sigmoid(logits[:self._n_elev] - neg_lse).max())

    def geometric_cue(self, img_rgb):
        """The geometric cue of Eq. (11): rho_sp = log((q_95 + eps) / (max(q_05, 0) + eps)) of
        the Omnidata relative depth, small inside an enclosed elevator."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t = F.interpolate(self._to_gpu(img_rgb), (512, 512), mode="bilinear",
                              align_corners=False)
            depth = self.omni.model((t - 0.5) / 0.5).squeeze().float()
            D = depth[20:-20, 20:-20]                 # drop the border ring
            q05, q95 = torch.quantile(D.reshape(-1), self._qs).tolist()
        eps = 1e-4
        return float(np.log((q95 + eps) / (max(q05, 0.0) + eps)))


class ArmedWindowFSM:
    """The armed window of Sec. 3.4 as a causal ARMED (1) / DISARMED (0) state over the
    EMA-smoothed cues; one `step` per scored frame. It opens when P_elev and rho_sp have held
    together for `sustain_in`, and closes when P_elev has stayed low for `sustain_out`.
    `events` = [(t, 0|1)] transitions stamped at decision time (after the sustain dwell), which
    `armed_at` replays for the ride detector."""

    def __init__(self, params):
        self.p = dict(params)      # config Elevator.detect.armed
        self.state = 0
        self.events = []
        self._p_ema = self._rho_ema = None   # EMA of P_elev / rho_sp
        self._t_last = None
        self._pend = None          # since when the pending transition's condition has held

    def step(self, t, p_elev, rho_sp):
        """`rho_sp` may be a zero-arg callable: the geometric cue is only evaluated when it can
        change the outcome (disarmed and the smoothed P_elev >= p_depth_wake)."""
        p = self.p
        if self._p_ema is None:
            a = None
            self._p_ema = p_elev
        else:
            a = 1.0 - np.exp(-(t - self._t_last) / p["ema_tau"])
            self._p_ema += a * (p_elev - self._p_ema)
        self._t_last = t
        if self.state == 1:
            self._rho_ema = None
        elif self._p_ema >= p["p_depth_wake"]:
            rho = rho_sp() if callable(rho_sp) else rho_sp
            self._rho_ema = rho if (a is None or self._rho_ema is None) else self._rho_ema + a * (rho - self._rho_ema)

        if self.state == 0:
            ok = (self._p_ema >= p["p_arm"] and self._rho_ema is not None
                  and self._rho_ema <= p["max_rho_sp"])
            if self._sustained(ok, t, p["sustain_in"]):
                self.state = 1
                self.events.append((t, 1))
        elif self._sustained(self._p_ema < p["p_disarm"], t, p["sustain_out"]):
            self.state = 0
            self.events.append((t, 0))
        return self.state == 1

    def _sustained(self, ok, t, dwell):
        """True once `ok` has held continuously for `dwell` seconds; resets on any break."""
        if not ok:
            self._pend = None
            return False
        if self._pend is None:
            self._pend = t
        if t - self._pend < dwell:
            return False
        self._pend = None
        return True


def armed_at(video, t):
    """Armed state at time t from the transition list the motion filter publishes on
    `video.elev_armed_events` (plain attr, same process). False before the first transition or when
    the FSM never ran."""
    ev = video.elev_armed_events
    if not ev:
        return False
    s = 0
    for te, st in ev:              # a handful of transitions per run
        if te > t:
            break
        s = st
    return s == 1


class ArmedCursor:
    """`armed_at` for a caller whose query times are monotone -- the ride FSM asks once per
    IMU sample: keeps the index of the next unconsumed transition and the state before it,
    so a query is O(1) instead of a rescan of `video.elev_armed_events`; a query that steps back in
    time restarts from the front. Same answer as `armed_at` for every t."""

    def __init__(self, video):
        self.video = video
        self._i = 0            # transitions consumed
        self._s = 0            # state after them
        self._t = -1e18        # last query

    def __call__(self, t):
        ev = self.video.elev_armed_events
        if t < self._t or self._i > len(ev):
            self._i, self._s = 0, 0
        self._t = t
        i, s = self._i, self._s
        while i < len(ev) and ev[i][0] <= t:
            s = ev[i][1]
            i += 1
        self._i, self._s = i, s
        return s == 1


_PRELOAD = {}     # the one scorer preload of this process (see start_preload)


def start_preload(device="cuda:0", image_hw=(480, 640)):
    """Begin loading the scorer on a daemon thread and warm each cue up once on a blank frame.
    demo.py calls this right after the reader starts; the load (~3 s) then overlaps the wait
    for the first frame, the system's construction and the first keyframes, and only the
    armed-window worker thread waits for it (`ArmedWindowGate._run`). Safe beside main-thread
    torch work because the model is loaded at float32 (see CueScorer._load). Idempotent."""
    if _PRELOAD:
        return
    st = _PRELOAD
    st.update(scorer=None, err=None)

    def _run():
        import torch
        try:
            scorer = CueScorer(device)
            blank = torch.zeros((3,) + tuple(image_hw), dtype=torch.uint8)
            scorer.semantic_cue(blank)
            scorer.geometric_cue(blank)
            torch.cuda.synchronize()
            st["scorer"] = scorer
        except Exception as e:               # re-raised by wait_preload, not lost here
            st["err"] = e

    st["thread"] = threading.Thread(target=_run, daemon=True)
    st["thread"].start()


def wait_preload():
    """Block until the preload is done and return its scorer; raises what the load raised.
    Without a prior start_preload the load happens here, synchronously."""
    if not _PRELOAD:
        start_preload()
    _PRELOAD["thread"].join()
    if _PRELOAD["err"] is not None:
        raise _PRELOAD["err"]
    return _PRELOAD["scorer"]


class ArmedWindowGate:
    """Live wrapper the motion filter samples once per `sample_dt` of video. The scorer comes
    from the process preload (`start_preload` / `wait_preload`); the worker waits for it on
    its first sample, so scoring starts on the same frame it always did while the frame loop
    never blocks on the load. Scoring runs on a worker thread with its own CUDA stream:
    `sample` hands the frame over and returns, the worker scores the two cues, steps the FSM
    and publishes `elev_armed` / `elev_armed_events` on the shared DepthVideo. The frame loop
    never waits on the SigLIP2 forward; the frontend calls `wait_upto` before it reads the
    transitions, so the ride FSM sees them in the order a synchronous scorer would produce.
    `elev_armed`, read by the motion filter's keyframe cadence, lags by the one sample in
    flight (a few ms on a 0.29 s cadence). The worker loads nothing: the preload's dtype hazard
    (see `start_preload`) does not apply to it."""

    def __init__(self, video, device, params):
        self.video = video
        self.device = device
        self.p = dict(params)                   # config Elevator.detect.armed
        self._last_t = -1e18
        self._scorer = None
        self._fsm = None
        self._err = None                        # worker exception, re-raised on the main thread
        self._q = queue.SimpleQueue()           # (tstamp, frame, upload event) for the worker
        self._cv = threading.Condition()
        self._inflight = deque()                # stamps handed over and not yet scored, oldest first
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        if not _PRELOAD:
            start_preload(device, tuple(video.images.shape[-2:]))

    def _run(self):
        """Worker thread: score every handed-over frame in order on its own CUDA stream."""
        import torch
        stream = torch.cuda.Stream(device=self.device)
        while True:
            tstamp, image_rgb, ready = self._q.get()
            if self._fsm is None:
                # first sample: the preload is normally long done (vis_init_wait ~0)
                try:
                    self._scorer = wait_preload()
                except Exception as e:
                    self._err = e
                    with self._cv:
                        self._inflight.popleft()
                        self._cv.notify_all()
                    continue
                self._fsm = ArmedWindowFSM(self.p)
                self.video.elev_armed_events = self._fsm.events   # shared by reference
            try:
                if ready is not None:
                    stream.wait_event(ready)     # the main stream's upload of this frame
                with torch.cuda.stream(stream):
                    if image_rgb.is_cuda:
                        # allocated on the main stream: keep its memory alive for this
                        # stream's reads after the main thread drops its reference
                        image_rgb.record_stream(stream)
                    # the geometric cue is passed unevaluated: the FSM only reads it near an
                    # arming decision, so its DPT forward stays off the per-sample cost
                    p_elev = self._scorer.semantic_cue(image_rgb)
                    self.video.elev_armed = self._fsm.step(
                        tstamp, p_elev, lambda: self._scorer.geometric_cue(image_rgb))
            except Exception as e:
                self._err = e
            finally:
                with self._cv:
                    self._inflight.popleft()
                    self._cv.notify_all()

    def sample(self, tstamp, image_rgb):
        """`image_rgb`: (3,H,W) uint8 frame, on the GPU (uploaded by the caller on the current
        stream) or on the host. No-op inside `sample_dt` of the previous sample."""
        if (tstamp - self._last_t) < self.p["sample_dt"]:
            return
        if self._err is not None:
            raise self._err
        self._last_t = tstamp
        ready = None
        if image_rgb.is_cuda:
            import torch
            ready = torch.cuda.Event()
            ready.record()                       # the upload sits on the caller's stream
        with self._cv:
            self._inflight.append(tstamp)
        self._q.put((tstamp, image_rgb, ready))

    def wait_upto(self, t):
        """Block until every sample stamped at or before t has been scored, so its transitions
        are on `video.elev_armed_events`. Cheap: at most the one sample handed over this frame, whose
        ~3 ms forward has usually finished under the feature net that ran since."""
        with self._cv:
            while self._inflight and self._inflight[0] <= t:
                if not self._worker.is_alive():
                    raise RuntimeError("armed-window gate worker died") from self._err
                self._cv.wait(timeout=1.0)
        if self._err is not None:
            raise self._err


# =============================================================================================
# Ride FSM: departure and arrival -- IMU rate, run by the tracking frontend
# =============================================================================================
DISARMED, ARMED, RIDING = "DISARMED", "ARMED", "RIDING"


class _TrailingMean:
    """Running mean of x over the trailing `win` seconds of a (t, x) stream."""

    def __init__(self, win):
        self.win = win
        self.buf = deque()
        self.sum = 0.0

    def clear(self):
        self.buf.clear()
        self.sum = 0.0

    def push(self, t, x):
        self.buf.append((t, x))
        self.sum += x
        while t - self.buf[0][0] > self.win:
            self.sum -= self.buf.popleft()[1]

    @property
    def span(self):
        return self.buf[-1][0] - self.buf[0][0]

    @property
    def mean(self):
        return self.sum / len(self.buf)


class _RestPin:
    """Per-sample verdict "may v_z be pinned to 0 now". Live ARMED and the late-arm replay each
    own one instance so both see identical verdicts."""

    def __init__(self, p):
        self.p = p
        self._acc = _TrailingMean(1.0)      # 1 s mean of a_zc: the sustained-push detector
        self._push_since = None
        self._rest_since = None
        self.pushing = False

    def step(self, t, a_zc):
        """Feed EVERY sample. A sustained push (1 s-mean |a_zc| >= ramp_acc, for at most
        ramp_max) suspends the pin so a gentle elevator ramp is not zeroed away as rest."""
        self._acc.push(t, a_zc)
        if self._acc.span < 0.8:            # under 1 s of history: no suspension
            self._push_since = None
            self.pushing = False
        elif abs(self._acc.mean) >= self.p["ramp_acc"]:
            if self._push_since is None:
                self._push_since = t
            self.pushing = t - self._push_since <= self.p["ramp_max"]
        else:
            self._push_since = None
            self.pushing = False

    def pin(self, t, rest, v_z):
        """True = zero v_z. Only while the body rests and nothing pushes; capped at depart_v
        for the first rest_clear seconds of a rest run (a real ride's v_z survives), uncapped
        after (a long rest with no push means any residual v_z is drift)."""
        if not rest or self.pushing:
            self._rest_since = None
            return False
        if self._rest_since is None:
            self._rest_since = t
        return abs(v_z) <= self.p["depart_v"] or t - self._rest_since >= self.p["rest_clear"]


class RideDetector:
    """IMU-rate ride FSM: the departure and the arrival of Sec. 3.4.

    Feed: `set_gravity` once; raw accelerometer rows via `step_imu_batch`; then one body
    attitude (+ world position) per keyframe via `update_attitude`, which drains the IMU queue
    through the FSM. Read: `in_ride` / `armed` / `depart_stamp` live, `rides` for completed
    (departure, arrival) rides, `rest_spans_upto()` for the solver's rest evidence.
    `arm_lookup` (t -> bool) is the armed window; None = ungated.
    """

    def __init__(self, params):
        self.p = p = dict(params)   # config Elevator.detect.ride

        self.state = DISARMED
        self.depart_stamp = None    # committed (back-dated) departure of the current/last ride
        self.rides = []             # completed (departure, arrival) rides, oldest first
        self.rejected = []          # (departure, t, why) rides the plausibility rules threw out
        self.rest = False           # body-rest predicate at the latest sample
        self.rest_spans = []        # closed (t0, t1) body-rest spans outside RIDING: windows
                                    # where a_z is known to be nothing but its constant error c
        self.arm_lookup = None      # armed window, t -> bool; None = always armed

        self._e_z, self._gmag = None, 9.81
        self._att = deque(maxlen=2)     # (t, Rotation) KF attitudes bracketing the IMU queue
        self._imu_q = deque()           # raw (t, ax, ay, az) awaiting an attitude bracket

        # signal chain
        self._t0 = None                 # first processed sample (warmup reference)
        self._tprev = None              # latest processed sample
        self._lp1 = self._lp2 = 0.0     # 1st / 2nd-stage low-passed a_z
        self._c = 0.0                   # the constant error c of a_z, as a running median
        self._c_buf = deque(maxlen=int(p["c_win"] / 0.2))  # a_lp sampled at 5 Hz
        self._c_last_t = -1e18
        self._quiet_since = None        # |a_z - c| < rest_acc_max since
        self._rest_open = None          # start of the currently open rest span
        self._rest_pin = _RestPin(p)

        # open-loop vertical velocity v_z + its trailing buffers
        self._v_z = 0.0
        self._v_z_win = _TrailingMean(0.5)  # smoothed v_z for the arrival criterion
        self._v_z_hist = deque()        # (t, v_z) over depart_win+backdate_max: departure rewind
        self._h = 0.0                   # open integral of v_z since arming: the rise so far
        self._h_depart = 0.0            # _h at the committed departure, so rise = _h - _h_depart
        self._h_hist = deque()          # (t, _h) over the same trail as _v_z_hist
        self._replay_buf = deque()      # (t, dt, a_lp, c, rest) trailing replay_win while
                                        # DISARMED; a_lp and c kept apart so the late-arm
                                        # replay can re-integrate against a repaired c

        # a visit = first departure .. left the elevator; it may hold several rides
        self._departed = False
        self._arrive_stamp = None       # this ride's arrival stamp
        self._above_since = None        # departure criterion: |v_z| >= depart_v since
        self._inband_since = None       # arrival criterion: |mean v_z| <= arrive_v since
        self._hold_until = None         # v_z held at 0 until (redepart_min after an arrival)
        self._need_rearm = False        # after leaving: require a FRESH arming edge

        # leaving the elevator (KF rate)
        self._arrive_pos = None         # world position at the last arrival
        self._arrive_pos_pending = False    # capture _arrive_pos at the next KF
        self._leave_since = None
        self._poshist = deque()      # (t, world position) of recent KFs: the departure check

    # ---- feed --------------------------------------------------------------------------------
    def set_gravity(self, e_z, g_mag=9.81):
        """The up axis e_z (in the frame of `update_attitude`'s R_wb) and |g|."""
        v = np.asarray(e_z, dtype=np.float64)
        self._e_z = v / np.linalg.norm(v)
        self._gmag = float(g_mag)

    def step_imu_batch(self, ts, accs):
        """Queue the raw accelerometer rows of one KF interval (~200 rows) for `update_attitude`."""
        self._imu_q.extend((t, a[0], a[1], a[2])
                           for t, a in zip(np.asarray(ts).tolist(), np.asarray(accs).tolist()))

    def update_attitude(self, t, R_wb, pos_w=None):
        """Push a KF body attitude (3x3 or scipy Rotation) and optionally its world position.
        Queued IMU rows up to t are projected onto e_z with the slerped attitude and run
        through the FSM; leaving the elevator is then evaluated at this KF."""
        t = float(t)
        rot = R_wb if isinstance(R_wb, Rotation) else Rotation.from_matrix(np.asarray(R_wb))
        self._att.append((t, rot))
        if pos_w is not None:
            self._poshist.append((t, np.asarray(pos_w, dtype=np.float64)))
            while t - self._poshist[0][0] > 10.0:
                self._poshist.popleft()
        if self._e_z is None or len(self._att) < 2:
            return
        (ta, Ra), (tb, Rb) = self._att
        if tb <= ta:                    # non-monotone stamp: keep the queue for the next KF
            return
        batch = []
        while self._imu_q and self._imu_q[0][0] <= tb:
            batch.append(self._imu_q.popleft())
        if batch:
            arr = np.asarray(batch)
            Rs = Slerp([ta, tb], Rotation.concatenate([Ra, Rb]))(np.clip(arr[:, 0], ta, tb))
            a_z = np.einsum("nij,nj->ni", Rs.as_matrix(), arr[:, 1:4]) @ self._e_z - self._gmag
            for t_s, a in zip(arr[:, 0].tolist(), a_z.tolist()):
                self._process(t_s, a)
        self._step_leave(t, pos_w)

    def finalize(self, t=None):
        """End of recording: close a ride still RIDING at t (default: the last sample)."""
        if self.state != RIDING:
            return
        self.rides.append((self.depart_stamp, self._tprev if t is None else float(t)))
        self.state = ARMED

    # ---- read --------------------------------------------------------------------------------
    @property
    def in_ride(self):
        """True between a committed departure and its confirmed arrival."""
        return self.state == RIDING

    @property
    def armed(self):
        """Armed (ARMED or RIDING): enables reversible prep such as dense keyframes before
        `in_ride` flips."""
        return self.state != DISARMED

    def rest_spans_upto(self):
        """Closed rest spans plus the currently open one: the windows over which
        `elevator.transport` fits the constant error c of a_z."""
        spans = list(self.rest_spans)
        if self._rest_open is not None and self._tprev is not None \
                and self._tprev > self._rest_open:
            spans.append((self._rest_open, self._tprev))
        return spans

    # ---- per-sample FSM ----------------------------------------------------------------------
    def _in_armed_window(self, t):
        return self.arm_lookup is None or bool(self.arm_lookup(t))

    def _process(self, t, a_z):
        """One IMU sample; a_z = the world-vertical acceleration of App. C.1, the specific
        force rotated onto e_z with gravity restored."""
        p = self.p
        if self._t0 is None:
            self._t0 = self._tprev = t
        dt = min(max(t - self._tprev, 1e-4), 0.1)
        self._tprev = t

        # signal chain: double EMA, then subtract the running median, the constant error c
        al = dt / (p["lp_tau"] + dt)
        self._lp1 += al * (a_z - self._lp1)
        self._lp2 += al * (self._lp1 - self._lp2)
        a_zc = self._lp2 - self._c
        self._rest_pin.step(t, a_zc)

        # body-rest predicate: the only use of the acceleration magnitude itself
        if abs(a_zc) < p["rest_acc_max"]:
            if self._quiet_since is None:
                self._quiet_since = t
            self.rest = t - self._quiet_since >= p["rest_dwell"]
        else:
            self._quiet_since = None
            self.rest = False

        # c may only learn where the true a_z - c is 0 (elevator AND body static): DISARMED
        # takes every sample (the 15 s median rides out gait), ARMED only resting ones, RIDING none.
        if (self.state == DISARMED or (self.state == ARMED and self.rest)) \
                and t - self._c_last_t >= 0.2:
            self._c_buf.append(self._lp2)
            self._c_last_t = t
            self._c = float(np.median(self._c_buf))

        # Rest outside RIDING opens a span where a_z is nothing but c: what `elevator.transport`
        # fits c over, and -- for the span following an arrival -- what closes the ride. Dated
        # from quiet onset but never before the arrival stamp, since a gentle arrival can keep
        # |a_z - c| under rest_acc_max through the whole deceleration tail.
        if self.rest and self.state != RIDING:
            t_q = self._quiet_since
            if self._arrive_stamp is not None:
                t_q = max(t_q, self._arrive_stamp)
            if self._rest_open is None:
                self._rest_open = t_q
        elif self._rest_open is not None:
            self.rest_spans.append((self._rest_open, t))
            self._rest_open = None

        # open-loop integral; the post-arrival hold and the rest pin bound its drift to one ride
        if self.state == DISARMED:
            self._v_z = 0.0
            self._h = 0.0
            if t - self._t0 >= p["warmup"]:     # an unlearned c cannot certify a_z - c
                self._replay_buf.append((t, dt, self._lp2, self._c, self.rest))
                while t - self._replay_buf[0][0] > p["replay_win"]:
                    self._replay_buf.popleft()
        else:
            self._v_z += a_zc * dt
            if self.state == ARMED and ((self._hold_until is not None and t < self._hold_until)
                                        or self._rest_pin.pin(t, self.rest, self._v_z)):
                self._v_z = 0.0
            self._h += self._v_z * dt
            self._v_z_win.push(t, self._v_z)
            self._v_z_hist.append((t, self._v_z))
            self._h_hist.append((t, self._h))
            while t - self._v_z_hist[0][0] > p["depart_win"] + p["backdate_max"] + 1.0:
                self._v_z_hist.popleft()
                self._h_hist.popleft()

        # transitions
        armed = self._in_armed_window(t)
        if self._need_rearm and not armed:
            self._need_rearm = False    # the window closed: its next opening is a fresh edge
        if self.state == DISARMED:
            if t - self._t0 >= p["warmup"] and armed and not self._need_rearm:
                self._arm(t)
        elif self.state == ARMED:
            if not self._departed and not armed:
                # before the first ride a closing window disarms freely; after it only leaving may
                self._reset_to_disarmed()
            elif abs(self._v_z) >= p["depart_v"]:
                if self._above_since is None:
                    self._above_since = t
                elif t - self._above_since >= p["depart_win"]:
                    self._depart(t)
            else:
                self._above_since = None
        elif abs(self._v_z_win.mean) <= p["arrive_v"]:  # RIDING
            if self._inband_since is None:
                self._inband_since = t
            elif t - self._inband_since >= p["arrive_dwell"]:
                self._arrive(t)
        else:
            self._inband_since = None

    # ---- transitions -------------------------------------------------------------------------
    def _to_armed(self):
        self.state = ARMED
        self._v_z = 0.0
        self._v_z_win.clear()
        self._above_since = self._inband_since = self._leave_since = None
        self._arrive_pos_pending = True   # re-anchor the "left the elevator" reference next KF

    def _arm(self, t):
        """DISARMED -> ARMED when the armed window opens; a late edge replays buffered IMU."""
        self._hold_until = None
        self._to_armed()
        self._replay_imu(t)

    def _arrive(self, t):
        """RIDING -> ARMED: the arrival. Publish the ride; what follows is ordinary VIO."""
        p = self.p
        # A ride whose rise stays under a floor was not an elevator: discard it (Sec. 3.4).
        rise = self._h - self._h_depart
        if abs(rise) < p["min_rise_m"]:
            self._reject(t, f"rise {rise:+.2f} m under min_rise_m")
            return
        # Arrival stamp = when the SMOOTHED v_z entered the zero band, minus the chain's lag
        # (2 tau of double-EMA group delay + half the 0.5 s mean window), never before the
        # departure. Detected late by design, so the stamp is moved back (Sec. 3.4).
        self._arrive_stamp = max(self._inband_since - (2 * p["lp_tau"] + 0.25), self.depart_stamp)
        self._hold_until = t + p["redepart_min"]
        self.rides.append((self.depart_stamp, self._arrive_stamp))
        self._to_armed()

    def _reject(self, t, why):
        """Throw the open ride out: published on `rejected` (the ride manager un-rides its
        keyframes), never on `rides`; the visit closes as after leaving the elevator."""
        self.rejected.append((self.depart_stamp, float(t), why))
        self._need_rearm = self.arm_lookup is not None
        self._reset_to_disarmed()

    def _depart(self, t, t_depart=None, h_depart=None):
        """ARMED -> RIDING: the departure. Live: move the stamp back to where v_z left the zero
        band (NOT the last instant the body was still: a robot walking as the elevator departs
        has no such instant, while v_z still oscillates about zero because gait is zero-mean),
        capped at backdate_max. The late-arm replay supplies its own uncapped `t_depart` and the
        rise integral at it."""
        p = self.p
        if t_depart is None:
            t0 = t
            for th, vh in reversed(self._v_z_hist):
                if abs(vh) <= p["arrive_v"]:
                    break
                t0 = th
            t_depart = max(t0 - 2 * p["lp_tau"], t - p["backdate_max"])
        # A rider stands still as the elevator leaves: a robot that moved depart_dh_max sideways
        # over the departure window was not in one. Refused here, before anything is committed.
        dh = self._depart_dh(t)
        if dh is not None and dh > p["depart_dh_max"]:
            self.depart_stamp = t_depart
            self._reject(t, f"moved {dh:.1f} m sideways over the departure window")
            return
        if h_depart is None:                 # the rise integral where the ride began
            h_depart = self._h_hist[0][1] if self._h_hist else self._h
            for th, hh in self._h_hist:
                if th > t_depart:
                    break
                h_depart = hh
        self.depart_stamp = t_depart
        self._h_depart = h_depart
        self._arrive_stamp = None
        self.state = RIDING
        self._above_since = self._inband_since = self._leave_since = None
        self._departed = True

    def _depart_dh(self, t):
        """Horizontal displacement of the body over the departure window ending at t, from the
        KF positions; None when fewer than two positions cover it."""
        if self._e_z is None:
            return None
        win = [(ts, pos) for ts, pos in self._poshist if t - self.p["depart_win"] - 0.5 <= ts <= t]
        if len(win) < 2:
            return None
        d = win[-1][1] - win[0][1]
        return float(np.linalg.norm(d - (d @ self._e_z) * self._e_z))

    def _replay_imu(self, t):
        """The armed window opened mid-ride: replay the IMU buffered while DISARMED to date the
        departure (Sec. 3.4, late arming), or decline and leave it to the live path."""
        p = self.p
        if not self._replay_buf:
            return
        buf = list(self._replay_buf)
        self._replay_buf.clear()

        # Pass 1 -- WHERE did v_z leave the zero band. Causal: each sample sees the c it actually
        # had, the same rest pin ARMED applies and the same zero band the live rewind uses.
        pin = _RestPin(p)
        v_z, t_d, c_pre = 0.0, None, buf[0][3]
        t_band, prewin = buf[0][0], 0.0     # in-band stretch preceding the candidate crossing
        for ts, dt, a_lp, c, rest in buf:
            v_z += (a_lp - c) * dt
            pin.step(ts, a_lp - c)
            if pin.pin(ts, rest, v_z):
                v_z = 0.0
            if abs(v_z) <= p["arrive_v"]:
                if t_d is not None:
                    t_band = ts                 # back in band: the stretch restarts here
                t_d, c_pre = None, c            # c learned while the elevator was still
            elif t_d is None:
                t_d = ts                        # candidate departure
                prewin = ts - t_band
        # Without a visible in-band stretch before the crossing, the integrator may merely be
        # escaping on a regime step c has not relearned yet.
        if t_d is None or t - t_d < p["depart_win"] or prewin < p["replay_prewin"]:
            return

        # Pass 2 -- HOW FAST, with c repaired: past the departure DISARMED's "not in a moving
        # elevator" assumption is broken, so integrate against `c_pre` from there.
        pin = _RestPin(p)
        v_z, h = 0.0, 0.0
        for ts, dt, a_lp, c, rest in buf:
            v_z += (a_lp - (c_pre if ts >= t_d else c)) * dt
            pin.step(ts, a_lp - c)
            if ts < t_d and pin.pin(ts, rest, v_z):
                v_z = 0.0                       # pinned only before the crossing, as live RIDING
            elif ts >= t_d:
                h += v_z * dt                   # the rise since the departure
        if abs(v_z) < p["depart_v"]:
            return

        self._c = c_pre
        self._c_buf.clear()
        self._c_buf.append(c_pre)
        self._v_z = v_z
        self._h = h
        self._v_z_win.clear()
        self._v_z_hist.clear()
        self._h_hist.clear()
        t_depart = t_d - 2 * p["lp_tau"]        # undo the double-EMA group delay, as live
        self._depart(t, t_depart=t_depart, h_depart=0.0)

    def _step_leave(self, t, pos_w):
        """KF rate. ARMED -> DISARMED once the robot has walked leave_dist from the last arrival
        -- the only way a visit closes. Not evaluated before the first ride: ARMED then
        follows the armed window directly, and a walk-up would false-disarm. Not evaluated
        during a ride either (see the module docstring: the horizontal estimate inside an
        elevator is not the robot's motion)."""
        if self.state != ARMED or not self._departed or self._e_z is None:
            self._leave_since = None
            return
        if pos_w is None:
            return
        pos = np.asarray(pos_w, dtype=np.float64)
        if self._arrive_pos_pending:
            self._arrive_pos = pos
            self._arrive_pos_pending = False
        d = pos - self._arrive_pos
        d_h = float(np.linalg.norm(d - (d @ self._e_z) * self._e_z))
        if d_h < self.p["leave_dist"]:
            self._leave_since = None
        elif self._leave_since is None:
            self._leave_since = t
        elif t - self._leave_since >= self.p["leave_dwell"]:
            # Leaving fires on displacement while the armed window may still be open, so the
            # next visit needs a FRESH edge: the window must close first. Ungated runs have none.
            self._need_rearm = self.arm_lookup is not None
            self._reset_to_disarmed()

    def _reset_to_disarmed(self):
        self.state = DISARMED
        self._departed = False
        self._v_z = 0.0
        self._v_z_win.clear()
        self._v_z_hist.clear()
        self._replay_buf.clear()
        self._above_since = self._inband_since = self._hold_until = self._quiet_since = None
        self._arrive_pos = None
        self._arrive_pos_pending = False
        self._leave_since = None
        self._h = 0.0
        self._h_hist.clear()
        # _poshist is kept: the next departure check needs the positions before arming
