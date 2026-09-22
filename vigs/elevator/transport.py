"""The transport state (u_k, h_k) of Sec. 3.2: how far the elevator went, and where it is written.

Everything downstream of `elevator.detect`, which only answers WHEN. Three parts, in the order
a ride passes through them:

* the SOLVER. `TransportSolver.solve_ride` is the initialization of App. C.1: it re-reads the
  ride's raw IMU under BA-converged attitudes and a frozen accelerometer bias, builds the
  world-vertical acceleration a_z, places the moment t_d at which the elevator leaves rest and
  the moment t_a at which it comes back to rest from the sustained excursions of a_z, and
  integrates Eq. (12) into (u, h) -- ONE series, ONE excursion search, ONE integration per
  ride. It borrows the DepthVideo tensors read-only and mutates no estimator state; that
  separation is what lets the detector stay open-loop while this layer uses the solution.
* the WRITERS. `seed_transport` and `resolve_transport` are the only writers of the per-KF
  transport scalars (u, h) on the shared DepthVideo, both off the same solve. This is the
  "estimated twice" of App. C.1: `seed_transport` runs when the departure is detected and
  back-fills the in-ride keyframes already born (provisional, since the arrival rest has not
  closed the ride yet), and `resolve_transport` re-integrates the whole ride once it has. u is
  clamped outside [t_d, t_a], so robot motion between the arrival and the k_1 keyframe
  integrates to nothing.
* the BA ASSEMBLY. `assemble_factors_17w` carries (u, h) through the in-loop solve in the
  17-wide state layout [pose6|vel3|bias6|u@15|h@16] of Sec. 3.2.1, and applies the departure
  and arrival constraints of Eq. (7).

Outside this module, `video.elev_h` / `video.elev_u` are also written by the KF-birth
warm-start (`depth_video.init_next_pose`), the BA retraction itself, and `elevator.ride_manager`'s
own lifecycle bookkeeping. Callers of the writers hold `video.get_lock()`.
"""
from collections import namedtuple

import numpy as np
import torch
from lietorch import SE3
from scipy.spatial.transform import Rotation, Slerp

from util.imu_utils import up_axis, up_axis_tensor
from geom.ba import (U_COL, H_COL, BASE_D, get_preint_factors_cpp, get_bias_factors_cuda,
                     get_bias_prior_factors_cuda)

# =============================================================================================
# The solver: Eq. (12), one solve per ride
# =============================================================================================
# One solved ride, as the writers below consume it. u/h are {kf: value}; h is the rise measured
# from t_d, so the caller adds the baseline h-bar of Eq. (7) its own ride departs from.
RideTransport = namedtuple("RideTransport", "rise u h")


def rest_after(det_info, t_arrive_det):
    """The arrival rest that CLOSES this ride, as (t_quiet, t_confirm): the detector's first
    `rest_spans` entry past the detected arrival, cut to its `rest_dwell` (how much quiet
    certifies rest, and so how wide the closure window is). None until one is certified,
    leaving the ride OPEN.

    It does two jobs -- it bounds the elevator's true t_a from above, so t_a can be searched up
    to it rather than over a blind fixed window, and it is the rest App. C.1 re-fits c on, the
    one that brings u back to zero.
    """
    if not det_info:
        return None
    best = None
    for r0, r1 in det_info.get("rest_spans", []):
        r0, r1 = float(r0), float(r1)
        if r1 - r0 < 0.05 or r0 < float(t_arrive_det) - 0.5:
            continue
        if r0 > float(t_arrive_det) + 15.0:
            continue
        if best is None or r0 < best[0]:
            best = (r0, min(r1, r0 + det_info["rest_dwell"]))
    return best


def _slice(series, t_lo, t_hi):
    """A view of a precomputed (t_s, a_z) over [t_lo, t_hi]. ONE `_a_z_series` per solve feeds
    both the excursion search and the integration, so they cannot disagree about the same ride
    -- building the series per consumer would give `mode='same'` smoothing different edges."""
    t_s, a_z = series
    i0 = max(0, int(np.searchsorted(t_s, t_lo)) - 1)
    i1 = min(len(t_s) - 1, int(np.searchsorted(t_s, t_hi)) + 1)
    if i1 - i0 < 20:
        return None
    return t_s[i0:i1 + 1], a_z[i0:i1 + 1]


def _solve_t_d(runs, t_hint):
    """t_d of Eq. (12), the moment the elevator leaves rest: the onset of the first sustained
    excursion of a_z, already walked back to the ramp foot by `_excursions`. With no excursion
    in the window the caller's hint stands."""
    if not runs:
        return float(t_hint)
    return float(runs[0][0])


def _solve_t_a(runs, t_arrive_det, pin):
    """t_a, the moment the elevator comes back to rest, from the excursions of a_z near the
    detected arrival: end of the last excursion up to the confirmation of `pin` (the
    `rest_after` span; the detected arrival + 0.8 s when there is none), plus a 0.10 s settle
    margin. Returns (t_a | None, misread); misread flags a candidate over 1.2 s before the
    detected arrival that no rest span corroborates -- a short-ride cruise misread the caller
    must not trust."""
    t_hi = pin[1] if pin is not None else t_arrive_det + 0.8
    stops = [b for _, b in runs if t_arrive_det - 2.5 <= b <= t_hi]
    if not stops:
        return None, False
    t_a_solved = stops[-1] + 0.10
    misread = t_a_solved < t_arrive_det - 1.2 and (pin is None or pin[0] > t_a_solved + 0.3)
    return t_a_solved, misread


def _excursions(series, t_lo, t_hi, th=0.18, smooth=0.45, mindur=0.30):
    """Sustained excursions of a_z ("the elevator is accelerating") inside [t_lo, t_hi] of an
    `_a_z_series`: |0.45 s-smoothed| > th for >= mindur (the robot's own bob is ~2 Hz, killed by
    the smoothing). Returns [(t_begin, t_end), ..]; the first begin is t_d and the last end
    bounds t_a.

    Takes the series rather than building one, for the reason `_slice` gives."""
    _ta = _slice(series, t_lo, t_hi)
    if _ta is None:
        return []
    t_s, a_z = _ta
    dt = float(np.median(np.diff(t_s)))
    w = max(1, int(smooth / dt))
    sm = np.convolve(a_z, np.ones(w) / w, mode='same')
    hot = np.abs(sm) > th
    runs, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            jj = min(j - 1, len(t_s) - 1)
            # One-signedness filter: an elevator pulse is single-signed while the robot's own
            # bob alternates sign, so its |mean| stays near zero even when |sm| clears th.
            if (t_s[jj] - t_s[i] >= mindur
                    and abs(float(np.mean(sm[i:jj + 1]))) >= 0.75 * th):
                # Walk the onset back to the ramp foot, where the acceleration -- and so the
                # elevator's velocity u -- is near zero, which is what t_d of Eq. (12) means.
                # The threshold necessarily fires part-way up a jerk-limited ramp, and anchoring
                # the integral where the elevator already moves costs that speed times the ride.
                i0 = i
                _sgn = float(np.sign(sm[i]))
                while (i0 > 0 and sm[i0 - 1] * _sgn > 0.05
                       and t_s[i] - t_s[i0 - 1] <= 1.5):
                    i0 -= 1
                # i0 == 0 means the scan stopped at the slice edge rather than the ramp
                # foot: the onset is still late by an unknown amount, so this ride's t_d is
                # under-read.
                runs.append((float(t_s[i0]), float(t_s[jj])))
            i = j
        else:
            i += 1
    return runs


def _integrate(series, t_d, kf_ck, t_a=None, rest_spans=None, rest_win=None):
    """Eq. (12): integrate a_z - c from t_d (u = 0) into (u, h), with c estimated on the rest
    windows; returns {kf: (u, h)} at the (kf, stamp) checkpoints in `kf_ck`.

    t_a enforces u = 0 at the arrival, the zero-velocity update of Sec. 3.2.2; rest_win anchors
    that closure at certified rest."""
    # Close on the certified rest window, not on the instantaneous t_a: the acceleration
    # threshold drops out before the deceleration tail ends, so the elevator is still sliding at
    # t_a and a one-point zero-velocity update there charges real rise to c. The instant form
    # below is a degraded fallback for "no rest span ever arrived".
    _rest_zupt = rest_win is not None and t_a is not None
    t_end = max(s for _, s in kf_ck)
    if t_a is not None:
        t_end = max(t_end, float(t_a))
    if _rest_zupt:
        t_end = max(t_end, float(rest_win[1]))
    # Window for c: prefer certified rest spans in the 30 s before t_d over the fixed 4 s
    # pre-window, which medians the walk into the elevator into c.
    _spans = None
    if rest_spans:
        _spans = [(max(float(r0), t_d - 30.0), min(float(r1), t_d))
                  for r0, r1 in rest_spans
                  if float(r1) > t_d - 30.0 and float(r0) < t_d]
        _spans = [s for s in _spans if s[1] - s[0] > 0.05] or None
    t_lo = (t_d - 30.5) if _spans else (t_d - 4.5)
    _ta = _slice(series, t_lo, t_end)
    if _ta is None:
        return {}
    t_s, a_z = _ta
    # c = median a_z over known rest (App. C.1): without it the constant error of a_z
    # integrates into the rise.
    c0, _from_spans = 0.0, False
    if _spans:
        m = np.zeros(len(t_s), dtype=bool)
        for r0, r1 in _spans:
            m |= (t_s >= r0) & (t_s < r1)
        if m.sum() >= 10:
            c0 = float(np.clip(np.median(a_z[m]), -0.3, 0.3))
            _from_spans = True
    if not _from_spans:
        pre = (t_s >= t_d - 4.0) & (t_s < t_d)
        if pre.sum() >= 10:
            c0 = float(np.clip(np.median(a_z[pre]), -0.3, 0.3))
    out, cks = {}, sorted(kf_ck, key=lambda x: x[1])
    u = h = 0.0
    ci = 0
    u_at_a = None
    _w_ut = _w_tt = 0.0        # rest-window least squares for the closure
    _w_n = 0
    while ci < len(cks) and cks[ci][1] <= t_d:      # checkpoints before t_d: at rest
        out[cks[ci][0]] = (0.0, 0.0)
        ci += 1
    for k in range(1, len(t_s)):
        if t_s[k] <= t_d:
            continue
        dt = float(t_s[k] - max(float(t_s[k - 1]), t_d))
        a = float(a_z[k - 1]) - c0
        h += u * dt + 0.5 * a * dt * dt
        u += a * dt
        if t_a is not None and u_at_a is None and t_s[k] >= float(t_a):
            u_at_a = u
        if _rest_zupt and float(rest_win[0]) <= t_s[k] <= float(rest_win[1]):
            _tau = float(t_s[k]) - float(t_d)
            _w_ut += u * _tau
            _w_tt += _tau * _tau
            _w_n += 1
        while ci < len(cks) and t_s[k] >= cks[ci][1]:
            out[cks[ci][0]] = (u, h)
            ci += 1
    while ci < len(cks):                               # tail KFs past the last IMU sample
        out[cks[ci][0]] = (u, h)
        ci += 1
    if t_a is not None:
        T_ride = float(t_a) - float(t_d)
        if u_at_a is None:
            u_at_a = u                                 # t_a past the last IMU sample
        dc = t_clamp = None
        if _rest_zupt and _w_n >= 10 and _w_tt > 0.0:
            dc = _w_ut / _w_tt                 # LS fit of the residual u = dc*(t - t_d)
            t_clamp = float(rest_win[0])               # at rest from the quiet onset on
        elif T_ride >= 1.0:
            dc = u_at_a / T_ride
            t_clamp = float(t_a)
        if dc is not None:
            for kk, t_ck in cks:
                if t_ck <= t_d or kk not in out:
                    continue
                d = min(t_ck, t_clamp) - float(t_d)   # clamp: flat once at rest
                uu, hh = out[kk]
                out[kk] = (uu - dc * d, hh - 0.5 * dc * d * d)
    return out


# =============================================================================================
# The estimator state the transport state is solved from
# =============================================================================================
class TransportSolver:
    """Read-only view of the DepthVideo the transport state is solved from.

    Built fresh per call (`from_video`) and thrown away: it borrows the live tensors rather than
    copying them, so it must not outlive the lock the caller holds."""
    def __init__(self, imus, poses, biass_w, kf_stamps, Rwg, init_g, Tcb):
        self.imus = imus
        self.poses = poses
        self.biass_w = biass_w
        self.kf_stamps = kf_stamps
        self.Rwg = Rwg
        self.init_g = init_g
        self.Tcb = Tcb

    @classmethod
    def from_video(cls, video):
        return cls(video.imus, video.poses, video.biass_w, video.kf_stamps,
                   video.Rwg, video.init_g, video.Tcb)

    def _a_z_series(self, t_lo, t_hi, kf_lo, kf_hi, bias_kf):
        """a_z of Eq. (12), shared by `_excursions` and `_integrate`: slice the IMU over
        [t_lo, t_hi], rotate each reading by the attitude R(t) slerped between the keyframes,
        remove the frozen accelerometer bias b_a of keyframe bias_kf, restore gravity and
        project on e_z. Returns (t_s, a_z), or None if the IMU window (< 20 samples) or the KF
        attitude track (< 2 stamps) is too short."""
        ts_all = self.imus[:, 0]
        i0 = max(0, int(np.searchsorted(ts_all, t_lo)) - 1)
        i1 = min(len(ts_all) - 1, int(np.searchsorted(ts_all, t_hi)) + 1)
        if i1 - i0 < 20:
            return None
        seg = self.imus[i0:i1 + 1]
        t_s, acc_s = seg[:, 0], seg[:, 4:7]
        T_cb = self.Tcb.matrix().squeeze(0).squeeze(0)
        ks = [k for k in range(int(kf_lo), int(kf_hi) + 1) if k in self.kf_stamps]
        if len(ks) < 2:
            return None
        Rt = [float(self.kf_stamps[k]) for k in ks]
        # one batched SE3->matrix->inverse for the whole attitude track, not one per KF
        T_cw = SE3(self.poses[ks][None]).matrix()[0]
        Rmats = (torch.linalg.inv(T_cw) @ T_cb)[:, :3, :3]
        sl = Slerp(np.asarray(Rt),
                   Rotation.from_matrix(Rmats.detach().cpu().numpy()))
        R_s = sl(np.clip(t_s, Rt[0], Rt[-1]))
        ba = self.biass_w[int(bias_kf), 3:6].detach().cpu().numpy().astype(np.float64)
        g_w = self.Rwg @ np.asarray(self.init_g, dtype=np.float64)
        _ez = up_axis(self.Rwg, self.init_g)
        a_z = (R_s.apply(acc_s - ba) + g_w) @ _ez
        return t_s, a_z

    def solve_ride(self, t_depart_det, t_arrive_det, kf_lo, kf_hi, bias_kf, kf_ck,
                     det_info=None):
        """Solve ONE ride: one `_a_z_series`, one excursion search, one integration of Eq. (12).

        `t_depart_det` / `t_arrive_det` are the detector's stamps -- hints that say where to
        look, never the answer; the solve places its own t_d and t_a. The ride is CLOSED when
        the detector has certified a rest span past the arrival (`rest_after`); until then the
        closure is one-sided and t_a unbounded, so the caller should wait rather than solve
        twice."""
        spans = [(float(r0), float(r1)) for r0, r1 in (det_info or {}).get("rest_spans", [])
                 if float(r1) > float(t_depart_det) - 30.0
                 and float(r0) < float(t_arrive_det) + 15.0]
        pin = rest_after(det_info, t_arrive_det)
        t_lo = min([float(t_depart_det) - 4.5] + [r0 for r0, _ in spans]) - 0.5
        t_hi = max([float(t_arrive_det) + 1.5] + [r1 for _, r1 in spans]
                   + [float(t) for _, t in kf_ck]) + 0.5
        klo, khi = int(kf_lo), int(kf_hi)
        while klo - 1 >= 1 and float(self.kf_stamps.get(klo - 1, -1e18)) >= t_lo:
            klo -= 1
        while (khi + 1) in self.kf_stamps and float(self.kf_stamps[khi + 1]) <= t_hi:
            khi += 1
        series = self._a_z_series(t_lo, t_hi, klo, khi, bias_kf)
        if series is None:
            return None
        # Search window for the excursions: from 1.5 s before the detected departure up to the
        # arrival bound -- the detected arrival + 1 s, stretched to the rest span's confirmation
        # when there is one, since the true t_a can trail the detector's stamp.
        t_hi_arrive = float(t_arrive_det) + 1.0
        if pin is not None:
            t_hi_arrive = max(t_hi_arrive, pin[1] + 0.2)
        runs = _excursions(series, float(t_depart_det) - 1.5, t_hi_arrive)
        t_d_solved = _solve_t_d(runs, t_depart_det)
        t_a_solved, misread = _solve_t_a(runs, float(t_arrive_det), pin)
        if pin is None:
            # OPEN ride: no certified arrival rest, so no closure and no upper clamp. The
            # elevator is still moving at the newest keyframe, and zeroing u there would tell
            # the dead-reckoned warm start it had stopped. t_a only places the sentinel.
            t_a_solved, t_hi_clamp = max(float(t) for _, t in kf_ck), float('inf')
        else:
            # A cruise misread says nothing about t_a: fall back to the detector's stamp.
            t_a_solved = float(t_arrive_det) if misread else (pin[0] if t_a_solved is None else t_a_solved)
            # Read t_a at certified rest, not where the acceleration threshold dropped out.
            t_a_solved = t_hi_clamp = max(float(t_a_solved), float(pin[0]))
        # Sentinel checkpoint -1 books the full rise even when no keyframe lands past t_a.
        ck = list(kf_ck) + [(-1, float(t_a_solved))]
        # An open ride gets no closure: a zero-velocity update at a keyframe the elevator is
        # still moving through would charge the rise to c.
        uh = _integrate(series, t_d_solved, ck, t_a=(t_a_solved if pin is not None else None),
                        rest_spans=spans, rest_win=pin)
        if not uh:
            return None
        rise = uh[-1][1] if -1 in uh else uh[max(k for k, _ in kf_ck if k in uh)][1]
        u, h = {}, {}
        for k, t in kf_ck:
            if k not in uh:
                continue
            u[k] = 0.0 if t <= t_d_solved or t >= t_hi_clamp else float(uh[k][0])
            h[k] = 0.0 if t <= t_d_solved else float(
                min(uh[k][1], rise, key=abs) if (rise and pin is not None) else uh[k][1])
        return RideTransport(rise=float(rise), u=u, h=h)


# =============================================================================================
# The two writers of (u, h) on the shared DepthVideo
# =============================================================================================
# The 17-wide latch `video.elev_transport_on` is already set whenever either writer runs, so both
# write u unconditionally and velos_w holds the elevator-frame v^E throughout.
def seed_transport(video, depart, upto, t_depart_det, det_info=None):
    """Eq. (12) at the detected departure: back-fill (u, h) for the in-ride KFs already born
    when the ride was detected (detection lags the departure by one frontend cycle).

    The arrival rest has not closed the ride yet, so this is the first of the two estimates of
    App. C.1: `resolve_transport` overwrites every value here once it has."""
    _ez = up_axis_tensor(video, video.poses.device)
    T_cb = video.Tcb.matrix().squeeze(0).squeeze(0)
    T_bc = torch.linalg.inv(T_cb)

    def _T_wb(j):   # imu->world of KF j (poses store w2c)
        T_cw = SE3(video.poses[j][None]).matrix()[0]
        return torch.linalg.inv(T_cw) @ T_cb

    _cks = [(j, float(video.kf_stamps[j])) for j in range(int(depart), int(upto))
            if j in video.kf_stamps]
    sol = None
    if video.imus is not None and video.Rwg is not None and _cks:
        sol = TransportSolver.from_video(video).solve_ride(
            float(t_depart_det), _cks[-1][1], int(depart), int(upto) - 1, int(depart), _cks,
            det_info=det_info)
    # sol is None when there is no solution yet (short IMU/attitude window, or no excursion of
    # a_z): fall through to the visual per-edge rise for h, u = v^W . e_z.
    if sol is not None:
        # Keep this ride's h-bar of Eq. (7): the solver's h is measured from ITS OWN t_d.
        _h_base = float(video.elev_h[int(depart)].item()) - sol.h.get(int(depart), 0.0)
        # A nonzero u at k_0 means t_d precedes it, i.e. the elevator was already moving there;
        # this is the u-bar of Eq. (7) and the seeded u carries it.
        video.elev_u[int(depart)] = sol.u.get(int(depart), 0.0)

    _h_prev_orig = None   # Pre-strip world height of KF j-1, since poses[j-1] gets flattened to
                          # departure height and (h_j - h_stripped_prev) is not the per-edge rise.
    for j in range(int(depart) + 1, int(upto)):
        T_wb_prev, T_wb_j = _T_wb(j - 1), _T_wb(j)
        _h_j = float(T_wb_j[:3, 3] @ _ez)
        _dh = _h_j - float(T_wb_prev[:3, 3] @ _ez)          # strip amount (vs possibly-stripped prev)
        _dh_edge = (_h_j - _h_prev_orig) if _h_prev_orig is not None else _dh   # true per-edge rise
        _h_prev_orig = _h_j
        # the same pose-vertical strip the birth seeding would have applied
        T_wb_j[:3, 3] = T_wb_j[:3, 3] - _dh * _ez
        T_cw_new = torch.linalg.inv(T_wb_j @ T_bc)
        _q = Rotation.from_matrix(T_cw_new[:3, :3].detach().cpu().numpy()).as_quat()
        video.poses[j] = torch.cat([
            T_cw_new[:3, 3].detach().cpu(),
            torch.tensor(_q, dtype=video.poses.dtype)]).to(video.poses.device)
        _u = 0.0
        if sol is not None and j in sol.h:
            video.elev_h[j] = _h_base + sol.h[j]
            _u = sol.u[j]
        else:
            video.elev_h[j] = video.elev_h[j - 1] + _dh_edge
            _u = float(video.velos_w[j] @ _ez)
        _vw = video.velos_w[j]
        video.elev_u[j] = _u
        video.velos_w[j] = _vw - float(_vw @ _ez) * _ez


def resolve_transport(video, depart, arrive, t_depart_det, t_arrive_det, det_info=None):
    """The ride's ONE real measurement: the re-integration of App. C.1, run once the arrival
    rest closes the ride.

    Re-solves Eq. (12) over KF[k_0, k_1] under the converged attitudes and with c fitted to the
    rest spans on BOTH sides, then writes (u, h) for every keyframe of the ride. Returns the
    change in h[k_1] so the caller can shift the post-arrival rows that were born carrying the
    provisional value, or None if no solution was reached.

    u is clamped to zero outside [t_d, t_a], so the KFs between the arrival and k_1 come back
    with u = 0 and h flat by construction.

    One deliberate approximation: k_1 is placed off the detector's arrival stamp, which the
    solver routinely moves later, so that keyframe can sit inside the deceleration tail.
    h[k_1] is what the whole post-arrival world hangs on -- the fold lifts by it and
    `assemble_factors_17w` pins every later row to it -- so it takes the full rise regardless.
    The cost is that this one keyframe is placed at the elevator's final height instead of its
    own, bounded by the rise left in the tail (~0.2 m for a 1 s lag)."""
    if depart is None or arrive is None or video.imus is None or video.Rwg is None:
        return None
    k0, k1 = int(depart), int(arrive)
    ks = [(k, float(video.kf_stamps[k])) for k in range(k0, k1 + 1) if k in video.kf_stamps]
    if len(ks) < 3:
        return None
    sol = TransportSolver.from_video(video).solve_ride(
        float(t_depart_det), float(t_arrive_det), k0, k1, k0, ks, det_info=det_info)
    if sol is None:
        return None
    h_base = float(video.elev_h[k0].item()) - sol.h.get(k0, 0.0)   # keep h-bar of Eq. (7)
    h_old = float(video.elev_h[k1].item())
    for k, _ in ks:
        video.elev_h[k] = h_base + sol.h[k]
        video.elev_u[k] = sol.u[k]
    # The detector's arrival stamp can precede t_a, so k_1 may sit inside the deceleration
    # tail; it is the arrival baseline either way, and h[k_1] is what the fold lifts by.
    video.elev_h[k1] = h_base + sol.rise
    video.elev_u[k1] = 0.0
    return float(video.elev_h[k1].item()) - h_old


# =============================================================================================
# In-loop BA assembly of the 17-wide state, with the constraints of Eq. (7)
# =============================================================================================
_ConstraintRows = namedtuple("_ConstraintRows", "out_rows out_nodes hbar_src is_k0 arr_rows arr_nodes")


def _constraint_rows(video, ks, depart, arrive, pending, dev):
    """Row selection for the departure and arrival constraints of Eq. (7) and the weighted terms
    of App. C.2, cached on `video` under the (edge list, ride markers) it was built for. Pure
    index work: nothing here reads h or u.

      out_rows   edge rows whose j node is outside every ride (they get the h and u constraints)
      out_nodes  their j nodes
      hbar_src   per such row, the k_1 whose h[k_1] is its h target; -1 = none (target 0).
                 `pending` is sorted and later rides overwrite earlier ones
      is_k0      per such row, whether the j node is a k_0 (u target = its own u, the u-bar of
                 Eq. (7)); None when there is no k_0
      arr_rows   in-ride edge rows whose j node is k_1, the arrival constraint (u = 0)
      arr_nodes  their j nodes
    A field is None when its set is empty."""
    key = (ks.tobytes(), None if depart is None else int(depart),
           None if arrive is None else int(arrive), tuple((int(a), int(b)) for a, b in pending))
    cache = getattr(video, "_constraint_rows_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    if depart is not None:
        riding = ks > depart
        if arrive is not None:
            riding &= ks <= arrive
    else:
        riding = np.zeros(len(ks), dtype=bool)
    for (p0, p1) in pending:
        riding |= (ks > p0) & (ks <= p1)           # rows of a ride awaiting its fold stay free
    nr = np.nonzero(~riding)[0]
    out_rows = out_nodes = hbar_src = is_k0 = None
    if len(nr):
        out_np = ks[nr]
        out_rows = torch.as_tensor(nr, device=dev)
        out_nodes = torch.as_tensor(out_np, device=dev)
        if pending:
            g = np.full(len(nr), -1, dtype=np.int64)
            for (p0, p1) in pending:
                g[out_np > p1] = int(p1)
            hbar_src = torch.as_tensor(g, device=dev)
        k0s = [int(p0) for (p0, p1) in pending]
        if depart is not None:
            k0s.append(int(depart))
        if k0s:
            is_k0 = torch.as_tensor(np.isin(out_np, k0s), device=dev)
    arr_rows = arr_nodes = None
    if arrive is not None:
        ar = np.nonzero(riding & (ks == arrive))[0]
        if len(ar):
            arr_rows = torch.as_tensor(ar, device=dev)
            arr_nodes = torch.as_tensor(ks[ar], device=dev)
    sel = _ConstraintRows(out_rows, out_nodes, hbar_src, is_k0, arr_rows, arr_nodes)
    video._constraint_rows_cache = (key, sel)
    return sel


def assemble_factors_15w(poses_bw, velos_w, biass_w, integrators, iii, jjj, iii_cpu, jjj_cpu,
                         info2s, Rwg, preint_scale):
    """The base system's 15-wide [pose6|vel3|bias6] joint factors, used whenever no ride is live
    or awaiting its fold (video.elev_in_ride False) -- "before the departure and after the fold
    it is exactly VIGS-SLAM's 15-dimensional system" (Sec. 3.2.1). Every row then has
    h = u = 0, so the folded poses are the poses and v^E is v^W: this is assemble_factors_17w
    with its two transport columns -- which the constraints pin to zero in that state anyway --
    left out, block for block, in the kernel's 10 + 5 block layout."""
    ps = preint_scale
    Hii, Hij, Hji, Hjj, vi, vj, _ = get_preint_factors_cpp(
        poses_bw, velos_w, biass_w, integrators, Rwg, iii_cpu, jjj_cpu, preint_scale=ps)
    Hbii, Hbij, Hbji, Hbjj, vbi, vbj = get_bias_factors_cuda(
        biass_w, iii, jjj, preint_scale=ps, info2s=info2s, D=BASE_D)
    Hbpii, vbpi = get_bias_prior_factors_cuda(biass_w, iii, preint_scale=ps, D=BASE_D)
    # Hbpii twice: the kernel's fixed 10-block index layout has two (i,i) prior slots
    Hint = torch.cat([Hii, Hij, Hji, Hjj, Hbii, Hbij, Hbji, Hbjj, Hbpii, Hbpii])
    vint = torch.cat([vi, vj, vbi, vbj, vbpi])
    return Hint, vint


def assemble_factors_17w(video, poses_bw, velos_w, biass_w, integrators,
                         iii, jjj, iii_cpu, jjj_cpu, info2s, t1, preint_scale):
    """Assemble the 17-wide [pose6|vel3|bias6|u@15|h@16] joint factors of Sec. 3.2.1, where
    vision, IMU and (v^E, u, h) are never separated.

      * velos_w stores v^E; the IMU factor substitutes Eq. (4), v^W = v^E + u*e_z
        (u = video.elev_u, col 15), which is what puts the transport columns of Eq. (5) and
        Eq. (6) into the inertial residuals.
      * The collinearity of v^E_z and u -- the reason for the update projection of Eq. (8) --
        is left unresolved here; `depth_video.inertial_ba` applies the projection.
      * Eq. (7): h = u = 0 at k_0 and on every row outside a ride, u = 0 at k_1. There is no
        post-arrival region while riding (arrive is None)."""
    ps = preint_scale
    w = 1e8                               # the weight w of Eq. (7)
    # frozen up axis e_z (Rwg is held fixed during a ride)
    e_z_t = up_axis_tensor(video, video.elev_h.device)
    # fold h into the world poses for the residual (pure translation: t_bw -= h*(R_bw @ e_z))
    h_win = video.elev_h[:t1]
    u_win = video.elev_u[:t1]
    R_bw = poses_bw.matrix()[0, :, :3, :3]
    Rbw_e_z = torch.einsum('nij,j->ni', R_bw, e_z_t)
    folded = poses_bw.data.clone()
    folded[0, :, :3] = folded[0, :, :3] - h_win[:, None] * Rbw_e_z
    poses_bw_folded = SE3(folded)
    # world velocity v^W = v^E + u*e_z for the IMU factor, Eq. (4) (velos_w stores v^E)
    vw = velos_w.clone()
    vw[0, :, :] = vw[0, :, :] + u_win[:, None] * e_z_t[None, :]
    # 17-wide IMU factors at the folded poses and world velocity (batched C++ builder)
    Hii, Hij, Hji, Hjj, vi, vj, _ = get_preint_factors_cpp(
        poses_bw_folded, vw, biass_w, integrators, video.Rwg,
        iii_cpu, jjj_cpu, transport_columns=True, preint_scale=ps)
    # Bias and bias-prior come out 17-wide from the kernels: both write only the 9:15 block
    # of a zero-initialised (num,D,D), so u@15 / h@16 are zero at no cost.
    Hbii, Hbij, Hbji, Hbjj, vbi, vbj = get_bias_factors_cuda(
        biass_w, iii, jjj, preint_scale=ps, info2s=info2s)
    Hbpii, vbpi = get_bias_prior_factors_cuda(biass_w, iii, preint_scale=ps)
    # Eq. (7) and App. C.2 on the j diagonal: h = u = 0 outside a ride (after an arrival h
    # holds at h[k_1]), u = 0 at k_1; v^E stays free on all axes.
    depart, arrive = video.elev_depart_idx, video.elev_arrive_idx
    _pending = sorted(video.elev_pending_rides)
    # Every ride ends at its own arrival, so a keyframe of an intermediate-floor stop is never
    # inside a ride window and takes the `not riding` branch below (h pinned to h-bar, u to 0).
    _ks = jjj_cpu                                        # (N,) int64: j node of each edge
    _dev = Hjj.device
    # Which rows get a constraint, and which KF each targets, depends only on the edge list and
    # the ride markers, which hold still across the BA iterations of a keyframe: those index
    # tensors are built once per (edge list, markers) and reused. The h/u VALUES they read are
    # gathered live on the device below, since the retraction moves them every iteration and
    # pulling them back would cost two stream syncs per iteration.
    _sel = _constraint_rows(video, _ks, depart, arrive, _pending, _dev)
    if _sel.out_rows is not None:               # h target (pre -> 0, post -> h-bar); u = 0
        _rows, _nodes = _sel.out_rows, _sel.out_nodes
        h_row, u_row = h_win[_nodes].double(), u_win[_nodes].double()
        # h-bar of Eq. (7) = h[k_1] of the last arrived-but-unfolded ride before k (0 if none)
        if _sel.hbar_src is None:
            _tgt = torch.zeros(len(_rows), dtype=torch.float64, device=_dev)
        else:
            _tgt = torch.where(_sel.hbar_src >= 0, h_win[_sel.hbar_src.clamp(min=0)].double(), 0.0)
        Hjj[0, _rows, H_COL, H_COL] += ps * w
        vj[0, _rows, H_COL] += (ps * w * (_tgt - h_row)).to(vj.dtype)
        Hjj[0, _rows, U_COL, U_COL] += ps * w
        # u target: 0 on non-ride rows except each k_0, whose elev_u already carries the u-bar
        # of Eq. (7), the velocity the elevator has already reached there (zero residual).
        _utgt = torch.zeros(len(_rows), dtype=torch.float64, device=_dev)
        if _sel.is_k0 is not None:
            _utgt = torch.where(_sel.is_k0, u_row, _utgt)
        vj[0, _rows, U_COL] += (ps * w * (_utgt - u_row)).to(vj.dtype)
    if _sel.arr_rows is not None:         # arrival constraint at k_1: u = 0 (h stays free)
        Hjj[0, _sel.arr_rows, U_COL, U_COL] += ps * w
        vj[0, _sel.arr_rows, U_COL] += (-ps * w * u_win[_sel.arr_nodes].double()).to(vj.dtype)
    # Hbpii twice: the kernel's fixed 10-block index layout has two (i,i) prior slots
    # (src/vigs_kernels.cu `ind1_all` / `ind2_all`).
    Hint = torch.cat([Hii, Hij, Hji, Hjj, Hbii, Hbij, Hbji, Hbjj, Hbpii, Hbpii])
    vint = torch.cat([vi, vj, vbi, vbj, vbpi])
    return Hint, vint
