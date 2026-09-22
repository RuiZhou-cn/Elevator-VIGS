"""The frontend's ride lifecycle: pin k_0 at the departure, pin k_1 at the arrival, hold the
rise in h past the arrival, and run the deferred fold of Sec. 3.2.4 once the ride has left the
sliding window.

`ElevatorRide` owns all ride state and touches the frontend only through the arguments it is
handed (`t1`, the factor graph). It runs the ride detector inside the armed window published by
the cue FSM (both in `elevator.detect`) and delegates the (u, h) solve to `elevator.transport`.
The frontend calls, per keyframe and in this order:

    elev_in = observe_kf(t1, is_last)   # feed the detector, read the ride flag
    step(elev_in, t1)                   # departure / arrival (after IMU init)
    fold_poll(t1, graph)                # fold rides that left the window (inside the BA loops)
    solve_poll()                        # re-solve an arrived ride once its arrival rest closes
    keep_kf(cand, elev_in)              # veto keyframe removal inside a ride

and `finalize_fold()` once at the end of the run.
"""
import numpy as np
import lietorch

from util.imu_utils import up_axis, up_axis_tensor
from .detect import RideDetector, ArmedCursor, armed_at
from .transport import rest_after
from . import transport


class ElevatorRide:
    """The ride state of one run; see the module docstring for the call order."""

    def __init__(self, video, elev_cfg, frontend_window):
        self.video = video
        self.init_g = video.init_g
        self.frontend_window = frontend_window
        self._pending_rides = []    # arrived rides awaiting their fold, oldest first
        self._depart_idx = None     # k_0, captured at the departure
        self._deferred = False      # the deferred fold of Sec. 3.2.4: the rise stays in h, not
                                    # in the poses, until the ride leaves the sliding window
        self._pending_depart_stamp = None   # stamps, not indices: removals shift indices
        self._pending_arrive_stamp = None

        # The detector is the only source of ride windows -- the path is always live, no config
        # says whether a ride is coming. Its thresholds are config Elevator.detect.ride.
        self._detector = RideDetector(elev_cfg["detect"]["ride"])
        # The armed window gates the IMU departure/arrival FSM (Sec. 3.4). Load-bearing: an
        # ungated IMU FSM produces catastrophic false rides. The cursor is `armed_at` with
        # O(1) queries for the FSM's monotone IMU-sample times.
        self._detector.arm_lookup = ArmedCursor(self.video)
        self._imu_cursor = 0       # ride detector: video.imus rows already fed
        self._up_axis_set = False
        self._unsolved = None       # (k_0 stamp, k_1 stamp, t_depart_det, t_arrive_det) of an
                                    # arrived ride not re-solved yet: the solve waits for the
                                    # arrival rest that App. C.1 re-fits c on
        self._tcb_np_cache = None       # _tcb_np()
        self._kf_inv, self._kf_inv_key = {}, None   # _kf_inv_map()
        self._rejected_seen = 0         # detector.rejected entries already released

    # ---- per-keyframe hooks (TrackFrontend.__update) -------------------------------------------
    def observe_kf(self, t1, is_last):
        """Step the detector on the newest KF (t1-1), before the BA, so k_0 is already in-ride
        and the IMU factors cannot misread the elevator's acceleration as the robot's. Returns
        the detector's ride flag."""
        _st = float(self.video.kf_stamps.get(t1 - 1, -1.0))
        if _st > 0 and self.video.armed_gate is not None:
            # the cues are scored on a worker thread: the armed-window transitions up to this
            # KF must be published before the detector, keep_kf and ride_hint read them
            self.video.armed_gate.wait_upto(_st)
        if self.video.IMU_initialized and _st > 0:
            self._feed_detector(t1 - 1, _st, is_last=is_last)
        return self._detector.in_ride

    def keep_kf(self, cand, elev_in):
        """True when KF `cand` must NOT be culled by the frontend's redundancy test -- the
        "removes no keyframe as redundant" of App. C.4:
        * dense ride keyframes (in-ride, or inside the armed window at its stamp): culling back
          to the normal cadence makes IMU edges too long to see the rise;
        * k_0 and k_1, where Eq. (7) constrains the transport state: low-motion rest frames are
          prime removal candidates, but dropping one loses a constraint;
        * anything inside a ride awaiting its fold: dropping its stamp would leave fold_poll
          unable to resolve the ride, silently releasing the hold."""
        _cand_t = self.video.kf_stamps.get(cand)
        if elev_in or self._detector.armed or (
                _cand_t is not None and armed_at(self.video, _cand_t)):
            return True
        if cand == self._depart_idx or cand == self.video.elev_arrive_idx:
            return True
        if self._deferred and self._pending_arrive_stamp is not None:
            _k1 = self._kf_inv_map().get(self._pending_arrive_stamp)
            if _k1 is not None and cand <= _k1:
                return True
        return False

    # ---- helpers -------------------------------------------------------------------------------
    def _tcb_np(self):
        """cam->imu extrinsic as a cached (4, 4) numpy array; Tcb is fixed at init."""
        if self._tcb_np_cache is None:
            self._tcb_np_cache = self.video.Tcb.matrix().squeeze(0).squeeze(0).detach().cpu().numpy()
        return self._tcb_np_cache

    def _kf_inv_map(self):
        """Cached stamp(round-4) -> KF index, keyed on the KF count so it invalidates on append
        and on index-shifting removals."""
        d = self.video.kf_stamps
        n = int(self.video.counter.value)
        key = (n, d.get(n - 1))
        if self._kf_inv_key != key:
            self._kf_inv = {round(float(v), 4): k for k, v in d.items()}
            self._kf_inv_key = key
        return self._kf_inv

    def _kf_stamp_arrays(self):
        """(indices, stamps) of every KF, index-sorted."""
        ks = self.video.kf_stamps
        idx = np.array(sorted(ks.keys()))
        return idx, np.array([ks[i] for i in idx])

    def _feed_detector(self, kf, stamp, is_last=False):
        """Set the up axis once after IMU init, batch-feed IMU rows up to this KF's stamp, then
        push its attitude and position (slerp bracketing + the "left the elevator" test)."""
        det = self._detector
        imus = self.video.imus
        ts = imus[:, 0]
        j = int(np.searchsorted(ts, stamp, side="right"))
        if not self._up_axis_set:
            det.set_gravity(up_axis(self.video.Rwg, self.init_g),
                            float(np.linalg.norm(self.init_g)))
            self._up_axis_set = True
            # Skip pre-init IMU rows: through a stale attitude they read as rest and fake a ride.
        elif j > self._imu_cursor:
            i0 = self._imu_cursor
            det.step_imu_batch(ts[i0:j], imus[i0:j, 4:7])
        self._imu_cursor = j
        Tcw = lietorch.SE3(self.video.poses[kf][None]).matrix()[0].detach().cpu().numpy()
        Twc = np.linalg.inv(Tcw)
        # Camera-in-world position: leave_dist is metres and the cam->IMU lever arm is
        # centimetres, so the body/camera distinction does not matter here.
        det.update_attitude(stamp, (Twc @ self._tcb_np())[:3, :3], Twc[:3, 3])
        if is_last:
            # Last keyframe of the run: only matters if recording ends mid-ride with no arrival
            # ever confirmed. Close the open ride here so `elev_in` reads False and the arrival
            # path runs on this KF.
            det.finalize(stamp)
        # Densification hint for the motion filter: the armed window opens seconds before the
        # departure, so densifying from arming puts k_0 and the first ride edges on the 1 s
        # cadence of App. C.4.
        self.video.ride_hint = det.armed or armed_at(self.video, stamp)

    def _det_info(self):
        """Detector-certified rest spans: the windows `elevator.transport` fits the constant
        error c of a_z over, the first past an arrival closing that ride -- and the dwell that
        certified them, which bounds each closure window."""
        return dict(rest_spans=self._detector.rest_spans_upto(),
                    rest_dwell=self._detector.p["rest_dwell"])

    def _kf_atmost(self, t_label):
        """Last KF at or before t_label: k_0 of Eq. (7), "the last keyframe at or before the
        departure". Nearest-stamp matching could snap past it onto an already-rising frame and
        collapse the rise."""
        idx, stamps = self._kf_stamp_arrays()
        atmost = idx[stamps <= t_label]
        if len(atmost):
            return int(atmost[-1])
        return int(idx[np.argmin(np.abs(stamps - t_label))])

    def _kf_atleast(self, t_label):
        """First KF at or after t_label: k_1 of Eq. (7), "the first keyframe at or after the
        detected arrival". Falls back to the nearest KF when none follow."""
        idx, stamps = self._kf_stamp_arrays()
        atleast = idx[stamps >= t_label]
        if len(atleast):
            return int(atleast[0])
        return int(idx[np.argmin(np.abs(stamps - t_label))])

    def _depart_stamp(self):
        """The detector's committed departure stamp, moved back to where the elevator left the
        zero-velocity band."""
        return self._detector.depart_stamp

    def _arrive_stamp(self):
        """Arrival stamp of the detector's newest completed ride; None before the first one."""
        rides = self._detector.rides
        return rides[-1][1] if rides else None

    # ---- ride state machine ---------------------------------------------------------------------
    def step(self, elev_in, t1):
        """Drive the ride state machine: departure, then arrival. elev_transport_on latches the
        17-wide state on for the rest of the run; elev_in_ride toggles only the ride window
        (h free, plus the constraints of Eq. (7) at k_0 and k_1)."""
        if not self.video.elev_transport_on:
            # Latch the 17-wide assembly on for the rest of the run (h/u hard-pinned 0
            # outside a ride).
            self.video.elev_transport_on = True
        det = self._detector
        if len(det.rejected) > self._rejected_seen:
            # The detector discarded the open ride (its rise stayed under a floor): un-ride its
            # keyframes, then this KF is ordinary VIO.
            self._rejected_seen = len(det.rejected)
            if self.video.elev_in_ride and self.video.elev_arrive_idx is None \
                    and self._depart_idx is not None:
                self._release()
            return
        # A second ride can depart while the previous one still awaits its fold, so run the
        # departure path even when k_1 is already set.
        if elev_in and (not self.video.elev_in_ride or self.video.elev_arrive_idx is not None):
            self._on_departure(t1)          # departure: pin k_0, free h
        elif (not elev_in) and self.video.elev_in_ride and self.video.elev_arrive_idx is None:
            self._on_arrival()              # arrival: pin k_1, enter the deferred-fold hold
            if not self._deferred:
                # _on_arrival returned early (no arrival stamp), so there is no hold to hand the
                # markers over to; on a real arrival they stay set, because the hold reads them.
                self.video.elev_depart_idx = None
                self.video.elev_arrive_idx = None
                self.video.elev_in_ride = False

    def _on_departure(self, t1):
        """Departure: pin k_0 and turn on the in-loop joint inertial solve, i.e. the 17-wide
        (v^E, u, h) assembly with the departure constraint of Eq. (7) at k_0."""
        if not self.video.kf_stamps:
            return
        _depart_t = self._depart_stamp()
        if _depart_t is None:
            return                # detector committed no departure stamp: nothing to act on
        depart = self._kf_atmost(_depart_t)    # k_0: the last keyframe at or before it
        self._depart_idx = depart
        self.video.elev_depart_idx = depart
        self.video.elev_arrive_idx = None
        # Clear h/u for the first ride only: a ride departing while an earlier one still awaits
        # its fold must keep that ride's h chain and its h[k_1] baseline (h-bar) intact.
        if not self._pending_rides:
            self.video.elev_h[:] = 0.0
            self.video.elev_u[:] = 0.0
        # Detection trails KF creation by one cycle, so KFs past k_0 already exist unseeded;
        # back-fill them under the lock (the PGBA thread runs concurrently).
        with self.video.get_lock():
            transport.seed_transport(self.video, depart, t1,
                                 self._depart_stamp() or self.video.kf_stamps.get(depart, 0.0),
                                 det_info=self._det_info())
        self.video.elev_in_ride = True

    def _release(self):
        """A discarded ride is un-ridden: every keyframe from its k_0 gets its own h back as a
        world translation along e_z (what the transport-augmented residuals saw, exactly the
        fold rule of Eq. (9)) and its u back into v^W, h/u return to the h-bar that k_0
        carried, and the ride markers fall back to the previous ride awaiting its fold, or to
        none. No entry in `rides`, no fold record, no submap: the map sees the poses move like
        any other rewrite."""
        depart = int(self._depart_idx)
        e_z_t = up_axis_tensor(self.video, self.video.poses.device)
        with self.video.get_lock():
            n = int(self.video.counter.value)
            h, u = self.video.elev_h, self.video.elev_u
            h_base = float(h[depart].item())         # h-bar inherited from an earlier ride
            for k in range(depart, n):
                hk = float(h[k].item()) - h_base
                if abs(hk) > 1e-9:
                    R_cw = lietorch.SE3(self.video.poses[k][None]).matrix()[0, :3, :3]
                    self.video.poses[k, :3] = self.video.poses[k, :3] - hk * (R_cw @ e_z_t)
                uk = float(u[k].item())
                if abs(uk) > 1e-9:
                    self.video.velos_w[k] = self.video.velos_w[k] + uk * e_z_t.to(self.video.velos_w.dtype)
            h[depart:n] = h_base
            u[depart:n] = 0.0
            self.video.gs_poses_rewritten = True      # the map must see these poses move
            if self._pending_rides:
                # an earlier ride still awaits its fold: its markers come back
                inv = self._kf_inv_map()
                k0, k1 = inv.get(self._pending_rides[-1][0]), inv.get(self._pending_rides[-1][1])
                self._depart_idx = k0
                self.video.elev_depart_idx = k0
                self.video.elev_arrive_idx = k1
                self.video.elev_in_ride = k0 is not None and k1 is not None
            else:
                self._depart_idx = None
                self.video.elev_depart_idx = None
                self.video.elev_arrive_idx = None
                self.video.elev_in_ride = False
            self._refresh_rel_poses(n)      # the pose graph follows the rewritten heights

    def _refresh_rel_poses(self, n):
        """Rewrite the pose graph's stored relative poses at the current heights of the first
        n keyframes, so PGBA cannot re-apply the ride-internal ones a rewrite just replaced."""
        pgo = getattr(self.video, "pgobuf", None)
        if pgo is None or pgo.rel_N.value <= 0:
            return
        rn = int(pgo.rel_N.value)
        ps = lietorch.SE3(self.video.poses[:n][None])
        rel = ps[:, pgo.rel_jj[:rn]] * ps[:, pgo.rel_ii[:rn]].inv()
        pgo.rel_poses[:rn] = rel.data[0].cpu()

    def _on_arrival(self):
        """Arrival: resolve k_1, write a clean bias over the ride, arm its one re-solve
        (solve_poll), and enter the deferred-fold hold of Sec. 3.2.4 -- the rise stays in h and
        the poses stay flat -- until the ride leaves the sliding window (fold_poll) or the run
        ends (finalize_fold)."""
        # Idempotent, keyed on this ride's arrival rather than on _deferred alone: a second ride
        # arrives with k_1 None while the previous hold is still pending.
        if self._deferred and self.video.elev_arrive_idx is not None:
            return
        depart = self._depart_idx
        # k_1 = the first frame at or after the arrival stamp, since an at-most keyframe would
        # land mid-deceleration (k_0 mirrors this the other way). The detector's arrival stamp
        # can still precede t_a; `resolve_transport` books the full rise on k_1.
        _arrive_t = self._arrive_stamp()
        if _arrive_t is None:
            return
        arrive = self._kf_atleast(_arrive_t)
        self.video.elev_arrive_idx = arrive
        # No fold at the arrival: keeping elev_in_ride and h alive holds the rise in h, with
        # the poses flat, so the BA cannot wash it (Sec. 3.2.4). h folds into the poses once the
        # ride has left the sliding window.
        _ks = self.video.kf_stamps
        self.video.biass_w[depart:arrive + 1] = self.video.biass_w[depart].clone()
        self._pending_depart_stamp = round(float(_ks.get(depart, -1.0)), 4)
        self._pending_arrive_stamp = round(float(_ks.get(arrive, -1.0)), 4)
        # Recorded three ways: the stamps this class folds by, index pairs for depth_video, and
        # ride markers for pgo_buffer's same-floor gate.
        self._pending_rides.append((self._pending_depart_stamp, self._pending_arrive_stamp))
        self.video.elev_pending_rides = self.video.elev_pending_rides + [(int(depart), int(arrive))]
        self.video.elev_rides = self.video.elev_rides + [(int(depart), int(arrive))]
        self._deferred = True
        # The ride is not re-solved here: App. C.1 re-fits c on the arrival rest, and
        # certification lags t_a by ~1 s. Solving now would give a one-sided fit and a t_a
        # bounded by the detector's early arrival stamp, costing metres of rise. Arm the solve
        # instead -- solve_poll runs it once the rest span closes, the fold forces it if none
        # ever does.
        _depart_t = self._depart_stamp()
        if _depart_t is not None:
            self._unsolved = (self._pending_depart_stamp, self._pending_arrive_stamp,
                              float(_depart_t), float(_arrive_t))

    def solve_poll(self):
        """Run the arrived ride's one re-solve, once the detector certifies a rest span past its
        arrival (`rest_after`): that span is where App. C.1 re-fits c, and it bounds t_a.

        The post-arrival KFs were born carrying the provisional h[k_1], so the solve's change in
        it is added to them. No-op once solved."""
        if self._unsolved is None:
            return
        if rest_after(self._det_info(), self._unsolved[3]) is None:
            return                          # arrival rest not certified yet -- keep polling
        self._run_solve()

    def force_solve(self, depart, arrive):
        """Last call before a ride folds: solve it even though no arrival rest arrived (the
        robot walked straight out). One-sided fit of c. Only acts on the ride about to fold."""
        if self._unsolved is None:
            return
        inv = self._kf_inv_map()
        if inv.get(self._unsolved[0]) != depart or inv.get(self._unsolved[1]) != arrive:
            return
        self._run_solve()          # one-sided fit; the solver reports the ride OPEN

    def _run_solve(self):
        s0, s1, t_depart_det, t_arrive_det = self._unsolved
        self._unsolved = None
        inv = self._kf_inv_map()
        k0, k1 = inv.get(s0), inv.get(s1)
        if k0 is None or k1 is None:
            return                          # KFs gone -> the seed value stands
        with self.video.get_lock():
            d = transport.resolve_transport(self.video, k0, k1, t_depart_det, t_arrive_det,
                                            det_info=self._det_info())
            n = int(self.video.counter.value)
            if d is not None and abs(d) > 1e-9 and k1 + 1 < n:
                # post-arrival rows carry h[k_1] as their h-bar; move them with it
                self.video.elev_h[k1 + 1:n] += d

    def fold_poll(self, t1, graph):
        """The deferred fold of Sec. 3.2.4: fold each pending ride once k_1 has left the sliding
        window and no visual edge connects to it, oldest first. Called several times per
        keyframe, since the trigger (graph.ii) changes mid-update."""
        if not self._deferred or not self._pending_rides:
            return
        inv = self._kf_inv_map()
        # Fold only once the ride has left the window and no live edge touches it, so its poses
        # can no longer move; whatever is left folds in finalize_fold.
        _remaining = []
        for (s0, s1) in self._pending_rides:        # oldest first, so the folds compose in order
            k0, k1 = inv.get(s0), inv.get(s1)
            if k0 is None or k1 is None:
                _remaining.append((s0, s1))
                continue
            _aged = k1 < t1 - self.frontend_window
            # ii.min() forces a GPU sync, so only pay it when a fold is otherwise due
            _do_fold = _aged and (
                graph.ii.numel() == 0 or int(graph.ii.min()) >= k1)
            if _do_fold:
                self.force_solve(k0, k1)   # never fold a rise that was never re-solved
                self._fold_ride(k0, k1)
                with self.video.get_lock():
                    self.video.elev_pending_rides = [p for p in self.video.elev_pending_rides
                                                     if p != (k0, k1)]
                continue                            # folded -> leaves the queue
            _remaining.append((s0, s1))
        self._pending_rides = _remaining
        self._deferred = bool(self._pending_rides)

    def _fold_ride(self, depart, arrive):
        """Eq. (9): write each keyframe's own rise h_k into its world pose as a pure
        translation along e_z, then clear h and u. Called once the ride has left the sliding
        window, when its keyframes are fixed."""
        e_z = up_axis(self.video.Rwg, self.init_g)
        e_z_t = up_axis_tensor(self.video, self.video.poses.device)
        with self.video.get_lock():
            h = self.video.elev_h
            h_ride = h[depart:arrive + 1].tolist()   # one host read for the record and the fold
            rise = h_ride[-1]
            # The map keeps the elevator interior as ONE rigid submap and needs the track the
            # poses absorb here (gaussian/utils/elevator_submap.py); shipped by
            # vigs.gs_sync_rewritten_poses. Recorded BEFORE the poses move: cam_h = each ride KF's
            # camera height in the elevator frame, which places the submap for non-keyframe
            # views (ElevatorSubmaps._h_at).
            _ts = self.video.tstamp
            _C = lietorch.SE3(self.video.poses[depart:arrive + 1]).inv().matrix()[:, :3, 3]
            self.video.elev_fold_records = self.video.elev_fold_records + [dict(
                depart_ts=float(_ts[depart].item()), arrive_ts=float(_ts[arrive].item()),
                ts=_ts[depart:arrive + 1].tolist(),
                h=h_ride,
                rise=rise, e_z=[float(x) for x in e_z],
                cam_h=[float(v) for v in (_C @ e_z_t).tolist()])]
            # Fold rule of Eq. (9): each ride KF k gets its own h[k] added to the live pose,
            # which is exactly what the transport-augmented residuals saw during the ride. Its
            # premise -- an in-ride pose chain flat in the elevator frame -- is enforced by the
            # rigid-elevator clamp (depth_video.ELEV_RIGID_M), not assumed.
            for k, hk in zip(range(depart, arrive + 1), h_ride):
                if abs(hk) < 1e-9:
                    continue
                R_cw = lietorch.SE3(self.video.poses[k][None]).matrix()[0, :3, :3]
                self.video.poses[k, :3] = self.video.poses[k, :3] - hk * (R_cw @ e_z_t)
            # Post-arrival KFs were born at the departure-floor height, so lift them by the rise
            # to continue from the now-risen k_1.
            lift = 0.0
            if abs(rise) > 1e-9:
                lift = rise                        # every post-arrival KF's h == h[k_1]
                for kp in range(arrive + 1, int(self.video.counter.value)):
                    R_cw = lietorch.SE3(self.video.poses[kp][None]).matrix()[0, :3, :3]
                    self.video.poses[kp, :3] = self.video.poses[kp, :3] - lift * (R_cw @ e_z_t)
            # Clear this ride only, and subtract the materialised h-bar from every later row's
            # h: the pose lift now owns it, and a later fold would double-count it.
            self.video.elev_h[depart:arrive + 1] = 0.0
            self.video.elev_u[depart:arrive + 1] = 0.0
            if abs(rise) > 1e-9:
                self.video.elev_h[arrive + 1:int(self.video.counter.value)] -= rise
            # Only the current ride owns the ride flag: an older pending ride can leave the
            # window while a later ride is live, and must not clear it.
            if self.video.elev_arrive_idx is not None and int(self.video.elev_arrive_idx) == int(arrive):
                self.video.elev_in_ride = False
            # Record the fold; the pose graph's relative poses follow the folded heights.
            self.video.elev_folded_rides = self.video.elev_folded_rides + [(int(depart), int(arrive))]
            self._refresh_rel_poses(int(self.video.counter.value))

    def finalize_fold(self):
        """Fold any ride still held at run end -- one that never left the sliding window, e.g.
        the sequence stops within a window of k_1. No BA runs after this."""
        if not self._deferred or not self._pending_rides:
            return
        inv = self._kf_inv_map()
        for (s0, s1) in list(self._pending_rides):     # oldest first, so folds compose in order
            k0, k1 = inv.get(s0), inv.get(s1)
            if k0 is None or k1 is None:
                continue
            self.force_solve(k0, k1)       # never fold an unsolved rise
            self._fold_ride(k0, k1)
        self._pending_rides = []
        self.video.elev_pending_rides = []
        self._deferred = False
