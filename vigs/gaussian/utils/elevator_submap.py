"""The elevator interior as ONE rigid submap of the Gaussian map.

Not described in the paper: a visualization detail of the released code.

In-ride keyframes are trained in the elevator frame E: the tracker holds their poses flat at
the departure height and books the rise in the transport state h (elevator/transport.py), so
the Gaussians they spawn already form a single coherent interior. The fold of Eq. (9) then
adds each keyframe's own h to its pose. Moving every Gaussian with its parent at that moment
(the plain pose-update path) would copy the interior along the shaft once per keyframe; instead
it is lifted as a body by the ride's rise -- it rests at the arrival floor in the stored map --
and a render at a time t inside the ride shifts it back to where the elevator was then,
h(t) - rise along e_z. Outside the ride it stands at the departure floor before and at the
arrival floor after (its last known position).

Membership is by parent keyframe (`GaussianModel.unique_kfIDs`, frame indices): every Gaussian
whose parent lies in [k_0, k_1] of a fold, later densified copies included (they inherit the
parent). h(t) between keyframes is linear in time on the fold's per-keyframe track; at a
keyframe it is that keyframe's own h, so a mapped view sees the interior exactly where it was
trained.

The interior is also mapped OUTSIDE the ride: the keyframes of the door wait before the
departure and after the arrival stand inside it too, and their Gaussians are a second copy of
it, parked at the departure floor (or the arrival floor) as part of the world. Left there, the
departure-floor copy slides down through the first metre of every in-ride view as the camera
rises (and the arrival-floor copy slides in at the arrival). So a Gaussian whose parent is not
a ride keyframe but which lies inside the body's oriented box -- at the departure floor for
parents before k_0, at the arrival floor for parents after k_1 -- is a "copy" (`_member_of`)
and is CULLED while its ride is in motion (moved HIDE_M along e_z, out of every frustum): from
inside a moving elevator nothing at either floor is visible, and at the departure and arrival
stamps the body coincides with the copy, so the swap is seamless. Outside the ride the copies
stay what they are, world geometry. (Moving the copies with the body instead is worse: whatever
else the box catches at a floor then travels inside the elevator -- in-ride PSNR 23.1 -> 21.8
dB.) When several rides qualify, the upcoming ride owns a departure-floor copy, the latest past
ride an arrival-floor copy.
"""

import numpy as np
import torch

MIN_BODY_POINTS = 20      # a body with fewer Gaussians gets no oriented box, hence no copies
BOX_PERCENTILE = 2.0      # the body box spans the 2..98 % of its points along each axis ...
BOX_PAD_M = 0.2           # ... plus this margin: a copy's walls ARE the body's walls, up to mapping noise
HIDE_M = 1.0e4            # a culled copy is parked this far along e_z: outside every frustum


class ElevatorSubmaps:
    def __init__(self):
        self.rides = []      # one dict per folded ride:
                             # depart_ts arrive_ts ts[] h[] rise e_z[3] cam_h[]
        self._tag = None     # cache: (key, (N,) ride index per Gaussian on cuda, -1 = world) -- BODY members
        self._mem = None     # cache: (key, (N,) offsets() table row per Gaussian on cuda, rows in use) -- body + copies

    def _key(self, gaussians):
        return (gaussians.kf_ids_version, gaussians.unique_kfIDs.shape[0], len(self.rides))

    def _ride_of(self, gaussians):
        """Ride index per Gaussian by PARENT keyframe (the body), -1 = not a ride keyframe's."""
        kf = gaussians.unique_kfIDs
        key = self._key(gaussians)
        if self._tag is None or self._tag[0] != key:
            tag = torch.full((kf.shape[0],), -1, dtype=torch.long)
            for r, ride in enumerate(self.rides):
                tag[(kf >= ride["depart_ts"]) & (kf <= ride["arrive_ts"])] = r
            self._tag = (key, tag.cuda())
        return self._tag[1]

    @staticmethod
    def _body_box(xyz, e_z):
        """Oriented box of a body's stored positions: axes = e_z and the horizontal principal
        directions of the points, extents = the BOX_PERCENTILE..100-BOX_PERCENTILE band of their
        projections. Returns (axes (3,3) rows, centre (3,), lo (3,), hi (3,)) or None for too
        few points."""
        if len(xyz) < MIN_BODY_POINTS:
            return None
        c = np.median(xyz, 0)
        horiz = (xyz - c) - np.outer((xyz - c) @ e_z, e_z)         # horizontal deviations
        _, v = np.linalg.eigh(horiz.T @ horiz)
        a0 = v[:, -1] - (v[:, -1] @ e_z) * e_z
        a0 /= np.linalg.norm(a0) + 1e-12
        axes = np.stack([a0, np.cross(e_z, a0), e_z])
        p = (xyz - c) @ axes.T
        lo, hi = np.percentile(p, BOX_PERCENTILE, 0) - BOX_PAD_M, np.percentile(p, 100 - BOX_PERCENTILE, 0) + BOX_PAD_M
        return axes, c + 0.0, lo, hi

    @staticmethod
    def _inside(xyz, box):
        axes, c, lo, hi = box
        p = (xyz - c) @ axes.T
        return np.all((p >= lo) & (p <= hi), 1)

    def _member_of(self, gaussians):
        """offsets() table row per Gaussian: 0 = world, 1 + 3r = ride r's body, 2 + 3r = its
        departure-floor copy, 3 + 3r = its arrival-floor copy."""
        key = self._key(gaussians)
        if self._mem is not None and self._mem[0] == key:
            return self._mem[1]
        self._member_rows(gaussians)
        return self._mem[1]

    def _rows_in_use(self, gaussians):
        """The distinct offsets() table rows present, so a table that is zero on every row in use
        can be reported as 'nothing to move' (None)."""
        self._member_of(gaussians)
        return self._mem[2]

    def _member_rows(self, gaussians):
        key = self._key(gaussians)
        tag = self._ride_of(gaussians).cpu()
        row = torch.where(tag >= 0, 1 + 3 * tag, torch.zeros_like(tag))
        if self.rides:
            kf = gaussians.unique_kfIDs.numpy()
            xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float64)
            tagn, rown = tag.numpy(), row.numpy()
            boxes = [self._body_box(xyz[tagn == r], np.asarray(ride["e_z"], np.float64))
                     for r, ride in enumerate(self.rides)]
            order = sorted(range(len(self.rides)), key=lambda r: self.rides[r]["depart_ts"])
            free = tagn < 0
            for r in order:                          # departure-floor copies: the upcoming ride wins
                box, ride = boxes[r], self.rides[r]
                if box is None:
                    continue
                cand = free & (kf < ride["depart_ts"])
                if cand.any():
                    e_z = np.asarray(ride["e_z"], np.float64)
                    hit = np.flatnonzero(cand)[self._inside(xyz[cand] + ride["rise"] * e_z, box)]
                    rown[hit] = 2 + 3 * r
                    free[hit] = False
            for r in reversed(order):             # arrival-floor copies: the latest past ride wins
                box, ride = boxes[r], self.rides[r]
                if box is None:
                    continue
                cand = free & (kf > ride["arrive_ts"])
                if cand.any():
                    hit = np.flatnonzero(cand)[self._inside(xyz[cand], box)]
                    rown[hit] = 3 + 3 * r
                    free[hit] = False
            row = torch.from_numpy(rown)
        self._mem = (key, row.cuda(), np.unique(row.numpy()))

    def add_fold(self, rec, gaussians):
        """Book a fold and return the (N,3) correction that turns the per-parent lift the pose
        update just applied (each ride Gaussian by its parent's own h) into the rigid one
        (every ride Gaussian by the rise): (rise - h[parent]) * e_z on ride parents, 0 else."""
        self.rides.append(rec)
        kf = gaussians.unique_kfIDs
        ts = torch.tensor(rec["ts"], dtype=torch.float64)
        h = torch.tensor(rec["h"], dtype=torch.float64)
        corr = torch.zeros(kf.shape[0], dtype=torch.float64)
        m = (kf >= rec["depart_ts"]) & (kf <= rec["arrive_ts"])
        if m.any():
            kfm = kf[m].double()
            pos = torch.searchsorted(ts, kfm).clamp_(max=len(ts) - 1)
            corr[m] = torch.where(ts[pos] == kfm, rec["rise"] - h[pos], torch.zeros_like(kfm))
        e_z = torch.tensor(rec["e_z"], dtype=torch.float32)
        return (corr.float()[:, None] * e_z[None, :]).cuda()

    @staticmethod
    def _h_at(ride, t, cam_h=None):
        """h(t) inside a ride. A keyframe reads its own h. Any other frame carries a pose the
        trajectory filler solved against the neighbouring keyframes -- inside an elevator that
        looks static, so its height tends to snap to one of them rather than follow the
        elevator in time -- but the elevator is wherever that pose sees it: given the camera's
        world height, h = cam_h - (the camera height in the elevator frame that the ride
        keyframes recorded, flat to a few mm under the rigid-elevator clamp). Without a camera
        height, or on a record without cam_h, linear in time."""
        ts = ride["ts"]
        k = int(np.searchsorted(ts, t))
        if k < len(ts) and ts[k] == t:
            return float(ride["h"][k])
        if cam_h is not None and ride.get("cam_h"):
            return float(np.clip(cam_h - np.interp(t, ts, ride["cam_h"]),
                                 min(ride["h"]), max(ride["h"])))
        return float(np.interp(t, ts, ride["h"]))

    @classmethod
    def _height(cls, ride, t, cam_h=None):
        """Elevator height at t relative to where the stored map keeps it (arrival floor)."""
        if t <= ride["depart_ts"]:
            return -ride["rise"]
        if t >= ride["arrive_ts"]:
            return 0.0
        return cls._h_at(ride, t, cam_h) - ride["rise"]

    def table(self, tstamp, cam_center=None):
        """(3R+1, 3) offset along e_z per _member_of row -- 0 the world (zero), 1 + 3r ride r's
        body, 2 + 3r / 3 + 3r its departure- / arrival-floor copy (HIDE_M while the ride is in
        motion) -- for a render at frame stamp `tstamp`, numpy, or None without a ride or stamp.
        cam_center: the view's world camera centre (3,), needed for non-keyframe stamps inside
        a ride -- see _h_at."""
        if not self.rides or tstamp is None:
            return None
        c = None
        if cam_center is not None:
            c = cam_center.detach().cpu().numpy() if torch.is_tensor(cam_center) else np.asarray(cam_center)
            c = c.reshape(3).astype(np.float64)
        table = np.zeros((3 * len(self.rides) + 1, 3), np.float32)   # row 0 = world
        for r, ride in enumerate(self.rides):
            e_z = np.asarray(ride["e_z"], np.float32)
            cam_h = None if c is None else float(c @ e_z)
            table[1 + 3 * r] = self._height(ride, float(tstamp), cam_h) * e_z        # the body
            if ride["depart_ts"] < float(tstamp) < ride["arrive_ts"]:                     # copies culled in motion
                table[2 + 3 * r] = table[3 + 3 * r] = HIDE_M * e_z
        return table

    def offsets(self, gaussians, tstamp, cam_center=None):
        """(N,3) offset along e_z per Gaussian (cuda) for a render at frame stamp `tstamp` -- the
        table row of each one's membership -- or None when nothing has to move (no ride, no
        stamp, or the stamp lies after every ride)."""
        table = self.table(tstamp, cam_center)
        if table is None or not np.any(table[self._rows_in_use(gaussians)]):
            return None
        return torch.as_tensor(table, device="cuda")[self._member_of(gaussians)]
