"""Loop closure across floors (Sec. 3.3): candidate search, relative-pose store and PGBA.

`search_lc_candidate` proposes keyframe pairs by induced flow and applies the three rules of
Sec. 3.3 -- no pair straddles a ride before its fold, the same-floor gate of `SAME_FLOOR_GATE_M`,
and no pair touches a keyframe from the elevator. Each detected loop runs the pose-graph bundle
adjustment (`_pgba`), whose result the tracker ships to the Gaussian map.
"""
from util.imu_utils import up_axis_tensor
import torch
import time
import numpy as np
import geom.projective_ops as pops

from torch.multiprocessing import Value
from queue import Empty
from scipy.spatial.transform import Rotation as R
from lietorch import SE3, Sim3
from factor_graph import FactorGraph
from util.utils import Log


# The same-floor gate of Sec. 3.3: a candidate pair whose heights along gravity, p^E_z + h,
# differ by more than about half a floor cannot be the same place (a building property, not a
# rig property).
SAME_FLOOR_GATE_M = 1.5

eps = 1e-8
def diff(x1, x2):
    return (x1 - x2) / 2. / eps


def num_jacobi(func, Gi, Gj=None, first=True):
    batch = Gi.shape[0]
    D = Gi.manifold_dim
    J_num = []
    for i in range(D):
        delta = torch.zeros(D, device='cuda').expand(batch, 1, D).type(torch.float64)
        delta[:, :, i] = eps
        if first:
            if isinstance(Gi, Sim3):
                J_num.append(diff(func(Sim3.exp(delta)*Gi, Gj), func(Sim3.exp(-delta)*Gi, Gj)))
            else:
                J_num.append(diff(func(SE3.exp(delta)*Gi, Gj), func(SE3.exp(-delta)*Gi, Gj)))
        else:
            if isinstance(Gi, Sim3):
                J_num.append(diff(func(Gi, Sim3.exp(delta)*Gj), func(Gi, Sim3.exp(-delta)*Gj)))
            else:
                J_num.append(diff(func(Gi, SE3.exp(delta)*Gj), func(Gi, SE3.exp(-delta)*Gj)))
    return torch.stack(J_num, dim=-1)


def global_relative_posesim3_constraints(ii, jj, poses, rel_poses, infos, pw=1e-5):
    Gii = Sim3((poses[:, ii].data.type(torch.float64)))
    Gjj = Sim3((poses[:, jj].data.type(torch.float64)))
    Gij = Sim3(SE3(rel_poses.data.type(torch.float64)))

    def func(Gii, Gjj):
        e = Gij * Gii * Gjj.inv()
        return e.log()

    # numerical jacobi
    Ji = num_jacobi(func, Gii, Gjj, first=True).type(torch.float32)
    Jj = num_jacobi(func, Gii, Gjj, first=False).type(torch.float32)
    # the 7 dimension corresdponds to translation, rotation, and scale
    r = func(Gii, Gjj).unsqueeze(-1).type(torch.float32)
    chi2 = torch.sum(r.transpose(2, 3) @ r)
    chi2_scaled = torch.sum(r.transpose(2, 3) @ infos @ r)

    wJiT = ((pw * Ji.double()).transpose(2, 3) @ infos.double()).float()
    wJjT = ((pw * Jj.double()).transpose(2, 3) @ infos.double()).float()
    Hsp = torch.stack([torch.matmul(wJiT, Ji), torch.matmul(wJiT, Jj), torch.matmul(wJjT, Ji), torch.matmul(wJjT, Jj)])    # 4x1xNx7x7
    vsp = -torch.stack([torch.matmul(wJiT, r), torch.matmul(wJjT, r)]).squeeze(-1)  # 2x1xNx7
    return Hsp, vsp, chi2, chi2_scaled, r

class PGOBuffer:
    def __init__(self, net, video, frontend, config, update_op=None):
        if update_op is not None:
            self.update_op = update_op
        else:
            self.update_op = net.update
        self.video = video
        self.frontend = frontend
        max_rel = 2e5
        self.pgba_thresh = config["pgba_thresh"]

        self.rel_N = Value('i', 0)
        self.rel_ii = torch.zeros(int(max_rel), dtype=torch.long, device='cpu').share_memory_()
        self.rel_jj = torch.zeros(int(max_rel), dtype=torch.long, device='cpu').share_memory_()
        self.rel_poses = torch.zeros(int(max_rel), 7, device="cpu", dtype=torch.float).share_memory_()
        self.rel_covs = torch.zeros(int(max_rel), 6, device="cpu", dtype=torch.float).share_memory_()
        self.rel_valid_percent = torch.zeros(int(max_rel), device="cpu", dtype=torch.float).share_memory_()

        self.kfs = set()
        self.lcii = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.lcjj = torch.as_tensor([], dtype=torch.long, device='cuda')
        
        self._stop = False

    def stop(self):
        self._stop = True

    @torch.amp.autocast("cuda", enabled=False)
    def add_rel_poses(self, ii, jj, target, weight):
        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        masks = valid > 1e-2
        ii, jj, target, weight = ii[masks], jj[masks], target[:, masks], weight[:, masks]
        
        N = ii.shape[0]
        if N < 1:
            return

        t1 = max(ii.max(), jj.max())+1
        poses = SE3(self.video.poses[:t1][None])
        rel_poses = poses[:, jj] * poses[:, ii].inv()

        # One linearisation. HI-SLAM2 iterates this four times with a pose update in between;
        # here the update is not applied (with the inertial loss in play, an update built from
        # reprojection alone would fight it), so further passes would recompute the same values.
        coords, valid, (_, Jj, _) = pops.projective_transform(
            poses, self.video.disps[None], self.video.intrinsics[None], ii, jj, jacobian=True)
        r = (target - coords).view(1, N, -1, 1)
        w = .001 * (valid * weight).view(1, N, -1, 1)
        Jj = Jj.reshape(1, N, -1, 6)
        wJjT = (w * Jj).transpose(2, 3)
        Hjj = torch.matmul(wJjT, Jj) + 1e-4*torch.eye(6, device='cuda')[None, None]
        vj = torch.matmul(wJjT, r)

        Hinv = torch.linalg.inv(Hjj)
        dx = Hinv @ vj

        V = Jj @ dx - r
        sig2 = (w * V).transpose(2, 3) @ V
        cov = sig2 * Hinv
        cov = torch.diagonal(cov, dim1=2, dim2=3)

        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        self.rel_ii[self.rel_N.value:self.rel_N.value+N] = ii
        self.rel_jj[self.rel_N.value:self.rel_N.value+N] = jj
        self.rel_poses[self.rel_N.value:self.rel_N.value+N] = rel_poses.data[0]
        self.rel_covs[self.rel_N.value:self.rel_N.value+N] = cov[0]
        self.rel_valid_percent[self.rel_N.value:self.rel_N.value+N] = valid
        self.rel_N.value += N

    def _pgba(self, LC_data):
        lcii, lcjj = LC_data['lcii'], LC_data['lcjj']
        # new graph for loop closure
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=-1)
        ii, jj = torch.cat((lcii, lcjj, self.frontend.graph.ii)), torch.cat((lcjj, lcii, self.frontend.graph.jj))
        graph.add_factors(ii, jj)

        t0 = max(8, min(ii.min().item(), jj.min().item())+1)
        
        kx = torch.unique(ii)
        m = torch.isin(self.frontend.graph.ii_inac, kx)
        graph.ii_inac = self.frontend.graph.ii_inac[m]
        graph.jj_inac = self.frontend.graph.jj_inac[m]
        graph.target_inac = self.frontend.graph.target_inac[:,m]
        graph.weight_inac = self.frontend.graph.weight_inac[:,m]

        with torch.no_grad():
            with self.video.get_lock():
                t1 = self.video.counter.value
                self.video.poses_sim3[:t1, 7] = 1
                self.video.poses_sim3[:t1, :7] = self.video.poses[:t1]
                Log(f"run with {graph.ii.shape[0]} factors from keyframe {t0} to {t1}", tag="PGBA")

                graph.update_pgba(t0=t0, t1=t1)

                for _ in range(6):
                    self.frontend.graph.update(None, None, use_inactive=True)

                self.video.dirty[:self.video.counter.value] = True
        
        self.add_rel_poses(ii[:2*len(lcii)], jj[:2*len(lcii)], graph.target[:,:2*len(lcii)], graph.weight[:,:2*len(lcii)])
        del graph
        return lcii, lcjj, self.frontend.graph.ii, self.frontend.graph.jj

    def run_pgba(self, LC_data_queue):
        try:
            LC_data = LC_data_queue.get_nowait()   # a timeout here is 1 ms idle per keyframe
        except Empty:
            LC_data = None
        if LC_data:
            poses_pre = self.video.poses[:self.video.counter.value].clone()
            lcii, lcjj, local_ii, local_jj = self._pgba(LC_data)
            poses_pos = self.video.poses[:self.video.counter.value].clone()
            dposes = SE3(poses_pos) * SE3(poses_pre).inv()
            dscale = self.video.poses_sim3[:self.video.counter.value, -1:]
            return dposes, dscale, lcii, lcjj, local_ii, local_jj
        else:
            return None, None, None, None, None, None

    def set_LC_data_queue(self, queue):
        self.LC_data_queue = queue

    def reset(self):
        self.lcii = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.lcjj = torch.as_tensor([], dtype=torch.long, device='cuda')

    def _kf_up_heights(self, idx, e_z_t):
        """The height along gravity of Sec. 3.3, p^E_z + h, per keyframe.

        Call under video.get_lock()."""
        M = SE3(self.video.poses[idx]).inv().matrix()
        return M[:, :3, 3] @ e_z_t + self.video.elev_h[idx]

    def search_lc_candidate(self, hist, kx, dev):
        """Propose loop-closure pairs (old KF in [0, hist) <-> query KF kx) into lcii/lcjj.

        Filtered in order by flow distance, the three rules of Sec. 3.3, and relative
        orientation."""
        ii = torch.arange(0, hist, device=dev)
        jj = torch.full_like(ii, kx)
        with self.video.get_lock():
            dd = self.video.distance(ii, jj)
        keep = dd < self.pgba_thresh
        if not bool(keep.any()):
            return
        ii, jj = ii[keep], jj[keep]

        def cut(drop):
            """Drop the masked candidates; False once nothing is left."""
            nonlocal ii, jj
            if int(drop.sum()):
                ii, jj = ii[~drop], jj[~drop]
            return ii.shape[0] > 0

        rides = self.video.elev_ride_spans()
        if rides:
            # A keyframe from the elevator observes only its inside, which looks the same at
            # every height, so never pair one (Sec. 3.3, third rule).
            inride = torch.zeros_like(ii, dtype=torch.bool)
            for k0, k1 in rides:
                hi = int(1e9) if k1 is None else k1
                inride |= ((ii >= k0) & (ii <= hi)) | ((jj >= k0) & (jj <= hi))
            if not cut(inride):
                return

            # Before the fold the rise is still held in h, so the poses after the ride sit at
            # the departure height and a pair straddling the ride induces a small flow: reject
            # it until the fold (Sec. 3.3, first rule).
            span = torch.zeros_like(ii, dtype=torch.bool)
            for k0, k1 in self.video.elev_ride_spans(unfolded_only=True):
                if k1 is not None:
                    span |= (ii < k0) & (jj > k1)
            if not cut(span):
                return

            # The same-floor gate (Sec. 3.3, second rule): a height difference over
            # SAME_FLOOR_GATE_M means a different floor. Needs Rwg, so it is inert before IMU
            # init, and it only runs once a ride has been detected (App. C).
            if self.video.Rwg is not None and self.video.init_g is not None:
                e_z_t = up_axis_tensor(self.video, dev)
                with self.video.get_lock():
                    h = self._kf_up_heights(torch.cat([ii, jj]), e_z_t)   # one SE3 inverse for both ends
                dh = (h[:ii.shape[0]] - h[ii.shape[0]:]).abs()
                if not cut(dh >= SAME_FLOOR_GATE_M):
                    return

        # Reject pairs whose views are more than ~120 deg apart: a translation-weighted flow
        # distance stays small for co-located frames looking opposite ways.
        with self.video.get_lock():
            Gij = (SE3(self.video.poses[jj]) * SE3(self.video.poses[ii]).inv()).data
        euls = R.from_quat(Gij[:, 3:].cpu().numpy()).as_euler('zxy', degrees=True)
        ok = torch.as_tensor(np.linalg.norm(euls, axis=1) < 120, device=dev)
        if bool(ok.any()):
            self.lcii = torch.cat([self.lcii, ii[ok][:10]])
            self.lcjj = torch.cat([self.lcjj, jj[ok][:10]])

    @torch.no_grad()
    def spin(self):
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev.index)
        torch.cuda.synchronize()
        while not self._stop:
            with self.video.get_lock():
                video_counter = self.video.counter.value
            kx = video_counter - 4
            if kx < 60 or kx in self.kfs:
                time.sleep(0.1)
                continue

            self.search_lc_candidate(video_counter - 55, kx, dev=dev)
            self.kfs.add(kx)

            wait_long = len(self.lcjj) > 0 and (kx - self.lcjj[0]) > 3
            if self.lcii.shape[0] > 24 or wait_long:
                self.LC_data_queue.put({'lcii': self.lcii, 'lcjj': self.lcjj})
                self.reset()
                self.kfs.update({kx+1, kx+2})