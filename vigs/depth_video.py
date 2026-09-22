"""The keyframe store and the bundle-adjustment entry points of the tracker.

`DepthVideo` holds every per-keyframe state (pose, inverse depth, velocity, bias, features) and,
for the elevator, the transport state (u_k, h_k) of Eq. (4) with the ride markers k_0 / k_1.
`inertial_ba` runs the visual-inertial solve of Sec. 3.2 -- 17-wide inside a ride, with the
update projection of Eq. (8) applied to the Gauss-Newton step -- and `cuda_pgba` the pose-graph
solve of Sec. 3.3, in which folded rides stay rigid. `init_next_pose` is the transport warm-start
of App. C.1.
"""
import os
import sys
from util.imu_utils import up_axis_tensor
_imu_cpp_build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_cpp", "build")
if _imu_cpp_build not in sys.path:
    sys.path.append(_imu_cpp_build)
from imu_integrator_cpp import IMUIntegrator

import torch
import lietorch
import vigs_backends
from lietorch import SE3, Sim3
from torch.multiprocessing import Value
from scipy.spatial.transform import Rotation

from modules.droid_net import cvx_upsample
import geom.projective_ops as pops
from geom.ba import JDSA, U_COL, H_COL, velo_retr, bias_retr, pose_retr
from pgo_buffer import global_relative_posesim3_constraints
from elevator import transport
import numpy as np


# Rigid-elevator band [m]: how far an in-ride keyframe may sit from the departure height
# (see the clamp in inertial_ba). Not described in the paper.
ELEV_RIGID_M = 0.25


class DepthVideo:
    def __init__(self, config, args, image_size, buffer):
        self.IMU_initialized = False
        # current keyframe count
        self.counter = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.is_initialized = False
        self.config = config
        self.disable_mono = config['Tracking']['disable_mono']
        self.imu_late_init_from = config['Tracking']['frontend']['imu_late_init_from']
        # state attributes
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.poses_sim3 = torch.zeros(buffer, 8, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_prior = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        # Full-resolution disparity (disps_up) and the normal prior are consumed by the Gaussian
        # map and the viewer only. A tracking run -- the default -- never reads them, so it
        # neither upsamples nor stores them: one cvx_upsample + host copy per BA update and
        # ~4 MB of host RAM per keyframe saved, trajectories unchanged.
        self.dense_outputs = bool(args.gsmapping or args.rerunvis or args.rerun_record)
        # The per-keyframe CPU image copy feeds the mapper's packets and the viewers only; a
        # tracking run has no reader for it, so it skips the device->host copy (and its sync).
        self.store_images = bool(self.dense_outputs or getattr(args, "droidvis", False))
        self.disps_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_() if self.dense_outputs else None
        self.normals = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float) if self.dense_outputs else None
        
        # feature attributes
        self.fmaps = torch.zeros(buffer, 1, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        self.poses_sim3[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float, device="cuda")

        # depth prior scale
        self.dscales = torch.ones(buffer, 2, 2, device='cuda', dtype=torch.float).share_memory_()
        
        # IMU states
        self.imus = args.imus
        self.Rwg = None
        self.init_g = args.init_g
        self.preints = {}
        self.kf_stamps = {}
        self.velos_w = torch.zeros(buffer, 3, device='cuda', dtype=torch.float)
        self.biass_w = torch.tensor(np.concatenate([args.init_bg, args.init_ba]), dtype=torch.float, device='cuda').repeat(buffer, 1)
        self.Tcb = args.Tcb
        # cam->imu and its inverse as (4, 4) matrices, fixed for the run (init_next_pose is per KF)
        self.T_cb = self.Tcb.matrix().squeeze(0).squeeze(0)
        self.T_bc = torch.linalg.inv(self.T_cb)

        # The transport state (u_k, h_k) of Eq. (4). The BA state is 17-wide
        # [pose6|vel3|bias6|u@15|h@16]; (u, h) are free only during a ride and pinned to 0
        # everywhere else.
        self.elev_in_ride = False      # between a detected departure and its arrival
        self.elev_transport_on = False  # transport state engaged at IMU init (velos_w then
                                       # holds v^E); the BA is 17-wide only while elev_in_ride
        self.elev_h = torch.zeros(buffer, device='cuda', dtype=torch.float)  # h: the rise
        self.elev_u = torch.zeros(buffer, device='cuda', dtype=torch.float)  # u: vertical velocity
        # Pose each keyframe slot last carried in a Gaussian-map packet, and whether it ever did
        # (vigs.gs_sync_rewritten_poses diffs the live poses against it to catch pose rewrites of
        # aged-out keyframes without a per-keyframe host loop)
        self.gs_sent_pose = torch.zeros(buffer, 7, device='cuda', dtype=torch.float)
        self.gs_sent_known = torch.zeros(buffer, device='cuda', dtype=torch.bool)
        self.gs_poses_rewritten = False    # a pose rewrite outside a fold record (a discarded ride)
        self.elev_depart_idx = None    # k_0 of Eq. (7): the departure constraint sits here
        self.elev_arrive_idx = None    # k_1 of Eq. (7): the arrival constraint; None while riding
        # Ride index pairs by lifecycle stage; indices are stable (KF removal only touches t1-2).
        self.elev_pending_rides = []   # arrived, not folded -> held by the weighted terms of App. C.2
        self.elev_rides = []           # every arrived ride -> skipped by the loop search
        self.elev_folded_rides = []    # folded -> rigid bridge: edge-policy flip, and the rows
                                       # are held fixed in the pose graph (Sec. 3.3)
        self.elev_fold_records = []    # per fold: the track the GS map places its rigid submap by
        self.elev_armed = False        # inside the armed window (published by motion_filter)
        self.elev_armed_events = []    # [(t, 0|1)] window transitions, for armed_at() lookups
        self.ride_hint = False         # detector armed-or-riding (published by track_frontend)
        self.armed_gate = None         # the motion filter's ArmedWindowGate; ride_manager waits on it


    def get_lock(self):
        return self.counter.get_lock()

    def elev_ride_spans(self, unfolded_only=False):
        """Arrived rides plus the live markers, deduped, as (k_0, k_1|None).

        k_1 is None while still riding. unfolded_only keeps only rides not yet folded into the
        poses, whose two ends still read the same height.
        """
        eps = [(int(a), int(b)) for a, b in self.elev_rides]
        if self.elev_depart_idx is not None:
            k1 = self.elev_arrive_idx
            eps.append((int(self.elev_depart_idx), None if k1 is None else int(k1)))
        if unfolded_only:
            folded = set((int(a), int(b)) for a, b in self.elev_folded_rides)
            eps = [e for e in eps if e[1] is None or e not in folded]
        return list(dict.fromkeys(eps))


    def rescale(self, s, t1):
        """Rescale every metric state of KFs [0, t1) -- poses, velocities, depths, PGO edges."""
        print('[INFO] Rescaling to Metric Scale:', s)
        self.poses[:t1,:3] *= s
        self.velos_w[:t1] *= s
        self.disps[:t1] /= s
        if self.dense_outputs:
            self.disps_up[:t1] /= s
        self.dscales[:t1] /= s
        # pose graph
        # Only begin adding the relative pose to PGO (add_rel_poses) after IMU init, since joint visual inertial BA will change all poses (not only scale)
        if hasattr(self, 'pgobuf') and self.pgobuf is not None:
            poses = SE3(self.poses[:t1][None])
            rel_poses = poses[:, self.pgobuf.rel_jj[: self.pgobuf.rel_N.value]] * poses[:, self.pgobuf.rel_ii[: self.pgobuf.rel_N.value]].inv()
            prev_norm = torch.linalg.norm(self.pgobuf.rel_poses[: self.pgobuf.rel_N.value][:, :3], dim=1)
            cur_norm = torch.linalg.norm(rel_poses.data[0][:, :3].cpu(), dim=1)
            rel_scale = cur_norm / prev_norm
            self.pgobuf.rel_poses[: self.pgobuf.rel_N.value] = rel_poses.data[0].cpu()
            self.pgobuf.rel_covs[:self.pgobuf.rel_N.value, :3] *= (rel_scale*rel_scale).unsqueeze(1)
            
        # gaussian, but anyway we build from beginning now
        if hasattr(self, 'gs') and self.gs is not None:
            self.gs.rescale(s)
            

    def rm_and_reintegrate(self, index):
        """Drop KF `index`: shift the per-KF states down a slot and rebuild its IMU edge."""
        self.kf_stamps[index] = self.kf_stamps[index+1]
        self.velos_w[index] = self.velos_w[index+1]
        self.biass_w[index] = self.biass_w[index+1]
        self.elev_h[index] = self.elev_h[index+1]
        self.elev_u[index] = self.elev_u[index+1]
        self.gs_sent_pose[index] = self.gs_sent_pose[index+1]
        self.gs_sent_known[index] = self.gs_sent_known[index+1]
        self.gs_sent_known[index+1] = False      # that slot is reused by the next keyframe
        if self.imus is not None:
            del self.preints[(index, index+1)]
            self.__preintegrate(index)
        
    def reintegrate_all(self):
        """Rebuild every IMU edge from the current biases (after IMU init / a bias jump)."""
        self.preints = {}
        for i in range(1, self.counter.value):
            self.__preintegrate(i)
            
    def __preintegrate(self, index):
        """Preintegrate the IMU samples spanning KFs (index-1, index) into self.preints."""
        if index < 1:
            return
        prev_stamp = self.kf_stamps[index-1]
        curr_stamp = self.kf_stamps[index]
        
        ts = self.imus[:, 0]   # sorted, strictly increasing

        i0 = max(0, np.searchsorted(ts, prev_stamp, side="right") - 1)
        i1 = min(len(ts) - 1, np.searchsorted(ts, curr_stamp, side="left"))

        measurements = self.imus[i0 : i1 + 1]

        bias = self.biass_w[index-1].cpu().numpy()
        c = self.config['IMU']
        inter = IMUIntegrator(prev_stamp, curr_stamp, bias[:3], bias[3:], self.init_g,
                              c['frequency'], c['gyroscope_noise_density'],
                              c['accelerometer_noise_density'], c['gyroscope_random_walk'],
                              c['accelerometer_random_walk'])
        inter.integrate(measurements)

        self.preints[(index-1,index)] = inter
    
    def init_next_pose(self, index, use_uncer=False):
        """Dead-reckon KF `index` from KF index-1 through their preintegration edge: pose,
        velocity and, inside a ride, the transport warm-start (u, h) of App. C.1 -- "during a
        ride the vertical part of the preintegrated motion goes into h_k and u_k rather than
        into p^E_k and v^E_k"."""
        def pose_to_SE3(pose):
            """7D [t, q] -> 4x4 T."""
            T = torch.eye(4)
            T[:3, :3] = torch.tensor(Rotation.from_quat(pose[3:].cpu().numpy()).as_matrix())
            T[:3, 3] = torch.tensor(pose[:3].cpu().numpy())
            return T

        def SE3_to_pose(T):
            """4x4 T -> 7D [t, q]."""
            q = Rotation.from_matrix(T[:3, :3].cpu().numpy()).as_quat()
            return torch.cat([torch.tensor(T[:3, 3].cpu().numpy(), dtype=T.dtype),
                              torch.tensor(q, dtype=T.dtype)])

        # Previous camera pose and velocity
        pose_prev = self.poses[index - 1]
        vel_prev = self.velos_w[index - 1]
        T_cam_prev = pose_to_SE3(pose_prev).cuda()
        # up axis e_z, needed once the transport state is live
        _ez = None
        if self.elev_depart_idx is not None:
            _ez = up_axis_tensor(self, pose_prev.device)
        # KF is born inside a ride: strip the vertical from its pose seed and carry it in h
        _ride_seed = (self.elev_in_ride and self.elev_depart_idx is not None
                      and self.elev_arrive_idx is None and index > self.elev_depart_idx)

        # IMU preintegration delta
        preint = self.preints[(index - 1, index)]
        dP = torch.tensor(preint.get_updated_dP(), dtype=pose_prev.dtype, device=pose_prev.device)
        dV = torch.tensor(preint.get_updated_dV(), dtype=pose_prev.dtype, device=pose_prev.device)
        dR_log = torch.tensor(preint.get_updated_dR_log(), dtype=pose_prev.dtype, device=pose_prev.device)
        
        trace_pos = 0.0
        if use_uncer:
            # usually only position have large uncertainty
            cov = preint.cov[:9, :9]
            cov_pos = cov[6:9, 6:9]
            trace_pos = np.trace(cov_pos)
            weight_rot = 1.0
            weight_vel = 1.0
            weight_pos = 1.0 if trace_pos < 1e-4 else 0.0

            # Downweight each component separately
            dP *= weight_pos
            dV *= weight_vel
            dR_log *= weight_rot
        
        dT = preint.dT
        g = torch.tensor(self.Rwg @ preint.g, dtype=pose_prev.dtype, device=pose_prev.device)
        # Rotation increment
        dR_mat = Rotation.from_rotvec(dR_log.cpu().numpy()).as_matrix()
        dR_mat = torch.tensor(dR_mat, dtype=pose_prev.dtype, device=pose_prev.device)

        # Tcb: body (IMU) to camera, and its inverse (both cached at construction)
        T_cb, T_bc = self.T_cb, self.T_bc

        # Compute pose of IMU at next frame
        T_imu_prev = torch.linalg.inv(T_cam_prev) @ T_cb   # imu to world
        R_imu_prev = T_imu_prev[:3, :3]
        p_imu_prev = T_imu_prev[:3, 3]

        # New imu rotation
        R_imu_next = R_imu_prev @ dR_mat

        # New imu velocity
        vel_imu_prev = vel_prev
        # velos_w holds v^E during a ride; reconstitute v^W = v^E + u*e_z of Eq. (4) here so
        # the dead-reckoned chain (and the h warm-start below) integrate the real motion.
        if self.elev_transport_on and self.elev_depart_idx is not None:
            vel_imu_prev = vel_prev + float(self.elev_u[index - 1].item()) * _ez
            # the inertial residual constrains only v^E_z + u, so v^E_z is an unobservable
            # null-space direction; project it out of the read (solve states untouched)
            if self.elev_arrive_idx is None:
                vel_imu_prev = vel_imu_prev - float(vel_prev @ _ez) * _ez
        vel_imu_next = vel_imu_prev + g * dT + R_imu_prev @ dV

        # New imu position
        if trace_pos < 1e-4:
            p_imu_next = p_imu_prev + vel_imu_prev * dT + 0.5 * g * dT * dT + R_imu_prev @ dP
        else:
            p_imu_next = p_imu_prev

        # In-ride, strip the world-vertical increment from the pose seed so the KF is born at
        # elevator height; _seed_strip is added back into the h warm-start's _dh below.
        _seed_strip = 0.0
        if _ride_seed:
            _seed_strip = float((p_imu_next - p_imu_prev) @ _ez)
            p_imu_next = p_imu_next - _seed_strip * _ez

        # Assemble T_imu_next
        T_imu_next = torch.eye(4, dtype=pose_prev.dtype, device=pose_prev.device)
        T_imu_next[:3, :3] = R_imu_next
        T_imu_next[:3, 3] = p_imu_next

        # Now transform back to camera
        T_cam_next = T_imu_next @ T_bc
        T_cam_next = torch.linalg.inv(T_cam_next)  # we store T_cw

        # Convert back to pose format
        pose_next = SE3_to_pose(T_cam_next).cuda()
        self.poses[index] = pose_next
        self.velos_w[index] = vel_imu_next

        # Transport warm-start: h[index] = h[index-1] + (p_imu_next - p_imu_prev).e_z, since
        # vision pulls the vertical back out of the poses (p^E_z is ~constant inside the
        # elevator) and h must carry it.
        if _ride_seed:
            _dh = float((p_imu_next - p_imu_prev) @ _ez) + _seed_strip
            self.elev_h[index] = self.elev_h[index - 1] + _dh
            # Split warm-start: route the elevator's vertical velocity into u, leaving v^E_z at
            # 0 (robot static inside); starting at u = 0 instead lets v^E_z absorb it.
            if self.elev_transport_on:
                _vw = self.velos_w[index]
                _u = float(_vw @ _ez)
                self.elev_u[index] = _u
                self.velos_w[index] = _vw - _u * _ez

        # Post-arrival KFs are born holding h[k_1], matching what Eq. (7) pins them to
        if (self.elev_in_ride
                and self.elev_arrive_idx is not None
                and index > self.elev_arrive_idx):
            self.elev_h[index] = self.elev_h[self.elev_arrive_idx]


    @torch.amp.autocast("cuda", enabled=False)
    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.tstamp[index] = item[0]
        
        # imu related update
        _new_kf = len(item) > 10 and item[10] is not None
        if _new_kf:
            self.kf_stamps[index] = item[10]
            if self.imus is not None:
                self.__preintegrate(index)
                # init_next_pose needs self.Rwg, set once the frontend's inertial init has run
                # (it fires when t1 reaches imu_late_init_from); the guard covers the keyframes
                # between reaching that index and the init landing, else `None @ preint.g` crashes.
                if index >= self.imu_late_init_from and self.Rwg is not None:
                    self.init_next_pose(index, use_uncer=True)
                    
        if self.store_images:
            self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_prior[index] = torch.where(depth>0, 1.0/depth, 0).cuda()

        if item[5] is not None and self.normals is not None:
            self.normals[index] = item[5]

        if item[6] is not None:
            self.intrinsics[index] = item[6]
        else:
            self.intrinsics[index] = self.intrinsics[0].clone()

        if len(item) > 7:
            self.fmaps[index] = item[7]

        if len(item) > 8:
            self.nets[index] = item[8]

        if len(item) > 9:
            self.inps[index] = item[9]
        

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj
    
    def upsample(self, ix, mask):
        """ upsample disparity (a map / viewer product: no-op for a tracking run, see dense_outputs) """
        if not self.dense_outputs:
            return
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze().cpu()

    def normalize(self, enforce_scale=None):
        """ normalize depth and poses """

        with self.get_lock():
            if enforce_scale is not None:
                s = enforce_scale
            else:    
                # this is assuming the mean depth should be 1 meter
                s = self.disps[:self.counter.value].mean().item() * self.config['Dataset']['scale_multiplier']

            self.poses[:self.counter.value,:3] *= s
            self.disps[:self.counter.value] /= s
            if self.dense_outputs:
                self.disps_up[:self.counter.value] /= s
            self.dscales[:self.counter.value] /= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj, sim3=False):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = Sim3(self.poses_sim3[None]) if sim3 else SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = vigs_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = vigs_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = vigs_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
        
    def distance_covis(self, ii=None):
        """ frame distance metric based on covisibility """
        ii = torch.as_tensor(ii)
        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        poses = self.poses[:self.counter.value].clone()
        d = vigs_backends.covis_distance(poses, self.disps, self.intrinsics[0], ii)
        d = d * (1. / self.disps[ii].median())
        return d

    def cuda_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, use_mono=False):
        """Visual-only dense BA over the keyframe window [t0, t1)."""
        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = int(torch.maximum(ii.max(), jj.max()).item()) + 1   # one device read, not two

            ht, wd = self.disps.shape[1:]
            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            dx, dz, dzcov = vigs_backends.ba(self.poses, self.disps, self.intrinsics[0], target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)

            if (not self.disable_mono) and use_mono:
                poses = lietorch.SE3(self.poses[:t1][None])
                disps = self.disps[:t1][None]
                dscales = self.dscales[:t1]
                disps, dscales, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None], self.disps_prior, dscales, ii, jj, self.mono_depth_alpha)
                self.disps[:t1] = disps[0]
                self.dscales[:t1] = dscales

            self.disps.clamp_(min=0.001)


    def inertial_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, use_mono=False):
        """Visual-inertial dense BA over [t0, t1), solving the 17-wide state in one Schur step."""
        # Global IMU-factor weight relative to vision; scales the preintegration, bias and
        # constraint blocks together (App. C.2).
        preint_scale = 1e-5
        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = int(torch.maximum(ii.max(), jj.max()).item()) + 1   # one device read, not two
            
            ht, wd = self.disps.shape[1:]
            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            iii = torch.arange(t0-1, t1-1, device='cuda')
            jjj = iii + 1
            if iii.numel() == 0:
                # Empty IMU-edge window (t1<=t0): nothing to integrate, leave state untouched.
                # Without this, assemble_factors_17w / get_preint_factors_cpp are handed an empty edge list.
                return

            # in imu/body frame
            poses_bw = self.Tcb.inv() * SE3(self.poses[:t1][None])
            velos_w = self.velos_w[:t1].unsqueeze(0)
            biass_w = self.biass_w[:t1].unsqueeze(0)
            # iii, jjj: torch tensors on GPU/CPU
            iii_cpu = iii.detach().cpu().numpy().astype(np.int64)
            jjj_cpu = jjj.detach().cpu().numpy().astype(np.int64)
            integrators = [self.preints[(i,j)] for i, j in zip(iii_cpu, jjj_cpu)]
            info2s = torch.tensor([inter.info2 for inter in integrators], dtype=torch.float, device='cuda').contiguous()
            # Rigid-elevator clamp scope: the live ride plus every arrived-but-unfolded one --
            # the same rows whose u/h the update projection holds below.
            _rigid_rides = []
            if self.Rwg is not None:
                if self.elev_depart_idx is not None:
                    _rigid_rides.append((int(self.elev_depart_idx),
                                         int(self.elev_arrive_idx) if self.elev_arrive_idx is not None
                                         else t1 - 1))
                _rigid_rides += [(int(_a), int(_b)) for (_a, _b) in self.elev_pending_rides]
                _rigid_rides = sorted(set(_rigid_rides))
            # State width of this call: 17-wide [pose6|vel3|bias6|u|h] only while a ride is live
            # or awaiting its fold (elev_in_ride: departure -> that ride's fold), when the two
            # transport columns act as consider parameters of the joint solve. Otherwise every
            # row has h = u = 0 and the base system's 15-wide assembly is the same system with
            # the pinned columns left out. Read once: the ride manager flips it between calls.
            wide = bool(self.elev_in_ride)
            for _ in range(itrs):
                # i->j,   j in camera frame, i in body frame
                Gibj = self.Tcb * poses_bw[:,jj] * poses_bw[:,ii].inv()
                Gij = Gibj * self.Tcb.inv()
                
                # ElevatorRide.step engages the transport state (elev_transport_on) once IMU
                # init has run -- the same condition that lets this call happen at all.
                assert self.elev_transport_on and self.Rwg is not None, \
                    "inertial_ba reached before IMU init: the transport state is not engaged"
                if wide:
                    Hint, vint = transport.assemble_factors_17w(
                        self, poses_bw, velos_w, biass_w, integrators, iii, jjj,
                        iii_cpu, jjj_cpu, info2s, t1, preint_scale)
                else:
                    Hint, vint = transport.assemble_factors_15w(
                        poses_bw, velos_w, biass_w, integrators, iii, jjj, iii_cpu, jjj_cpu,
                        info2s, self.Rwg, preint_scale)

                # gblind: the kernel's optional gravity-blind visual block is not used (the
                # transport state is solved per ride in elevator/transport.py); the empty (0, 3)
                # tensor skips it and only satisfies the kernel's signature
                gblind = torch.zeros((0, 3), dtype=torch.float, device='cuda')

                dx = vigs_backends.inertial_ba(poses_bw.data[0], self.disps, self.intrinsics[0], Gij.data[0], Gibj.data[0],
                                                self.Tcb.data[0,0], Hint, vint, target, weight, eta, ii, jj, t0, t1, 1, lm, ep, False, gblind)

                velos_w = velo_retr(velos_w, dx[None, :, 6:9], torch.arange(t1-t0) + t0)
                # 9:15 not 9:, so u/h (cols 15/16) never leak into the bias retraction.
                biass_w = bias_retr(biass_w, dx[None, :, 9:15], torch.arange(t1-t0) + t0)
                if wide and self.Rwg is not None:
                    # retract the transport state: u (col 15) and h (col 16); velos_w above
                    # already retracted v^E from dx[:, 6:9]
                    _du = dx[:, U_COL]     # delta u
                    _dh = dx[:, H_COL]     # delta h
                    # Weighted terms of App. C.2: every keyframe of an arrived-but-unfolded ride
                    # is held at its converged (u, h), so the free post-arrival solve cannot
                    # wash the rise back to 0.
                    _pending = self.elev_pending_rides
                    if _pending:
                        _du = _du.clone(); _dh = _dh.clone()
                        for (_k0, _k1) in _pending:
                            _lo = max(int(_k0) - t0, 0)
                            _hi = min(int(_k1) + 1 - t0, t1 - t0)
                            if _lo < _hi:
                                _du[_lo:_hi] = 0.0; _dh[_lo:_hi] = 0.0
                    # The update projection of Eq. (8): the camera does not observe the
                    # transport state, so its Gauss-Newton components carry no information and
                    # are discarded, which is the treatment of a consider parameter. The other
                    # states keep the components they received from the joint solve.
                    if self.elev_depart_idx is not None:
                        _rlo = max(int(self.elev_depart_idx) - t0, 0)
                        # While riding the range ends at the latest keyframe; once the arrival
                        # is detected it ends at k_1, which stays live for its own constraint
                        # (rows past k_1 are covered by _pending).
                        _rhi = (t1 - t0) if self.elev_arrive_idx is None else \
                               min(max(int(self.elev_arrive_idx) - t0, 0), t1 - t0)
                        if _rlo < _rhi:
                            _du = _du.clone(); _dh = _dh.clone()
                            _du[_rlo:_rhi] = 0.0; _dh[_rlo:_rhi] = 0.0
                    self.elev_u[t0:t1] = self.elev_u[t0:t1] + _du
                    self.elev_h[t0:t1] = self.elev_h[t0:t1] + _dh

                # Rigid-elevator clamp (pose side; not described in the paper): the update
                # projection above assumes the camera is blind to the elevator's vertical
                # motion, so h owns the rise alone. A glass elevator breaks that -- vision
                # tracks the world outside and lifts the in-ride pose, which then holds the
                # rise a second time. Bound each in-ride pose to a band around the departure
                # height: real motion inside (centimetres) passes untouched, a see-through
                # lift cannot accumulate.
                if _rigid_rides:
                    _ez = up_axis_tensor(self, self.elev_h.device)
                    # One SE3->matrix for the window: e_z in every body frame, and from it the
                    # world height t_wb.e_z = -(t_bw . R_bw e_z) (poses_bw is w2b)
                    _Rg = torch.einsum('nij,j->ni', poses_bw.matrix()[0, :, :3, :3], _ez)
                    _hw = -(poses_bw.data[0, :, :3] * _Rg).sum(-1)      # world height per KF
                    for (_rlo_k, _rhi_k) in _rigid_rides:
                        _lo, _hi = max(_rlo_k, t0), min(_rhi_k + 1, t1)
                        if _lo >= _hi:
                            continue
                        _rows = torch.arange(_lo, _hi, device='cuda')
                        _off = _hw[_rows] - _hw[_rlo_k]
                        # excess beyond the band, removed as a pure translation along e_z
                        _ex = _off - torch.clamp(_off, -ELEV_RIGID_M, ELEV_RIGID_M)
                        poses_bw.data[0, _rows, :3] += _ex[:, None] * _Rg[_rows]

            self.poses[:t1] = (self.Tcb * poses_bw).data[0]
            self.velos_w[:t1] = velos_w[0]
            self.biass_w[:t1] = biass_w[0]

            if (not self.disable_mono) and use_mono:
                poses = lietorch.SE3(self.poses[:t1][None])
                disps = self.disps[:t1][None]
                dscales = self.dscales[:t1]
                disps, dscales, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None], self.disps_prior, dscales, ii, jj, self.mono_depth_alpha)
                self.disps[:t1] = disps[0]
                self.dscales[:t1] = dscales
            
            self.disps.clamp_(min=0.001, max=10)


    def cuda_pgba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, se3=False):
        """Sim3 pose-graph BA over [t0, t1) with the buffered loop-closure constraints."""
        poses = Sim3(self.poses_sim3[:t1][None])
        
        # rel pose constraints (loop closure)
        rel_N = self.pgobuf.rel_N.value
        iip_lc = self.pgobuf.rel_ii[:rel_N].cuda()
        jjp_lc = self.pgobuf.rel_jj[:rel_N].cuda()
        rel_poses = self.pgobuf.rel_poses[:rel_N].cuda()[None]
        infos = 1 / self.pgobuf.rel_covs[:rel_N].cuda()
        infos = torch.cat((infos, infos.min(dim=1, keepdim=True)[0]), dim=1)
        infos = infos.unsqueeze(2).expand(*infos.size(), infos.shape[-1]) * torch.eye(infos.shape[-1], device='cuda')[None]
        infos[torch.isnan(infos) | torch.isinf(infos)] = 0.

        # The pose-graph Hessian is built in Python (unlike local BA which is fully CUDA),
        # so the iteration loop runs here rather than inside the CUDA kernel.
        for _ in range(itrs):
            Hsp, vsp, _, _, _ = global_relative_posesim3_constraints(iip_lc, jjp_lc, poses, rel_poses, infos, pw=1e-3)

            iip = iip_lc
            jjp = jjp_lc
            
            disps = self.disps[:t1][None]

            B, P, ht, wd = disps.shape
            N = ii.shape[0]
            D = poses.manifold_dim

            # 1: compute jacobians and residuals
            coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
                poses, disps, self.intrinsics[None], ii, jj, jacobian=True)

            r = (target - coords).view(B, N, -1, 1)
            w = .001 * (valid * weight).view(B, N, -1, 1)

            # 2: construct linear system
            Ji = Ji.reshape(B, N, -1, D)
            Jj = Jj.reshape(B, N, -1, D)
            wJiT = (w * Ji).transpose(2,3)
            wJjT = (w * Jj).transpose(2,3)

            Jz = Jz.reshape(B, N, ht*wd, -1)

            Hii = torch.matmul(wJiT, Ji)
            Hij = torch.matmul(wJiT, Jj)
            Hji = torch.matmul(wJjT, Ji)
            Hjj = torch.matmul(wJjT, Jj)
            Hs = torch.cat((Hii, Hij, Hji, Hjj))

            vi = torch.matmul(wJiT, r).squeeze(-1)
            vj = torch.matmul(wJjT, r).squeeze(-1)
            vs = torch.cat((vi, vj))

            Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
            Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

            w = w.view(B, N, ht*wd, -1)
            r = r.view(B, N, ht*wd, -1)
            wk = torch.sum(w*r*Jz, dim=-1)
            Ck = torch.sum(w*Jz*Jz, dim=-1)

            # disable scale if se3
            if se3:
                Hsp[:, :, :, -1, :] = 0
                Hsp[:, :, :, :, -1] = 0
                vsp[:, :, :, -1] = 0
                Hs[:, :, -1, :] = 0
                Hs[:, :, :, -1] = 0
                vs[:, :, -1] = 0
                Ei[:, :, -1, :] = 0
                Ej[:, :, -1, :] = 0
            
            dx, dz = vigs_backends.pgba(poses.data[0], self.disps, eta,
                                Hs, vs, Ei[0], Ej[0], Ck[0], wk[0],
                                Hsp, vsp, ii, jj, iip, jjp, t0, t1, lm, ep, True)

            # A folded ride is a rigid IMU-measured bridge: zero its Sim3 rows so the loop
            # error lands on the trajectory outside the elevator (Sec. 3.3).
            _folded = self.elev_folded_rides
            for (_k0, _k1) in _folded:
                _mlo = max(int(_k0) - t0, 0)
                _mhi = min(int(_k1) + 1 - t0, t1 - t0)
                if _mlo < _mhi:
                    dx[_mlo:_mhi] = 0.0

            poses = pose_retr(poses, dx[None], torch.arange(t0, t1))
    
        self.poses_sim3[:t1] = poses.data
        self.disps.clamp_(min=0.001, max=10)
