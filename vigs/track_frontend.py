"""The sliding-window tracker: keyframe graph, local visual-inertial BA and IMU initialization.

Per keyframe, `TrackFrontend` steps the ride lifecycle (`elevator.ride_manager.ElevatorRide`:
departure, arrival, deferred fold of Sec. 3.2.4) before the bundle adjustment, and lets it veto
the removal of keyframes from the elevator (App. C.4).
"""
import torch
import lietorch
import numpy as np
from factor_graph import FactorGraph
from geom.ba import InitializeGravityDirectionDynamic, InitializeVeloBiasGdir, InitializeFullInertialBA, BA_prepare
from elevator.ride_manager import ElevatorRide
from collections import defaultdict

class TrackFrontend:
    def __init__(self, net, video, config, args, update_op=None):
        self.video = video

        if update_op is not None:
            self.update_op = update_op
        else:
            self.update_op = net.update
        
        self.graph = FactorGraph(video, self.update_op, max_factors=48)
        self.args = args
        # local optimization window
        self.t1 = 0

        self.delete_count = defaultdict(int)
        
        # frontent variables
        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2
        self.warmup = 10

        self.frontend_nms = config["frontend_nms"]
        self.keyframe_thresh = config["keyframe_thresh"]
        self.keep_kf_once_deleted = config.get("keep_kf_once_deleted", False)
        self.frontend_window = config["frontend_window"]
        self.frontend_thresh = config["frontend_thresh"]
        self.frontend_radius = config["frontend_radius"]
        self.video.mono_depth_alpha = config["mono_depth_alpha"]
        
        self.Tcb = args.Tcb
        self.init_bg = args.init_bg
        self.init_ba = args.init_ba
        self.init_g = args.init_g
        
        self.scale = 1
        self.imu_init_fix_scale = False  # should be False since IMU can provide metric scale
        self.imu_late_init_from = config["imu_late_init_from"]

        # Ride lifecycle (detector, departure/arrival/fold) -- always live, see elevator/.
        self.elev = ElevatorRide(video, args.elevator, self.frontend_window)

    def __graph_update(self, iters, t0=None, t1=None, use_inactive=False, use_mono=False, mode='inertial', disable_vision=False):
        """mode: 'vision_only' | 'inertial' (+ '_tracking' = newest KF only), or the one-shot
        'initial_inertial' IMU initialisation (gravity, velocity/bias, joint BA, metric scale)."""
        if mode != "initial_inertial":
            for i in range(iters):
                self.graph.update(t0, t1, use_inactive=use_inactive, use_mono=use_mono, inertial=('inertial' in mode), tracking=('tracking' in mode), disable_vision=disable_vision)
        else:
            t0=0
            poses_cw = lietorch.SE3(self.video.poses[:self.t1][None])
            poses_bw = self.Tcb.inv() * poses_cw
            velos_w = self.video.velos_w[:self.t1].unsqueeze(0)
            biass_w = self.video.biass_w[:self.t1].unsqueeze(0)
            Rwg = self.video.Rwg
            fix_front = 0   # fix the velocity of the very first frame as zero vector

            # step 1
            if Rwg is None:
                Rwg = np.eye(3)
                for itr in range(iters):
                    Rwg = InitializeGravityDirectionDynamic(t0, self.t1, poses_bw, velos_w, biass_w, self.video.preints, Rwg)

            # step 2
            scale = 1
            # Step scale on the bias increment of the two init solves (upstream VIGS knob);
            # 1 = estimate bias during initialization.
            init_bias_scale = 1
            for _ in range(iters):
                velos_w, biass_w, Rwg, scale = InitializeVeloBiasGdir(t0, self.t1, poses_bw, velos_w, biass_w, self.video.preints, Rwg, scale, self.imu_init_fix_scale, fix_front=fix_front, bias_scale=init_bias_scale)
            self.video.velos_w[:self.t1] = velos_w[0]
            self.video.biass_w[:self.t1] = biass_w[0]
            self.video.reintegrate_all()

            # step 3
            disps = self.video.disps[:self.t1][None]
            intrs = self.video.intrinsics[:self.t1][None]

            newgraph = FactorGraph(self.video, self.graph.update_op, corr_impl="alt", max_factors=1000)
            newgraph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
            
            for itr in range(iters):
                t0, target, weight, eta, ii, jj, opt_ii, upmask = newgraph.get_network_update_full_graph(t0=t0)
                
                for _ in range(2):
                    H, E, C, v, w = BA_prepare(target, weight, eta, poses_bw, disps, intrs[:,:,:], ii, jj, self.video.Tcb, fixedp=0, D=15, t0=t0)
                    poses_bw, velos_w, biass_w, disps, scale = InitializeFullInertialBA(t0, self.t1, poses_bw, velos_w, biass_w, disps,
                                                                                          ii, self.video.preints, Rwg, scale, H, E, C, v, w, imu_init_fix_scale=self.imu_init_fix_scale, fix_front=15, bias_scale=init_bias_scale)
                    poses_cw = self.Tcb * poses_bw
                # must update each iter
                self.video.poses[:self.t1] = poses_cw.data[0]
                self.video.disps[:self.t1] = disps[0]
            
            self.video.upsample(torch.unique(opt_ii), upmask) # upsample the disp, using upmask
            self.video.velos_w[:self.t1] = velos_w[0]
            self.video.biass_w[:self.t1] = biass_w[0]

            self.video.reintegrate_all()
            self.video.Rwg = Rwg
            if not self.imu_init_fix_scale:
                self.scale = scale
                self.video.rescale(scale, self.t1)

            self.video.dirty[:self.t1] = True

            self.video.IMU_initialized = True

    def __update(self, is_last):
        """ add edges, perform update """

        self.t1 += 1

        # Step the detector before the BA, so k_0 is already in-ride and the IMU factors cannot
        # misread the elevator's acceleration as the robot's. The BA stays visual-inertial
        # through the ride, with per-KF (v^E, u, h) in the 17-wide joint solve of Sec. 3.2.1.
        elev_in = self.elev.observe_kf(self.t1, is_last)
        if self.video.IMU_initialized:
            self.elev.step(elev_in, self.t1)

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, remove=True)

        self.video.dscales[self.t1-1] = self.video.disps[self.t1-1].median() / self.video.disps_prior[self.t1-1].median()

        for itr in range(self.iters1):
            if self.t1 > self.imu_late_init_from:
                self.elev.fold_poll(self.t1, self.graph)
                self.__graph_update(1, None, None, use_inactive=True, use_mono=itr>1, mode='inertial', disable_vision=False)
            else:
                # vision_only_tracking just to save computational cost, also fine to use vision_only
                self.__graph_update(1, None, None, use_inactive=True, use_mono=itr>1, mode='vision_only_tracking')
                self.__graph_update(1, None, None, use_inactive=True, use_mono=itr>1, mode='vision_only')

        self.elev.solve_poll()      # the ride's one re-solve, once its arrival rest closes

        d = self.video.distance([self.t1-3], [self.t1-2], bidirectional=True)
        
        already_skip = (self.keep_kf_once_deleted and self.delete_count[self.t1-2] > 0)
        dT = (self.video.kf_stamps[self.video.counter.value-1] - self.video.kf_stamps[self.video.counter.value-3])
        d_covis = self.video.distance_covis([self.t1-2])
        covis_thresh = 0.1
        criteria =  d.item() < self.keyframe_thresh and d_covis.item() < covis_thresh and (dT < 3) and not already_skip
        # Never cull inside a ride: dense ride keyframes, k_0 and k_1, a ride awaiting its fold.
        if criteria and self.elev.keep_kf(self.t1 - 2, elev_in):
            criteria = False


        if criteria:
            self.graph.rm_keyframe(self.t1 - 2)
            self.delete_count[self.t1-2] += 1
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
            update_idx = []
        else:
            if self.t1 > self.imu_late_init_from:
                for itr in range(self.iters2):
                    self.elev.fold_poll(self.t1, self.graph)
                    self.__graph_update(1, None, None, use_inactive=True, mode="inertial", disable_vision=False)
            else:
                for itr in range(self.iters2):
                    self.__graph_update(1, None, None, use_inactive=True, mode="vision_only")

            if self.t1 == self.imu_late_init_from:
                self.__graph_update(5, None, None, use_inactive=True, mode="vision_only")
                print()
                print("[INFO] Begin IMU Initialization...")
                self.__graph_update(8, t0=1, t1=None, use_inactive=True, mode="initial_inertial")
                

            # Exclude the newest keyframe from mapping indices -- its pose/depth are still being optimized
            # and may be unstable. Exception: if this is the last keyframe, include it anyway.
            if is_last:
                update_idx = torch.arange(self.graph.ii.min(), self.t1, device='cuda')
            else:
                update_idx = torch.arange(self.graph.ii.min(), self.t1-1, device='cuda')

        # set pose/disps/velos_w/biass_w for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()
        self.video.velos_w[self.t1] = self.video.velos_w[self.t1-1]
        self.video.biass_w[self.t1] = self.video.biass_w[self.t1-1]

        # (a new ride KF's h is warm-started in DepthVideo.init_next_pose, which has the edge)

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        
        # after IMU initialization, update everything
        if self.t1 == self.imu_late_init_from:
            update_idx = torch.arange(0, self.t1-1, device='cuda')
            self.video.dirty[0:self.t1] = True
            if hasattr(self.video, 'gs') and self.video.gs is not None:
                self.video.gs.remove_all_gaussians()

        return update_idx
    
    def __initialize(self):
        """ initialize the SLAM system """

        self.t1 = self.video.counter.value

        # initial optimization
        self.graph.add_neighborhood_factors(0, self.t1, r=3)
        for itr in range(8):
            self.__graph_update(1, 1, use_inactive=True, use_mono=False, mode="vision_only")
            
        # refine optimization
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for i in range(self.t1):
            self.video.dscales[i] = self.video.disps[i].median() / self.video.disps_prior[i].median()
        for itr in range(8):
            self.__graph_update(1, 1, use_inactive=True, use_mono=itr>2, mode="vision_only")

        
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for itr in range(8):
            self.__graph_update(1, 1, use_inactive=True, use_mono=itr>2, mode="vision_only")

        # useful for e.g. fastlivo
        self.video.normalize()

        # initialization complete
        self.video.is_initialized = True
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()
        self.video.velos_w[self.t1] = self.video.velos_w[self.t1-1]
        self.video.biass_w[self.t1] = self.video.biass_w[self.t1-1]
        
        with self.video.get_lock():
            self.video.dirty[:self.t1] = True
        self.graph.rm_factors(self.graph.ii < self.t1-4, store=True)
        return torch.arange(self.t1-1, device='cuda')

    def __call__(self, is_last):
        """ main update """
        self.to_update = []

        # do initialization
        if not self.video.is_initialized and self.video.counter.value == self.warmup:
            self.to_update = self.__initialize()
            
        # do update
        elif self.video.is_initialized and self.t1 < self.video.counter.value:
            self.to_update = self.__update(is_last)

        return self.to_update
