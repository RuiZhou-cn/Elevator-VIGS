"""The Gaussian mapper: seeds Gaussians from each keyframe's depth and trains the map online.

Pose-update packets (loop closure, deferred fold) move every Gaussian with its parent keyframe;
a fold record additionally lifts the keyframes from the elevator as one rigid submap
(`gaussian.utils.elevator_submap`, not described in the paper). `color_refinement` is the final
refinement of App. C.
"""
import random
import time
import numpy as np
import torch
import threading
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3

from util.utils import Log, clone_obj
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.slam_utils import update_pose, get_loss_normal, get_loss_mapping_rgbd
from util.poses import to_se3_vec
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from gaussian.utils.elevator_submap import ElevatorSubmaps


def parent_rows(tstamps, kf_ids):
    """Packet row of every Gaussian's parent keyframe, in Gaussian order. `nonzero()` lists the
    (row, gaussian) matches by packet row, which only coincides with Gaussian order while no
    densified copy sits behind a younger keyframe's points -- indexing by it directly hands
    each Gaussian the delta of whichever parent occupies its rank in stamp order."""
    pairs = (tstamps.unsqueeze(1) == kf_ids.unsqueeze(0)).nonzero()
    assert pairs.shape[0] == kf_ids.shape[0], \
        f"pose-update packet misses the parent of {kf_ids.shape[0] - pairs.shape[0]} Gaussians"
    rows = torch.empty(kf_ids.shape[0], dtype=torch.long)
    rows[pairs[:, 1]] = pairs[:, 0]
    return rows


class GSBackEnd(mp.Process):
    def __init__(self, config, save_dir, args, use_gui=False):
        super().__init__()
        self.first_mapping = False
        self.config = config
        self.args = args
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui

        self.opt_params = munchify(config["opt_params"])
        self.config["Training"]["monocular"] = False

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        # Lock to prevent rescale() (called from tracking thread) from modifying
        # gaussian tensors while process_track_data() is mid forward/backward.
        self._gaussian_lock = threading.Lock()
        # for evaluation and visualization, should all use same background
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        # the elevator interiors as rigid submaps, placed per render stamp
        # (utils/elevator_submap.py); not described in the paper
        self.submaps = ElevatorSubmaps()

        self.cameras_extent = 6.0
        self.set_hyperparams()

        if self.use_gui:
            from gaussian.gui import gui_utils, slam_gui   # glfw / OpenGL: only with a display
            self.q_main2vis = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(3)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_gaussian_lock']  # threading.Lock is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._gaussian_lock = threading.Lock()
        

    def rescale(self, s):
        with self._gaussian_lock:
            self.gaussians.rescale(s)
            for k, v in self.viewpoints.items():
                v.T *= s
                v.T_gt *= s
                v.depth *= s
        
    def render_at(self, viewpoint):
        """render() with the elevator submaps placed for this viewpoint's stamp."""
        return render(viewpoint, self.gaussians, self.background,
                      xyz_offset=self.submaps.offsets(self.gaussians, viewpoint.tstamp))

    def offload_caches(self):
        """Drop the per-viewpoint GPU copies of image / depth / normal (~6 MB per keyframe) for
        the final BA; the CPU originals stay and every consumer falls back to a per-use .cuda()."""
        for v in self.viewpoints.values():
            v.original_image_gpu = v.depth_gpu = v.normal_gpu = None

    def restore_caches(self):
        """Re-upload what offload_caches() dropped once the BA graph is gone: one H2D pass now
        instead of one per colour-refinement iteration (26k of them)."""
        for v in self.viewpoints.values():
            v.cache_gpu()

    def save_map(self, ply_path):
        self.gaussians.save_ply(ply_path)

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.lambda_dnormal = self.config["Training"]["lambda_dnormal"]

    def update_gaussian_points(self, packet, indices, xyz):
        """Move each Gaussian with its parent keyframe's pose update: xyz taken relative to the
        parent's old camera centre, rotated by the parent's world-frame rotation change,
        rescaled, and re-attached to the new centre. Poses are stored world->camera."""
        poses_now = SE3(packet['poses'].cuda())                          # (N,) new poses
        poses_prev = packet['pose_updates'].cuda().inv() * poses_now     # now = delta * prev
        M_now, M_prev = poses_now.matrix(), poses_prev.matrix()
        R_now = M_now[..., :3, :3].transpose(-1, -2)
        R_prev = M_prev[..., :3, :3].transpose(-1, -2)
        C_now = torch.linalg.inv(M_now)[..., :3, 3][indices]            # camera centres in world
        C_prev = torch.linalg.inv(M_prev)[..., :3, 3][indices]
        R_delta = (R_now @ R_prev.transpose(-1, -2))[indices]           # prev -> now, world frame
        updates_scale = packet['scale_updates'].cuda()[indices]
        rel = (xyz - C_prev).unsqueeze(1)                                # (M,1,3)
        rot = torch.bmm(rel, R_delta.transpose(1, 2)).squeeze(1)         # (M,3)
        return C_now + rot / updates_scale
    
    def remove_all_gaussians(self):
        self.gaussians = GaussianModel(0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.initialized = False
        self.viewpoints = {}
    
    def process_track_data(self, packet):
        with self._gaussian_lock:
            return self._process_track_data_impl(packet)

    def _process_track_data_impl(self, packet):
        if not hasattr(self, "projection_matrix"):
            H, W = packet["images"].shape[-2:]
            self.K = K = list(packet["intrinsics"][0]) + [W, H]
            self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H).transpose(0, 1).cuda()

        if (packet['pose_updates'] is not None):
            with torch.no_grad():
                tstamps = packet['tstamp']
                indices = parent_rows(tstamps, self.gaussians.unique_kfIDs)
                updates_scale = packet['scale_updates'].cuda()[indices]
                updates = packet['pose_updates'].cuda()[indices]
                xyz = self.gaussians.get_xyz
                new_xyz = self.update_gaussian_points(packet, indices, xyz)
                for rec in (packet.get('elevator_folds') or []):
                    # the per-parent update lifted each ride Gaussian by its own parent's h;
                    # the elevator is one rigid body, so lift them all by the rise instead and
                    # keep the track that places the submap per render stamp
                    new_xyz = new_xyz + self.submaps.add_fold(rec, self.gaussians)

                self.gaussians._xyz[:] = new_xyz

                scale = self.gaussians.get_scaling
                scale = scale / updates_scale
                self.gaussians._scaling[:] = self.gaussians.scaling_inverse_activation(scale)
 
                rot = SO3(self.gaussians.get_rotation)
                rot = SO3(updates.data[:,3:]) * rot
                self.gaussians._rotation[:] = rot.data

        w2c = SE3(packet["poses"]).matrix().cuda()
        
        depth_packet=packet['depths']
        for i in range(len(packet['viz_idx'])):
            idx = tstamp = packet['tstamp'][i].item()     # viewpoints are keyed by frame stamp

            viewpoint = Camera.init_from_tracking(packet["images"][i]/255.0, depth_packet[i], packet["normals"][i], w2c[i], idx, self.projection_matrix, self.K, tstamp)
            if idx not in self.current_window:                    
                self.current_window = [idx] + self.current_window[:-1] if len(self.current_window) > 10 else [idx] + self.current_window
            if not self.initialized:
                self.reset()
                self.viewpoints[idx] = viewpoint
                self.add_next_kf(0, viewpoint, depth_map=depth_packet[0].numpy(), init=True)
                self.first_mapping = True
                self.initialized = True
            elif idx not in self.viewpoints:
                self.viewpoints[idx] = viewpoint
                self.add_next_kf(idx, viewpoint, depth_map=depth_packet[i].detach().clone().numpy())
            else:
                self.viewpoints[idx] = viewpoint

        if packet['pose_updates'] is not None:
            self.map(packet['tstamp'][packet['viz_idx']].tolist(), iters=20, include_global=False, max_viewpoints=12)
        else:
            if self.first_mapping:
                # max(1, ...): reset() empties current_window, so a first packet carrying a
                # single viz_idx would divide by zero before map()'s own empty-window guard.
                self.map(self.current_window, iters=self.init_itr_num//max(1, len(self.current_window)), include_global=False)
                self.first_mapping = False
            else:
                self.map(self.current_window, iters=10, include_global=True)

        if self.use_gui:
            from gaussian.gui import gui_utils
            keyframes = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    kf_window=current_window_dict,
                    gtcolor=viewpoint.original_image,
                    gtdepth=viewpoint.depth.numpy()))

    def finalize(self):
        self.color_refinement(iteration_total=self.gaussians.max_steps)

        poses_cw = []
        for view in self.viewpoints.values():
            T_w2c = np.eye(4)
            T_w2c[0:3, 0:3] = view.R.cpu().numpy()
            T_w2c[0:3, 3] = view.T.cpu().numpy()
            poses_cw.append(np.hstack(([view.tstamp], to_se3_vec(T_w2c))))
        poses_cw.sort(key=lambda x: x[0])
        return np.stack(poses_cw)

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx, iteration="after_opt"):
        eval_rendering(gtimages, gtdepthdir, traj, self.gaussians,self.save_dir, self.background,
            self.projection_matrix, self.K, kf_idx, iteration=iteration, submaps=self.submaps)
        eval_rendering_kf(self.viewpoints, self.gaussians, self.save_dir, self.background, iteration=iteration,
                          submaps=self.submaps)

    def add_next_kf(self, frame_idx, viewpoint, init=False, depth_map=None):
        if np.sum(np.abs(depth_map)) == 0:
            return

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)


    def map(self, current_window, iters, include_global=True, max_viewpoints=20):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        
        if include_global:
            random_viewpoint_stack = []
            current_window_set = set(current_window)
            for cam_idx, viewpoint in self.viewpoints.items():
                if cam_idx not in current_window_set:
                    random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            
            if include_global:
                viewpoints = viewpoint_stack + [random_viewpoint_stack[idx] for idx in torch.randperm(len(random_viewpoint_stack))[:2]]
            else:
                viewpoints = viewpoint_stack
            # during pgba, viewpoints too many, will OOM, so only use random max_viewpoints
            if len(viewpoints) > max_viewpoints:
                current_viewpoints = [viewpoints[idx] for idx in torch.randperm(len(viewpoints))[:max_viewpoints]]
            else:
                current_viewpoints = viewpoints
              
               
            for __, viewpoint in enumerate(current_viewpoints):
                try:
                    render_pkg = self.render_at(viewpoint)
                    image, viewspace_point_tensor, visibility_filter, radii, depth = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"])

                    loss_mapping += self.lambda_dnormal * get_loss_normal(depth, viewpoint) / 10.
                    loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                except Exception:   # a view the rasteriser cannot render is skipped (base system)
                    pass

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                # Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        # enforce scale not super large
        with torch.no_grad():
            current_scale = self.gaussians.get_scaling
            new_scale = current_scale.clamp(max=0.1)
            self.gaussians._scaling[:] = self.gaussians.scaling_inverse_activation(new_scale)
            
    def color_refinement(self, iteration_total):
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(view.uid)})
            if self.config["Training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        for iteration in (pbar := trange(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = self.render_at(viewpoint_cam)
            image, depth = render_pkg["render"], render_pkg["depth"]
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b

            gt_image = viewpoint_cam.original_image_gpu if viewpoint_cam.original_image_gpu is not None else viewpoint_cam.original_image.cuda()
            loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += get_loss_mapping_rgbd(self.config, image, depth, viewpoint_cam)
            if iteration < 7000:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam)
            else:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam) / 2
            loss.backward()
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            if self.use_gui and iteration % 50 == 0:
                from gaussian.gui import gui_utils
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")
