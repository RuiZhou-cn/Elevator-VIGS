"""The system: reader-side entry `track` per frame, the offline stage in `terminate`.

Wires the motion filter, the sliding-window frontend (with the ride lifecycle), the loop-closure
thread and the Gaussian mapper together. `gs_sync_rewritten_poses` carries a deferred fold
(Eq. 9), or a discarded ride, to the map as one pose-update packet.
"""
import torch
from lietorch import SE3
from modules.droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from track_frontend import TrackFrontend
from track_backend import TrackBackend
from util.trajectory_filler import PoseTrajectoryFiller
from util.utils import load_config

from collections import OrderedDict
from torch.multiprocessing import Process
from gs_backend import GSBackEnd
from pgo_buffer import PGOBuffer

import gc
import numpy as np
import os
import threading
import queue
from util.poses import to_se3_vec
from modules.trt_runner import UpdateModuleTRTRunner


def parse_extra_params(args, config):
    """Fold the config's IMU block into `args`: init states, scaled samples, Tcb, elevator block."""
    args.init_g = np.array(config['IMU']['init_g'])
    args.init_bg = np.array(config['IMU']['init_bg'])
    args.init_ba = np.array(config['IMU']['init_ba'])
    args.imu_scale = config.get('IMU', {}).get('imu_scale', 1.0)
    if config['IMU'].get('Tcb_file'):
        # per-sequence extrinsics file, relative to the image dir (FAST-LIVO2)
        args.Tcb_np = np.loadtxt(os.path.join(args.imagedir, config['IMU']['Tcb_file']))
    else:
        args.Tcb_np = np.array(config['IMU']['Tcb_np'])
    if args.imus is not None:
        if config['IMU']['imu_in_nanoseconds']:
            args.imus[:, 0] /= 1e9
        args.imus[:, 0] += config['IMU']['imu_time_offset']
        args.imus[:, -3:] *= args.imu_scale
    args.Tcb = SE3(torch.tensor(to_se3_vec(args.Tcb_np), dtype=torch.float, device='cuda')[None, None])
    # 'Elevator' block: config_defaults.py plus whatever per-sequence labels the yaml adds. The
    # elevator path is always live; nothing tells the run whether a ride is coming.
    args.elevator = config['Elevator']
    return args

class VIGS:
    def __init__(self, args):
        super(VIGS, self).__init__()
        self.load_weights(args.weights)
        self.config = config = load_config(args.config)
        args = parse_extra_params(args, config)
        self.args = args
        self.gsmapping = args.gsmapping
        # every input frame is retained here (~0.6 MB each, GBs over a long sequence) for
        # traj_filler / eval_rendering at the end -- the default online tracking-only run
        # (no map, no --offline post-process) runs neither.
        self.keep_images = bool(args.gsmapping) or bool(args.offline)
        self.images = {}

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(config, args, args.image_size, args.buffer)

        update_op = UpdateModuleTRTRunner.try_create(self.net)
        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, config, config["Tracking"]["disable_mono"],
                                    update_op=update_op)

        # frontend process
        self.frontend = TrackFrontend(self.net, self.video, config["Tracking"]["frontend"], args, update_op=update_op)

        # backend process
        self.backend = TrackBackend(self.net, self.video, config["Tracking"]["backend"])

        # 3dgs (only built for a mapping run: a tracking run never touches it)
        self.gs = GSBackEnd(config, self.args.output, args, args.gsvis) if self.gsmapping else None
        training_cfg = config["Training"]
        self._gs_parallel = training_cfg["parallel"]
        # packets whose newest keyframe is below this index are not mapped (config Training)
        self._gs_defer_until_kf = training_cfg["gs_defer_until_kf"]
        if self.gsmapping:
            self.video.gs = self.gs
            self.gs.video = self.video
            if self._gs_parallel:
                queue_size = training_cfg.get("queue_size", 2)
                self._gs_queue = queue.Queue(maxsize=queue_size)
                self._gs_thread = threading.Thread(target=self._gs_worker, daemon=True)
                self._gs_thread.start()
        # A fold rewrites already-mapped poses, so remember what was last sent to the GS map
        # (video.gs_sent_pose / gs_sent_known, per keyframe slot); this flag says whether any was.
        self._gs_any_sent = False
        self._gs_folds_shipped = 0   # entries of video.elev_fold_records already carried by a packet
        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        # visualizer
        if args.droidvis:
            from util.vigs_visualization import vigs_visualization
            self.visualizer = Process(target=vigs_visualization, args=(self.video,))
            self.visualizer.start()
    
        # rerun visualizer / recorder
        if args.rerunvis or getattr(args, 'rerun_record', False):
            from util.vigs_visualization_rerun import vigs_visualization_rerun
            self.visualizer = Process(
                target=vigs_visualization_rerun,
                args=(self.video,),
                kwargs=dict(
                    web_port=9876,                           # HTTP port for the web viewer
                    grpc_port=9877,                          # gRPC port for the data stream
                    enable_viewer=args.rerunvis,
                    record_path=(f"{self.args.output}/rerun_stream.rrd"
                                 if getattr(args, 'rerun_record', False) else None),
                )
            )
            self.visualizer.start()
        # global PGBA backend
        self.pgba = config["Tracking"]["pgba"]["active"]
        if self.pgba:
            pgba_update_op = UpdateModuleTRTRunner.try_create(self.net, pgba=True)
            self.video.pgobuf = PGOBuffer(
                self.net, self.video, self.frontend,
                config["Tracking"]["pgba"], update_op=pgba_update_op
            )

            # Use a thread-safe queue for communication *within the same process*
            self.LC_data_queue = queue.Queue()
            self.video.pgobuf.set_LC_data_queue(self.LC_data_queue)

            # Run PGBA in a background thread (same CUDA context)
            self.mp_backend = threading.Thread(
                target=self.video.pgobuf.spin,
                daemon=True,
            )
            torch.cuda.set_device(0)          
            _ = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            self.mp_backend.start()

    def load_weights(self, weights):
        """ load trained model weights """
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
    
    def _gs_drain_and_join(self):
        """Drop every pending GS packet and wait for the in-flight one: a pose correction must
        not be overwritten by a stale pre-correction packet still in the queue."""
        while not self._gs_queue.empty():
            try:
                self._gs_queue.get_nowait()
                self._gs_queue.task_done()
            except queue.Empty:
                break
        self._gs_queue.join()

    def _gs_worker(self):
        """Background thread: consumes mapping packets and runs GS optimization."""
        while True:
            data = self._gs_queue.get()
            if data is None:  # sentinel to stop
                self._gs_queue.task_done()
                break
            self.gs.process_track_data(data)
            self._gs_queue.task_done()

    def call_gs(self, viz_idx, dposes=None, dscale=None, final=False, update_idx=None, blocking=False,
                elevator_folds=None):
        """Ship a mapping packet for keyframes `viz_idx` to the GS backend.

        Queued unless blocking / serial mapping; a full queue drops its oldest packets.
        """
        if not self.gsmapping:
            return
        data = {'viz_idx':  viz_idx.to(device='cpu'),
                'tstamp':   self.video.tstamp[viz_idx].to(device='cpu'),
                'poses':    self.video.poses[viz_idx].to(device='cpu'),
                'images':   self.video.images[viz_idx.cpu()],
                'normals':  self.video.normals[viz_idx.cpu()],
                'depths':   1./self.video.disps_up[viz_idx.cpu()].to(device='cpu'),
                'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu') * 8,
                'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                'scale_updates': dscale.to(device='cpu') if dscale is not None else None,
                'update_idx': update_idx.to(device='cpu') if update_idx is not None else None}

        data['final'] = final
        # fold records for the rigid elevator submaps (gaussian/utils/elevator_submap.py),
        # only ever with dposes
        data['elevator_folds'] = list(elevator_folds) if elevator_folds else None
        if self._gs_defer_until_kf and viz_idx.max() < self._gs_defer_until_kf:
            return

        # record the pose each KF carries in this packet (gs_sync_rewritten_poses diffs against it)
        self.video.gs_sent_pose[viz_idx] = self.video.poses[viz_idx].detach()
        self.video.gs_sent_known[viz_idx] = True
        self._gs_any_sent = True
        self._gs_folds_shipped += len(elevator_folds or ())

        if blocking or not self._gs_parallel:
            self.gs.process_track_data(data)
        else:
            # Drop oldest pending packets to keep only the most recent ones
            while self._gs_queue.full():
                try:
                    self._gs_queue.get_nowait()
                    self._gs_queue.task_done()
                except queue.Empty:
                    break
            self._gs_queue.put_nowait(data)

    def gs_sync_rewritten_poses(self, force=False):
        """Re-send poses a fold (or a discarded ride) rewrote outside the local BA window to
        the GS map, as one full-map pose-update packet. Event-driven: a pending fold record, the
        release flag, or force=True at run end. It is NOT a per-tick scan for moved poses --
        that scan fired on the ordinary refinements of aged-out keyframes too (61 full-map
        packets, 62 s, on a ride-free sequence), which the base system never ships between
        PGBA packets either.
        """
        if not self.gsmapping or not self._gs_any_sent:
            return
        # a fold not yet carried to the map ships NOW -- before this tick's PGBA packet, whose
        # deltas are relative to the already-folded poses and would bury the lift
        pending = self.video.elev_fold_records[self._gs_folds_shipped:]
        if not (pending or self.video.gs_poses_rewritten or force):
            return
        with torch.no_grad():   # bookkeeping only -- call_gs below runs GS training (needs grad)
            N = int(self.video.counter.value)
            if N < 2:
                return
            last = N if force else N - 1      # mirror PGBA: skip the newest, in-flight KF
            live = self.video.poses[:last]
            known = self.video.gs_sent_known[:last]
            # a slot never shipped compares against itself: identity delta
            old = torch.where(known[:, None], self.video.gs_sent_pose[:last], live)
            if force and not pending and not self.video.gs_poses_rewritten:
                # run end with no event outstanding: ship only if anything actually moved
                moved = ((live - old).abs().amax(dim=1) > 1e-4) & known
                if not bool(moved.any()):
                    return
            dposes = SE3(live) * SE3(old).inv()
            dscale = torch.ones(last, 1)
        if self._gs_parallel:
            self._gs_drain_and_join()
        self.call_gs(torch.arange(0, last, device=self.video.poses.device),
                     dposes, dscale, blocking=True, elevator_folds=pending)
        self.video.gs_poses_rewritten = False

    def track(self, t, tstamp, image, intrinsics=None, is_last=False):
        """Main thread: run frame `t` (index) at `tstamp` (seconds) through the pipeline."""

        with torch.no_grad():
            if self.keep_images:
                self.images[t] = image

            # check there is enough motion
            self.filterx.track(t, tstamp, image, intrinsics)

            # local bundle adjustment
            viz_idx = self.frontend(is_last=is_last)

        if self.gsmapping:
            # before the PGBA block, so a same-tick PGBA delta stays consistent
            self.gs_sync_rewritten_poses()

        if len(viz_idx) and self.pgba:
            dposes, dscale, lcii, lcjj, local_ii, local_jj = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                update_idx = torch.unique(torch.cat([lcii, lcjj]))
                if self._gs_parallel:
                    self._gs_drain_and_join()
                self.call_gs(torch.arange(0, self.video.counter.value-1, device='cuda'), dposes[:-1], dscale[:-1], update_idx=update_idx, blocking=True)

        if len(viz_idx):
            self.call_gs(viz_idx)

    def gs_stop(self):
        """Stop the parallel GS worker once every queued packet is mapped, so the map that is
        saved or scored next is complete. No-op without the map or the worker."""
        if self.gsmapping and self._gs_parallel:
            self._gs_queue.join()
            self._gs_queue.put(None)  # stop the worker thread
            self._gs_thread.join()

    def terminate(self, inertial=False):
        """Run the final BA, refine and fill the trajectory; returns the full poses [t, q]."""
        # Release the ONLINE state the offline stage never touches, so the global BA's factor
        # graph (~1.4 MB per edge, up to 1e4 edges) fits next to the map: the PGBA buffer (its
        # TRT update runner and loop-closure store; it also references the frontend, so
        # `del self.frontend` alone freed nothing), the frontend's local graph with its 48 full
        # correlation volumes, the motion filter's encoders + Omnidata engines, and the mapper's
        # per-viewpoint GPU caches (the CPU originals stay; every consumer falls back to a per-use
        # .cuda()). These objects sit in reference cycles (frontend <-> graph <-> video), so the
        # cyclic GC has to run before the caching allocator can hand the blocks back.
        # Result-identical; without it a 978-keyframe run with the Gaussian map ran out of
        # memory in the first bundle-adjustment pass.
        a0 = torch.cuda.memory_allocated()
        if self.pgba:
            self.video.pgobuf = None
        del self.frontend
        del self.filterx
        self.gs_stop()                         # drain the parallel mapper before the BA, not after
        if self.gsmapping:
            self.gs.offload_caches()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[terminate] online state released: allocated {a0 / 2**30:.1f} -> {torch.cuda.memory_allocated() / 2**30:.1f} GB, "
              f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB, free {free / 2**30:.1f} of {total / 2**30:.1f} GB", flush=True)

        poses_pre = self.video.poses[:self.video.counter.value].clone()
        self.backend(7, inertial=inertial)
        self.backend(12, inertial=inertial)
        del self.backend
        poses_pos = self.video.poses[:self.video.counter.value].clone()
        dposes = SE3(poses_pos) * SE3(poses_pre).inv()
        dscale = torch.ones(self.video.counter.value, 1)
        torch.cuda.empty_cache()

        # Final Color Refinement
        if self.gsmapping:
            self.gs.restore_caches()
            # Pose refinement by re-rendering loss.
            self.call_gs(torch.arange(0, self.video.counter.value, device='cuda'), dposes, dscale, final=True, blocking=True)
            updated_poses = self.gs.finalize()
            self.video.poses[:self.video.counter.value] = torch.tensor(updated_poses[:,1:])
                    
        traj_full = self.traj_filler(self.images)
        if self.gsmapping:
            self.gs.eval_rendering(self.images, self.args.gtdepthdir, traj_full.matrix().data, self.video.tstamp[:self.video.counter.value].to(device='cpu'))
        return traj_full.inv().data.cpu().numpy()
