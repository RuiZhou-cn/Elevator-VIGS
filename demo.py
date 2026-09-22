"""Run Elevator-VIGS on one image sequence with its IMU file.

    python demo.py --imagedir <images/> --imufile <imu.txt> --calib <calib.txt>
                   --config config/<dataset>.yaml --output outputs/<run> [--gsmapping] [--offline]

Tracking always runs the elevator path (ride detector, transport state, deferred fold). The
run stops at the ONLINE state the paper reports and writes traj_kf_beforeBA.txt (keyframe
poses, TUM rows), config.yaml and init_dump.json; --gsmapping trains the Gaussian map alongside
and saves it as 3dgs_before_final.ply with its rendering scores under psnr/before_opt/;
--offline adds the final bundle adjustment and, with the map, the color refinement.
eval_elevator_mono.py and eval_public_mono.py build this command line per sequence.
"""
import os
import json
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'vigs'))
import yaml
import torch
import cv2
import re
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from queue import Full as QueueFull   # bound name: `queue` is shadowed inside mono_stream
from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Event
# `from vigs import VIGS` and the detector preload are imported inside main(), after the
# reader process has started: torch.multiprocessing's spawn re-executes this module's top
# level in the reader, and the system's import graph (transformers, TensorRT, the Gaussian
# mapper, ...) costs it ~3 s before it can decode the first frame; cv2 + torch is ~0.7 s.

def frame_key(name):
    """Numeric sort/stamp key for an image filename: the last number in it.

    mono_stream and get_tstamps_full MUST order the images the same way -- the keyframe
    stamps are looked up positionally, so a disagreement binds every stamp to the wrong
    frame. A lexical sort scrambles variable-digit ns timestamps (9.9s vs 10s) and
    prefixed names (frame_9 vs frame_10) alike, so both go through this one key.
    """
    return float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", name)[-1])

def get_tstamps_full(imagedir, start, length, stride, rgb_file_in_nanoseconds=True):
    tstamps_full = np.array([frame_key(x) for x in sorted(os.listdir(imagedir), key=frame_key)], dtype=np.float64)[..., np.newaxis]
    if rgb_file_in_nanoseconds:
        tstamps_full /= 1e9
    tstamps_full = tstamps_full[start:start+length][::stride]
    return tstamps_full

def mono_stream(queue, imagedir, calib, undistort=False, cropborder=0, start=0, length=100000, stride=1, rgb_file_in_nanoseconds=True, drained=None):
    """Reader process: decode, undistort, crop, resize to ~341x640 (multiple of 8) and put
    (t, timestamp, image[1,3,H,W], intrinsics[1,4], is_last) on `queue`.

    `drained` is set by the consumer once it has received the is_last frame; see the
    handshake at the end of this function for why the reader must not exit before then.
    """
    ppid = os.getppid()
    RES = 341 * 640
    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])
    # same key as get_tstamps_full -- the two orderings must agree, see frame_key
    image_list = sorted(os.listdir(imagedir), key=frame_key)[start:start+length][::stride]

    undistort_maps = None
    for t, imfile in enumerate(image_list):
        timestamp = frame_key(imfile)
        if rgb_file_in_nanoseconds:
            timestamp /= 1e9

        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        intrinsics = torch.tensor(calib[:4])

        if len(calib) > 4 and undistort:
            if undistort_maps is None:
                # cv2.undistort rebuilds these maps on every call (~7 ms/frame here)
                undistort_maps = cv2.initUndistortRectifyMap(
                    K, calib[4:], None, K, image.shape[1::-1], cv2.CV_16SC2)
            image = cv2.remap(image, *undistort_maps, cv2.INTER_LINEAR)

        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)
        is_last = (t == len(image_list)-1)
        # Bounded put: the queue is full whenever the consumer is behind, and an
        # unbounded put() would block forever if the consumer has died -- orphaning this
        # process with no way to reach the teardown below. Re-check the parent each second.
        while True:
            try:
                queue.put((t, timestamp, image[None], intrinsics[None], is_last), timeout=1.0)
                break
            except QueueFull:
                if os.getppid() != ppid:
                    return   # consumer is gone; nothing to feed and nothing to serve

    # Do not return yet. torch.multiprocessing hands tensors over as file descriptors, so
    # the consumer's rebuild_storage_fd has to reach THIS process's resource_sharer socket
    # to claim them. The queue buffers up to maxsize frames, so returning right after the
    # last put() leaves those frames unclaimable and kills the consumer with
    # "FileNotFoundError: No such file or directory".
    # The consumer sets `drained` the moment it receives the is_last frame, which -- the
    # queue being FIFO -- means every frame we sent is already mapped into its address
    # space and we are free to go. The ppid check keeps us from outliving a parent that
    # died before getting there.
    if drained is not None:
        while not drained.wait(timeout=1.0):
            if os.getppid() != ppid:
                break


def save_trajectory(vigs, traj_full, imagedir, output, start=0, length=100000, stride=1, final=False, tstamps_full=None, suffix=''):
    """Write the keyframe trajectory (TUM rows, T_wc) as traj_kf<suffix>.txt, or a numbered
    snapshot under traj/ when not final; `traj_full` (every frame) adds traj_full<suffix>.txt."""
    t = vigs.video.counter.value
    tstamps = vigs.video.tstamp[:t]
    poses_wc = lietorch.SE3(vigs.video.poses[:t]).inv().data
    if final:
        np.save(f"{output}/intrinsics.npy", vigs.video.intrinsics[0].cpu().numpy()*8)
    if tstamps_full is None:
        tstamps_full = get_tstamps_full(imagedir, start, length, stride)
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    if final:
        np.savetxt(f"{output}/traj_kf{suffix}.txt", ttraj_kf)  # for evo evaluation
    else:
        os.makedirs(f"{output}/traj", exist_ok=True)
        np.savetxt(f"{output}/traj/traj_kf{suffix}_{t:04d}.txt", ttraj_kf)  # for evo evaluation
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full{suffix}.txt", ttraj_full)


def save_imu_init(vigs, output):
    """init_dump.json: the IMU-init outputs the elevator evaluation reads back. Rwg + init_g give the
    run's own gravity-up, the axis the real heights are measured along (eval_elevator_mono.py);
    scale is the metric scale the init recovered. Absent when the run never initialised."""
    if vigs.video.Rwg is None:
        return
    json.dump({"Rwg": np.asarray(vigs.video.Rwg).tolist(),
               "init_g": [float(v) for v in vigs.video.init_g],
               "scale": float(vigs.frontend.scale)},
              open(f"{output}/init_dump.json", "w"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--imufile", type=str, help="path to imu measurement file")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=1200, help="number of keyframes to buffer")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true", help="live point-cloud display (OpenGL window)")
    parser.add_argument("--rerunvis", action="store_true", help="live visualization in the Rerun web viewer")
    parser.add_argument("--rerun_record", action="store_true", help="save rerun stream to .rrd file for later replay")
    parser.add_argument("--gsvis", action="store_true", help="live Gaussian-map display (OpenGL window); implies --gsmapping")
    parser.add_argument("--gsmapping", action="store_true", help="train the Gaussian map alongside tracking")
    parser.add_argument("--final_ba_inertial", action="store_true", help="with --offline: keep the inertial residuals in the final BA (default: visual only, as the paper)")
    parser.add_argument("--offline", action="store_true", help="run the offline post-process after tracking: final BA, and with --gsmapping the GS color refinement, scoring the refined map (psnr/after_opt/). Default: stop at the ONLINE state, which is what the paper reports -- no final BA, no refinement, the online map scored (psnr/before_opt/)")
    parser.add_argument("--start", type=int, default=0, help="start frame" )
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")
    args = parser.parse_args()
    if args.gsvis and not args.gsmapping:
        # the viewer lives inside the Gaussian mapper (GSBackEnd), so there is no map to show without it
        print("--gsvis shows the Gaussian map being trained, enabling --gsmapping")
        args.gsmapping = True

    if torch.cuda.is_available():
        print("GPU Available:", torch.cuda.get_device_name(0))

    os.makedirs(args.output, exist_ok=True)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    rgb_file_in_nanoseconds = config.get('IMU', {}).get('rgb_file_in_nanoseconds', True)
    args.imus = None
    if args.imufile is not None:
        try:
            args.imus = np.loadtxt(args.imufile, delimiter=',')
        except ValueError:                      # whitespace-separated variant
            args.imus = np.loadtxt(args.imufile, delimiter=' ')
        if args.imus.shape[1] > 7:
            # keep only [t, gyro xyz, accel xyz]; some datasets append extra GT columns
            # (e.g. q_W_I_imu) that would break the downstream m[4:] accel slice / imu_scale.
            args.imus = args.imus[:, :7]
    shutil.copy(args.config, f"{args.output}/config.yaml")
    torch.multiprocessing.set_start_method('spawn')

    vigs = None
    queue = Queue(maxsize=8)

    drained = Event()   # set below once the last frame has been received; releases the reader
    # daemon: if this process dies mid-run (an OOM, a construction error), the reader is
    # terminated at exit instead of being joined -- it would otherwise sit in put() waiting
    # for a parent that is itself waiting on it, and the dead run keeps its GPU memory.
    reader = Process(target=mono_stream, args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length, args.stride, rgb_file_in_nanoseconds, drained), daemon=True)
    reader.start()
    from vigs import VIGS
    from elevator.detect import start_preload

    # built with the config's rgb_file_in_nanoseconds and handed to every save_trajectory
    # call, so its None-fallback never re-divides already-second stamps by 1e9
    tstamps_full = get_tstamps_full(args.imagedir, args.start, args.length, args.stride, rgb_file_in_nanoseconds)

    def save_traj(traj_full, final, suffix):
        save_trajectory(vigs, traj_full, args.imagedir, args.output, start=args.start,
                        length=args.length, stride=args.stride, final=final, suffix=suffix,
                        tstamps_full=tstamps_full)

    pbar = tqdm(range(len(tstamps_full)), desc="Processing keyframes")
    # The ride detector's SigLIP2 / depth-network load (~3 s) runs on a thread; it overlaps the
    # wait for the first frame, the system's construction and the first keyframes, and only
    # the armed-window worker waits for it (elevator.detect.start_preload).
    start_preload()
    while True:
        (t, timestamp, image, intrinsics, is_last) = queue.get()
        if is_last:
            # This get() returned, so every frame the reader sent is now mapped here and
            # it can exit -- its teardown overlaps the last frame's tracking.
            drained.set()
        pbar.update()

        if vigs is None:
            args.image_size = [image.shape[2], image.shape[3]]
            vigs = VIGS(args)

        vigs.track(t, timestamp, image, intrinsics=intrinsics, is_last=is_last)

        pbar.set_description(f"Processing keyframe {vigs.video.counter.value}"
                             + (f" gs {vigs.gs.gaussians._xyz.shape[0]}" if args.gsmapping else ""))

        if t % 100 == 0:                # trajectory snapshot, kept when a run dies mid-way
            save_traj(None, final=False, suffix='')

        if is_last:
            pbar.close()
            break

    reader.join()
    if hasattr(vigs, 'mp_backend'):
        vigs.video.pgobuf.stop()
        vigs.mp_backend.join(timeout=1.0)
    save_imu_init(vigs, args.output)

    # fold any rise still held in the transport state into the world poses (Eq. (9); recovers
    # a ride that never left the sliding window); no-op otherwise. Must precede any traj save.
    vigs.frontend.elev.finalize_fold()
    if args.gsmapping:
        # ship the finalize-fold (and any other pose rewrite) to the map before it
        # is saved or scored
        vigs.gs_sync_rewritten_poses(force=True)
    if args.offline:
        # the post-process: final BA, then with the map the GS colour refinement and the
        # refined map scored (psnr/after_opt/)
        if args.gsmapping:
            vigs.gs.save_map(f'{args.output}/3dgs_before_final.ply')
        traj_full_beforeBA = vigs.traj_filler(vigs.images)
        save_traj(traj_full_beforeBA.inv().data.cpu().numpy(), final=True, suffix='_beforeBA')
        traj = vigs.terminate(inertial=args.final_ba_inertial and vigs.video.IMU_initialized)
        if args.gsmapping:
            vigs.gs.save_map(f'{args.output}/3dgs_final.ply')
        save_traj(traj, final=True, suffix='_afterBA')
    else:
        # the ONLINE state, the default and what the paper reports: no final BA, no GS colour
        # refinement. Stop the parallel GS worker first so the saved map and the scored map
        # are the same one. Scoring the online map needs poses for the non-keyframe eval
        # frames, so the map run fills the trajectory first; traj_filler leaves the keyframe
        # poses untouched, so traj_kf_beforeBA.txt is the same file with or without the map.
        vigs.gs_stop()
        if args.gsmapping:
            vigs.gs.save_map(f'{args.output}/3dgs_before_final.ply')
        traj_full = vigs.traj_filler(vigs.images) if args.gsmapping else None
        save_traj(None if traj_full is None else traj_full.inv().data.cpu().numpy(),
                  final=True, suffix='_beforeBA')
        if args.gsmapping:
            vigs.gs.eval_rendering(vigs.images, args.gtdepthdir, traj_full.matrix().data,
                                   vigs.video.tstamp[:vigs.video.counter.value].to(device='cpu'),
                                   iteration="before_opt")
    print("Finished Processing, outputs in ", args.output)

if __name__ == '__main__':
    main()