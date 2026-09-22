import os
import time
import torch
import numpy as np
import rerun as rr
from lietorch import SE3
import vigs_backends

# headless safety (no Qt/GUI on cluster)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("DISPLAY", None)

# helpers to draw a camera frustum as 3D line segments
_CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [ 1, -1, 1.5],
    [ 1,  1, 1.5],
    [-1,  1, 1.5],
    [-0.5, 1, 1.5],
    [ 0.5, 1, 1.5],
    [ 0, 1.2, 1.5]
], dtype=np.float32)

_CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]
], dtype=np.int32)

def _camera_lines(scale=0.05):
    pts = (_CAM_POINTS * scale).astype(np.float32)
    return [np.stack([pts[a], pts[b]], axis=0) for a, b in _CAM_LINES]  # list[(2,3)]

def _to_rr_transform(mat4):
    """Transform3D from a 4x4 world_T_cam: rerun takes the rotation and the translation
    separately, not one homogeneous matrix."""
    mat4 = mat4.astype(np.float32)
    return rr.Transform3D(translation=mat4[:3, 3], mat3x3=mat4[:3, :3])

def vigs_visualization_rerun(
    video,
    device="cuda:0",
    app_id="droid_viz",
    web_port=9876,                # HTTP port for the web viewer
    grpc_port=9877,               # gRPC port for the data stream
    enable_viewer=True,           # start live web viewer (--rerunvis)
    record_path=None,             # save .rrd file for replay (--rerun_record)
    filter_thresh=0.5,
):
    torch.cuda.set_device(device)
    rr.init(app_id)

    # Start gRPC data server + web viewer (no GUI/X11 required)
    started_server = False
    if enable_viewer:
        try:
            from urllib.parse import quote
            grpc_uri = rr.serve_grpc(grpc_port=grpc_port)
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=grpc_uri)
            started_server = True
            web_url = f"http://127.0.0.1:{web_port}?url={quote(grpc_uri, safe='')}"
            print(f"[Rerun] Open in browser: {web_url}")
        except Exception as e:
            print(f"[Rerun] Could not start web server: {e}")

    # Record to file for later playback.
    # rr.save() overwrites the gRPC sink, so only use it when the live viewer is not active.
    if record_path and not started_server:
        try:
            rr.save(record_path)
            print(f"[Rerun] Recording to {record_path}")
        except Exception as e:
            print(f"[Rerun] Recording not started: {e}")

    rr.log("world", rr.ViewCoordinates.RDF, static=True)   # OpenCV-ish axes

    filter_thresh = float(filter_thresh)
    cam_segs_local = _camera_lines(scale=0.1)  # larger for current cam

    try:
        while True:
            # Check which frames need updating
            with video.get_lock():
                dirty_index, = torch.where(video.dirty.clone())

            if len(dirty_index) == 0:
                time.sleep(0.01)
                continue

            # Mark processed
            video.dirty[dirty_index] = False

            # Fetch tensors (the 1/8-resolution disparities and images)
            poses = torch.index_select(video.poses, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()  # [N, 4, 4] world_T_cam
            images = torch.index_select(video.images, 0, dirty_index.cpu())
            intrinsics = video.intrinsics[0].clone()
            disps = torch.index_select(video.disps, 0, dirty_index)
            images = images.cpu()[..., 3::8, 3::8].permute(0, 2, 3, 1) / 255.0  # [N, H, W, 3]
            # Consistency filter
            thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            count = vigs_backends.depth_filter(video.poses, video.disps, intrinsics, dirty_index, thresh)
            count = count.cpu()
            disps = disps.cpu()
            
            masks = ((count >= 2) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True)))

            points = vigs_backends.iproj(SE3(poses).inv().data, disps.cuda(), intrinsics).cpu()  # [N,H,W,3]

            
            # Log per keyframe
            for i in range(len(dirty_index)):
                kf_idx = int(dirty_index[i].item())
                world_from_cam = Ps[i]
                cam_path = f"world/cameras/{kf_idx:06d}"

                # Pose
                rr.log(cam_path, _to_rr_transform(world_from_cam))

                # Frustum lines
                segs_world = []
                for seg in cam_segs_local:
                    seg_h = np.concatenate([seg, np.ones((2, 1), dtype=np.float32)], axis=1)  # (2,4)
                    seg_w = (world_from_cam @ seg_h.T).T[:, :3]
                    segs_world.append(seg_w)
                rr.log(f"{cam_path}/frustum", rr.LineStrips3D(np.stack(segs_world, axis=0)))  # (S,2,3)

                # Colored point cloud
                mask = masks[i].reshape(-1).numpy()
                pts = points[i].reshape(-1, 3)[mask].numpy()
                clr = images[i].reshape(-1, 3)[mask].numpy()
                if pts.size > 0:
                    rr.log(f"world/points/{kf_idx:06d}", rr.Points3D(pts, colors=(clr * 255).astype(np.uint8)))

    except KeyboardInterrupt:
        pass
