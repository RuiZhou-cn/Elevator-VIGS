import torch
import vigs_backends
import numpy as np
import open3d as o3d

from lietorch import SE3

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def vigs_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    vigs_visualization.video = video
    vigs_visualization.cameras = {}
    vigs_visualization.points = {}
    vigs_visualization.warmup = 8
    vigs_visualization.scale = 1.0
    vigs_visualization.cam_scale = 1.0
    vigs_visualization.cam_line_width = 1.0
    vigs_visualization.ix = 0
    vigs_visualization.filter_thresh = 0.005
    vigs_visualization.follow_cam = True
    vigs_visualization.latest_c2w = None

    def toggle_follow_cam(vis):
        vigs_visualization.follow_cam = not vigs_visualization.follow_cam

    def increase_cam_line_width(vis):
        vigs_visualization.cam_line_width = min(vigs_visualization.cam_line_width + 1.0, 20.0)
        vis.get_render_option().line_width = vigs_visualization.cam_line_width

    def decrease_cam_line_width(vis):
        vigs_visualization.cam_line_width = max(vigs_visualization.cam_line_width - 1.0, 1.0)
        vis.get_render_option().line_width = vigs_visualization.cam_line_width

    def increase_cam_scale(vis):
        vigs_visualization.cam_scale *= 1.5
        with vigs_visualization.video.get_lock():
            vigs_visualization.video.dirty[:vigs_visualization.video.counter.value] = True

    def decrease_cam_scale(vis):
        vigs_visualization.cam_scale /= 1.5
        with vigs_visualization.video.get_lock():
            vigs_visualization.video.dirty[:vigs_visualization.video.counter.value] = True

    def increase_filter(vis):
        vigs_visualization.filter_thresh *= 2
        with vigs_visualization.video.get_lock():
            vigs_visualization.video.dirty[:vigs_visualization.video.counter.value] = True

    def decrease_filter(vis):
        vigs_visualization.filter_thresh *= 0.5
        with vigs_visualization.video.get_lock():
            vigs_visualization.video.dirty[:vigs_visualization.video.counter.value] = True

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                dirty_index, = torch.where(video.dirty.clone())

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            disps = torch.index_select(video.disps, 0, dirty_index)
            images = torch.index_select(video.images, 0, dirty_index.cpu())
            images = images.cpu()[...,3::8,3::8].permute(0,2,3,1) / 255.0
            points = vigs_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = vigs_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            count = vigs_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .1*disps.mean(dim=[1,2], keepdim=True)))
            
            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in vigs_visualization.cameras:
                    vis.remove_geometry(vigs_visualization.cameras[ix])
                    del vigs_visualization.cameras[ix]

                if ix in vigs_visualization.points:
                    vis.remove_geometry(vigs_visualization.points[ix])
                    del vigs_visualization.points[ix]

                # add camera actor
                s = vigs_visualization.cam_scale
                if i == (len(dirty_index) -1):
                    cam_actor = create_camera_actor(True, scale=0.1*0.5*s)
                else:
                    cam_actor = create_camera_actor(False, scale=0.05*0.5*s)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                vigs_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                
                # add point actor
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                vigs_visualization.points[ix] = point_actor

            # store latest camera-to-world for follow-cam
            if len(Ps) > 0:
                vigs_visualization.latest_c2w = Ps[-1]

            # keep the window usable during inference: either follow the newest camera or
            # restore the view control captured at the top of this callback
            if vigs_visualization.follow_cam and vigs_visualization.latest_c2w is not None:
                latest_c2w = vigs_visualization.latest_c2w
                # place viewer 0.5 units behind camera along its local -Z axis
                viewer_c2w = latest_c2w.copy()
                viewer_c2w[:3, 3] = latest_c2w[:3, 3] - 0.5 * latest_c2w[:3, 2]
                viewer_w2c = np.linalg.inv(viewer_c2w)
                cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam_params.extrinsic = viewer_w2c
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            elif len(vigs_visualization.cameras) >= vigs_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            vigs_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    # create Open3D visualization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)
    vis.register_key_callback(ord("F"), toggle_follow_cam)
    vis.register_key_callback(ord("E"), increase_cam_scale)
    vis.register_key_callback(ord("D"), decrease_cam_scale)
    vis.register_key_callback(ord("W"), increase_cam_line_width)
    vis.register_key_callback(ord("Q"), decrease_cam_line_width)
    vis.create_window(height=int((480*2-200)//2*1.3), width=1920-960, left=960, top=0)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.1))

    vis.run()
    vis.destroy_window()
