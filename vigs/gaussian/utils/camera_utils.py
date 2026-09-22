"""The per-view camera the mapper trains and renders with: the pose (with the pose and exposure
deltas the mapping loss optimizes), the intrinsics and projection matrix, and GPU copies of the
keyframe's image, depth and normal priors."""
import torch
from torch import nn
from gaussian.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        normal,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.normal = normal
        self.grad_mask = None
        self.cache_gpu()

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_tracking(color, depth, normal, pose, idx, projection_matrix, K, tstamp=None):
        cam = Camera(
            idx,
            color,
            depth,
            normal,
            pose,
            projection_matrix,
            K[0],
            K[1],
            K[2],
            K[3],
            focal2fov(K[0], K[-2]),
            focal2fov(K[1], K[-1]),
            K[-1],
            K[-2])
        cam.R = pose[:3, :3]
        cam.T = pose[:3, 3]
        cam.tstamp = tstamp
        return cam
    
    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )
        
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def cache_gpu(self):
        """GPU copies of image / depth / normal, so training does not re-upload them per use
        (GSBackEnd.offload_caches drops them for the final BA, restore_caches rebuilds them)."""
        dev = self.device
        self.original_image_gpu = (torch.as_tensor(self.original_image, dtype=torch.float32, device=dev)
                                   if self.original_image is not None else None)
        self.depth_gpu = (torch.as_tensor(self.depth, dtype=torch.float32, device=dev).unsqueeze(0)
                          if self.depth is not None else None)
        self.normal_gpu = (torch.as_tensor(self.normal, dtype=torch.float32, device=dev)
                           if self.normal is not None else None)
