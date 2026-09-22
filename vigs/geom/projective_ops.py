import torch

from .pinhole import proj_pinhole, iproj_pinhole
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float(), indexing='ij')

    return torch.stack([x, y], dim=-1)

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def actp_imu(Gij, X0, Gbij, Tcb, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        if Gbij is not None:
            X1b = Gbij[:,:,None,None] * X0
            X, Y, Z, d = X1b.unbind(dim=-1)
        else:
            X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        Ja = torch.stack([
            d,  o,  o,  o,  Z, -Y,
            o,  d,  o, -Z,  o,  X,
            o,  o,  d,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(B, N, H, W, 4, 6)

        if Tcb is not None:
            Ja = Tcb.matrix() @ Ja
        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """

    # inverse project
    X0, Jz = iproj_pinhole(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)   # (pinhole)

    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()

    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # project
    intr_proj = intrinsics[:,jj]
    x1, Jp = proj_pinhole(X1, intr_proj, jacobian=jacobian, return_depth=return_depth)   # (pinhole)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def projective_transform_imu(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False, Tcb=None):
    """ map points from ii->jj, with `poses` in the body (IMU) frame when Tcb is given """
    X0, Jz = iproj_pinhole(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    # transform
    if Tcb is not None:
        Gbibj = poses[:,jj] * poses[:,ii].inv()
        Gbij = Gbibj * Tcb.inv()
        Gij = Tcb * Gbij
    else:
        Gbij = None
        Gij = Gbibj = poses[:,jj] * poses[:,ii].inv()

    X1, Ja = actp_imu(Gij, X0, Gbij, Tcb, jacobian=jacobian)
    
    x1, Jp = proj_pinhole(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too far to camera
    basesize = 1 / torch.clamp(torch.norm(Gij.data[:, :, :3], dim=2)*40, min=5., max=100.)
    valid = (X0[...,3] > basesize[...,None, None]).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gbibj[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid
