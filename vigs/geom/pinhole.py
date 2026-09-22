import torch

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:4].unbind(dim=-1)

_GRID = {}   # (ht, wd, device) -> (y, x) pixel grid; rebuilt per call it is two host->device copies


def _pixel_grid(ht, wd, device):
    key = (int(ht), int(wd), str(device))
    g = _GRID.get(key)
    if g is None:
        g = torch.meshgrid(
            torch.arange(ht).to(device).float(),
            torch.arange(wd).to(device).float(), indexing='ij')
        _GRID[key] = g
    return g


def iproj_pinhole(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = _pixel_grid(ht, wd, disps.device)

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj_pinhole(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.1, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None
