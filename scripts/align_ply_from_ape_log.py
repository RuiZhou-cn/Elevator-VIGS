"""Load a saved Gaussian map (ply) and align it with the Sim(3) that evo fitted to its trajectory.

`load_gaussians_from_ply` rebuilds a GaussianModel from the ply demo.py saves;
`align_ply_from_ape_log` reads the scale, rotation and translation from
an `evo_ape -vas` log and writes the aligned ply next to the ground truth, which is what
eval_public_mono.py does for every mapped sequence.
"""
import math
import os
import re
import sys

import numpy as np
import torch
from plyfile import PlyData

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vigs'))
from gaussian.scene.gaussian_model import GaussianModel


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = list(v.data.dtype.names)

    x = v.data["x"].astype(np.float32)
    y = v.data["y"].astype(np.float32)
    z = v.data["z"].astype(np.float32)
    xyz = np.stack([x, y, z], axis=1)

    f_dc_cols = [n for n in names if n.startswith("f_dc_")]
    f_rest_cols = [n for n in names if n.startswith("f_rest_")]
    scale_cols = [n for n in names if n.startswith("scale_")]
    rot_cols = [n for n in names if n.startswith("rot_")]
    has_opacity = "opacity" in names

    f_dc_cols.sort(key=lambda s: int(s.split("_")[-1]))
    f_rest_cols.sort(key=lambda s: int(s.split("_")[-1]))
    scale_cols.sort(key=lambda s: int(s.split("_")[-1]))
    rot_cols.sort(key=lambda s: int(s.split("_")[-1]))

    # Infer SH degree from feature counts
    num_f_dc = len(f_dc_cols)            # expected 3
    num_f_rest = len(f_rest_cols)        # expected 3*((sh+1)^2 - 1)
    if num_f_dc % 3 != 0:
        raise ValueError(f"Unexpected f_dc count: {num_f_dc}")
    total_channels_per_color = (num_f_rest // 3) + 1 if num_f_rest > 0 else 1
    sh_degree = int(round(math.sqrt(total_channels_per_color) - 1))
    if (sh_degree + 1) ** 2 != total_channels_per_color:
        raise ValueError(f"Cannot infer SH degree from f_rest count={num_f_rest}")

    # Build features [N, 3, (sh+1)^2], split into dc [:, :, 0:1] and rest [:, :, 1:]
    N = xyz.shape[0]
    C = (sh_degree + 1) ** 2
    features = np.zeros((N, 3, C), dtype=np.float32)
    if num_f_dc > 0:
        f_dc_flat = np.stack([v.data[c].astype(np.float32) for c in f_dc_cols], axis=1)  # [N, 3]
        features[:, :, 0] = f_dc_flat.reshape(N, 3)
    if num_f_rest > 0:
        f_rest_flat = np.stack([v.data[c].astype(np.float32) for c in f_rest_cols], axis=1)  # [N, 3*(C-1)]
        features[:, :, 1:] = f_rest_flat.reshape(N, 3, C - 1)

    features_nc3 = np.transpose(features, (0, 2, 1))  # [N, C, 3]

    # Opacity [N, 1] in inverse-sigmoid space per save_ply
    if has_opacity:
        opacity = v.data["opacity"].astype(np.float32).reshape(N, 1)
    else:
        opacity = np.zeros((N, 1), dtype=np.float32)

    # Scaling can be isotropic (1) or anisotropic (3)
    if len(scale_cols) == 0:
        scaling = np.zeros((N, 1), dtype=np.float32)
    else:
        scaling = np.stack([v.data[c].astype(np.float32) for c in scale_cols], axis=1)  # [N, S]

    # Rotation is quaternion [w, x, y, z] length 4
    if len(rot_cols) == 4:
        rotation = np.stack([v.data[c].astype(np.float32) for c in rot_cols], axis=1)  # [N, 4]
    else:
        rotation = np.zeros((N, 4), dtype=np.float32)
        rotation[:, 0] = 1.0

    # Create model and set tensors (in the same representation as saved)
    gm = GaussianModel(sh_degree=sh_degree, config=None)
    gm._xyz = torch.from_numpy(xyz).to(device)
    gm._features_dc  = torch.from_numpy(features_nc3[:, 0:1, :]).to(device)    # [N, 1, 3]
    gm._features_rest= torch.from_numpy(features_nc3[:, 1:,  :]).to(device) # [N, C-1, 3]
    gm._opacity = torch.from_numpy(opacity).to(device)
    # _scaling is stored in log space and save_ply writes the raw tensor, so it is assigned as is
    gm._scaling = torch.from_numpy(scaling).to(device)
    gm._rotation = torch.from_numpy(rotation).to(device)

    # Initialize aux tensors expected by the optimizer/pipeline
    gm.max_radii2D = torch.zeros((N,), device=device)
    gm.xyz_gradient_accum = torch.zeros((N, 1), device=device)
    gm.unique_kfIDs = torch.zeros((N,), dtype=torch.int32, device="cpu")
    gm.n_obs = torch.zeros((N,), dtype=torch.int32, device="cpu")

    # Make SH active degree consistent
    gm.active_sh_degree = min(gm.max_sh_degree, sh_degree)
    return gm


def parse_ape_log(log_path):
    """
    Reads the evo APE log and returns (scale: float, R: (3,3) np.array, t: (3,) np.array).
    """
    with open(log_path, "r") as f:
        text = f.read()

    # Rotation block
    rot_pat = r"Rotation of alignment:\s*\[\[([^\]]+)\]\s*\[([^\]]+)\]\s*\[([^\]]+)\]\]"
    m_rot = re.search(rot_pat, text)
    if not m_rot:
        # handle compact one-line [[...],[...],[...]]
        rot_pat2 = r"Rotation of alignment:\s*\[\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\]\]"
        m_rot = re.search(rot_pat2, text)
        if not m_rot:
            raise ValueError("Could not parse rotation matrix from log.")

    def _row_to_floats(row_str):
        return [float(x) for x in re.split(r"[,\s]+", row_str.strip()) if x]

    r1 = _row_to_floats(m_rot.group(1))
    r2 = _row_to_floats(m_rot.group(2))
    r3 = _row_to_floats(m_rot.group(3))
    R = np.array([r1, r2, r3], dtype=np.float64)

    # Translation
    t_pat = r"Translation of alignment:\s*\[([^\]]+)\]"
    # [^\]] already matches newlines, so this handles the multi-line form too
    m_t = re.search(t_pat, text)
    if not m_t:
        raise ValueError("Could not parse translation from log.")
    t = np.array(_row_to_floats(m_t.group(1)), dtype=np.float64).reshape(3)

    # Scale
    s_pat = r"Scale correction:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    m_s = re.search(s_pat, text)
    if not m_s:
        raise ValueError("Could not parse scale correction from log.")
    s = float(m_s.group(1))

    return s, R, t


def _quat_mul(q1, q2):
    """Hamilton product (w,x,y,z)."""
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w,x,y,z], dim=-1)


def _rotmat_to_quat(R):
    """(3,3) -> (4,) quaternion (w,x,y,z)."""
    t = R.trace()
    if t > 0:
        r = torch.sqrt(1.0 + t)
        w = 0.5 * r
        s = 0.5 / r
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        diag = torch.tensor([R[0,0], R[1,1], R[2,2]], device=R.device, dtype=R.dtype)
        i = int(torch.argmax(diag))
        if i == 0:
            r = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            x = 0.5 * r; s = 0.5 / r
            y = (R[0,1] + R[1,0]) * s
            z = (R[0,2] + R[2,0]) * s
            w = (R[2,1] - R[1,2]) * s
        elif i == 1:
            r = torch.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2])
            y = 0.5 * r; s = 0.5 / r
            x = (R[0,1] + R[1,0]) * s
            z = (R[1,2] + R[2,1]) * s
            w = (R[0,2] - R[2,0]) * s
        else:
            r = torch.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2])
            z = 0.5 * r; s = 0.5 / r
            x = (R[0,2] + R[2,0]) * s
            y = (R[1,2] + R[2,1]) * s
            w = (R[1,0] - R[0,1]) * s
    q = torch.stack([w,x,y,z])
    return torch.nn.functional.normalize(q, dim=0)


def align_gaussians_inplace(gm, global_scale, r_a_np, t_a_np):
    """
    Apply x' = R (s x) + t, scale sizes by s, compose rotations with R (left-multiply).
    Operates on the GIVEN GaussianModel (in-place).
    Assumes:
      - gm._xyz: (N,3)
      - gm._scaling: (N,3) in log-space (log-sigmas or log-scales)
      - gm._rotation: (N,4) quaternions as (w,x,y,z)
    """
    device = gm._xyz.device
    dtype  = gm._xyz.dtype
    s = float(global_scale)

    R = torch.from_numpy(r_a_np).to(device=device, dtype=dtype)        # (3,3)
    t = torch.from_numpy(t_a_np).to(device=device, dtype=dtype).view(1,3)

    N = gm._xyz.shape[0]
    with torch.no_grad():
        # means
        gm._xyz = gm._xyz * s
        gm._xyz = gm._xyz @ R.T + t
        # log-scales (each axis)
        gm._scaling = gm._scaling + math.log(s)
        # orientations (compose: q' = q_R * q_old)
        q_R = _rotmat_to_quat(R).view(1,4).expand(N,4)
        q_new = _quat_mul(q_R, gm._rotation)
        gm._rotation = torch.nn.functional.normalize(q_new, dim=1)


def align_ply_from_ape_log(
    ape_log_path: str,
    ply_in: str,
    ply_out: str,
):
    """
    Parse evo APE results, load a Gaussian PLY, align in-place, and save.
    """
    global_scale, R, t = parse_ape_log(ape_log_path)

    gs = load_gaussians_from_ply(ply_in)

    align_gaussians_inplace(gs, global_scale, R, t)

    gs.save_ply(ply_out)   # raises on failure

    return dict(scale=global_scale, R=R, t=t, out=ply_out)

