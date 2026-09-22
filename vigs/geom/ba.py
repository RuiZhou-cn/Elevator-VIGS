"""Factor assembly and the IMU initialization of the tracker's bundle adjustment.

Visual factors (`BA_prepare`), the depth-prior alignment (`JDSA`), the preintegration and bias
factors of Eqs. (2) and (3), and the three-step inertial initialization (gravity direction,
velocity/bias, joint BA with metric scale). The in-ride factors carry the two transport columns
of Eqs. (5) and (6) at the 17-wide state (`TRANSPORT_D`, `U_COL`, `H_COL`), built in C++.
"""
import os
import math
import sys
import numpy as np
import torch
import vigs_backends

_imu_cpp_build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../imu_cpp/build")
if _imu_cpp_build not in sys.path:
    sys.path.append(_imu_cpp_build)

import sophuspy as sp

from .chol import schur_solve_mono_prior, solve_dR, block_solve_imu, schur_solve_imu
import geom.projective_ops as pops

from torch_scatter import scatter_sum

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

# apply retraction operator to velocities
def velo_retr(velos, dv, ii):
    ii = ii.to(device=dv.device)
    return velos + scatter_sum(dv, ii, dim=1, dim_size=velos.shape[1])

# apply retraction operator to biases
def bias_retr(biass, dv, ii):
    ii = ii.to(device=dv.device)
    return biass + scatter_sum(dv, ii, dim=1, dim_size=biass.shape[1])

def BA_prepare(target, weight, eta, poses, disps, intrinsics, ii, jj, T_ci_c0=None,
               H=None, v=None, fixedp=1, D=6, t0=None):
    """ Construct linear system for Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # 1: compute jacobians and residuals
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform_imu(
        poses, disps, intrinsics, ii, jj, jacobian=True, Tcb=T_ci_c0)

    r = (target - coords).view(B, N, -1, 1)
    rw = .001 * (valid * weight).view(B, N, -1, 1)

    # 2: construct linear system
    if D != 6:
        Jnull = torch.cat([torch.zeros_like(Ji), torch.zeros_like(Ji)], dim=-1)[...,:D-6]
        Ji = torch.cat([Ji, Jnull], dim=-1).reshape(B, N, -1, D)
        Jj = torch.cat([Jj, Jnull], dim=-1).reshape(B, N, -1, D)
    else:
        Ji = Ji.reshape(B, N, -1, D)
        Jj = Jj.reshape(B, N, -1, D)
    wJiT = (rw * Ji).transpose(2,3)
    wJjT = (rw * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    rw = rw.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(rw*r*Jz, dim=-1)
    Ck = torch.sum(rw*Jz*Jz, dim=-1)
    
    # only optimize keyframe poses from fixedp (= t0 when given) on
    if t0 is not None:
        fixedp = t0
    P = P - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)
    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)
    C += eta.view(*C.shape) + 1e-7
    E = E.view(B, P, M, D, ht*wd)

    if H is None:
        H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
            safe_scatter_add_mat(Hij, ii, jj, P, P) + \
            safe_scatter_add_mat(Hji, jj, ii, P, P) + \
            safe_scatter_add_mat(Hjj, jj, jj, P, P)
        v = safe_scatter_add_vec(vi, ii, P) + \
            safe_scatter_add_vec(vj, jj, P)
        return H, E, C, v, w
    else:
        H += safe_scatter_add_mat(Hii, ii, ii, P, P) + \
             safe_scatter_add_mat(Hij, ii, jj, P, P) + \
             safe_scatter_add_mat(Hji, jj, ii, P, P) + \
             safe_scatter_add_mat(Hjj, jj, jj, P, P)
        v += safe_scatter_add_vec(vi, ii, P) + \
             safe_scatter_add_vec(vj, jj, P)
        return E, C, w


def get_prior_depth_aligned(depth_prior, scales):
    M, ht, wd = depth_prior.shape
    hs, ws = scales.shape[-2:]
    meshx, meshy = torch.meshgrid(torch.linspace(0, hs-1-1e-6, ht), torch.linspace(0, ws-1-1e-6, wd), indexing='ij')
    grid = torch.stack((meshy, meshx), -1).cuda()
    grid = grid.unsqueeze(0).expand(M, -1, -1, -1).contiguous()
    mscales_bi, Jbi = vigs_backends.bi_inter(scales, grid)
    depth_prior_aligned = depth_prior * mscales_bi
    return depth_prior_aligned, Jbi


def JDSA(target, weight, eta, poses, disps, intrinsics, disps_prior, dscales, ii, jj, alpha):

    B, P, ht, wd = disps.shape

    # 1: compute jacobians and residuals
    C, w = vigs_backends.proj_trans(poses.data.squeeze(), disps[0], intrinsics[0], target, weight, ii, jj)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    disps_prior = disps_prior[kx]
    m = (disps_prior > 0).to(torch.float).view(-1, ht*wd)

    hs, ws = dscales.shape[-2:]
    disps_bi, Jbi = get_prior_depth_aligned(disps_prior, dscales[kx])

    rd = (disps[0,kx] - disps_bi).view(-1, ht*wd)
    Jd = torch.ones_like(rd).view(1, -1, 1, ht*wd)
    Jso = -m.unsqueeze(-1) * disps_prior.view(-1, ht*wd).unsqueeze(-1) * Jbi.view(M, ht*wd, -1)[None]

    alpha = torch.ones(M,ht*wd,1).float().cuda() * alpha

    D = hs*ws
    fixedp = kx[0]
    kx = kx - fixedp
    wJsoT = (alpha * Jso).transpose(2,3)
    Hs = safe_scatter_add_mat(wJsoT @ Jso, kx, kx, M, M).view(B, M, M, D, D)
    Es = safe_scatter_add_mat(wJsoT * Jd, kx, kx, M, M).view(B, M, M, D, ht*wd)
    vs = safe_scatter_add_vec(-wJsoT @ rd[None].unsqueeze(-1), kx, M)
    kx += fixedp

    alpha = alpha.squeeze()
    C = C[None] + m * alpha * (Jd * Jd).squeeze() + (1-m) * eta.view(*C.shape)
    w = w[None] - m * alpha * rd * Jd.squeeze()

    # 3: solve the system
    dso, dz, dzcov = schur_solve_mono_prior(C, w, Hs, Es, vs, dzcov=True)

    # A near-singular prior block lets the Schur solve overflow to +-inf. dscales is persistent
    # state, so a single non-finite step would stay in it for the rest of the run and turn every
    # later BA input into NaN; disps is sanitised below but only against the positive overshoot.
    # Dropping the bad step keeps the keyframe at its current scale, as a failed factorisation
    # does (CholeskySolver returns a zero step). No-op whenever the solve is finite.
    dso = torch.nan_to_num(dso, nan=0.0, posinf=0.0, neginf=0.0)
    dz = torch.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)

    # 4: apply retraction
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    dscales[kx] += dso.view(-1, hs, ws)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.001)

    return disps, dscales, dzcov


# ---- inertial factors ------------------------------------------------------------------------

# State width with the transport state (u_k, h_k) of the paper's Eq. (4): each keyframe of a
# ride carries two more scalars than the base system's 15, so the solve widens to 17.
TRANSPORT_D = 17   # in-ride state width: [pose6|vel3|bias6|u@15|h@16]
BASE_D = 15        # the base system's state outside a ride: [pose6|vel3|bias6]
U_COL = 15         # u: the elevator's vertical velocity
H_COL = 16         # h: the elevator's rise from the departure floor

def get_preint_factors(poses_bw, velos_w, biass_w, preints, Rwg, ii, jj, GDir=False, wo_pose=False, scale=1, preint_scale=1e-5):
    """Per-edge preintegration factors, assembled one edge at a time in Python; the IMU
    initialisation path (get_preint_factors_cpp is the batched hot path of the in-loop BA).
    `preints` is the {(i, j): integrator} dict of C++ integrators, whose methods take
    contiguous float64 arrays."""
    Jinti, Jintj, Jgs = [], [], []
    eint, info = [], []
    # batch the GPU->CPU reads once before the loop to avoid a GPU sync per edge per assembly
    Twb_np = poses_bw[0].inv().matrix().cpu().numpy()
    velos_np = velos_w[0].cpu().numpy()
    biass_np = biass_w[0].cpu().numpy()
    for i, j in zip(ii, jj):
        id0, id1 = i.item(), j.item()
        inter = preints[(id0, id1)]
        inter.set_new_bias(biass_np[id0])
        _args = tuple(np.ascontiguousarray(x, dtype=np.float64)
                      for x in (Twb_np[id0], Twb_np[id1], velos_np[id0], velos_np[id1], Rwg))
        err = inter.compute_error(*_args, scale)
        # jacobian at scale=1 (upstream convention): only the residual carries `scale`
        Ji, Jj, Jvi, Jvj, Jbg, Jba, Js = inter.jacobian(*_args, 1.0)
        Js = Js.reshape(-1, 1)
        eint.append(err)
        info.append(inter.info)
        if wo_pose:
            Jinti.append(np.concatenate([Jvi, Jbg, Jba], axis=-1))
            Jintj.append(np.concatenate([Jvj, np.zeros_like(Jbg), np.zeros_like(Jba)], axis=-1))
        else:
            Jinti.append(np.concatenate([Ji, Jvi, Jbg, Jba], axis=-1))
            Jintj.append(np.concatenate([Jj, Jvj, np.zeros_like(Jbg), np.zeros_like(Jba)], axis=-1))
        if GDir:
            Jgs.append(np.concatenate([inter.jacobian_GDir(*_args), Js], axis=-1))
    info = torch.tensor(np.stack(info, axis=0)[None], dtype=torch.float, device='cuda')
    Jinti = torch.tensor(np.stack(Jinti, axis=0)[None], dtype=torch.float, device='cuda')
    Jintj = torch.tensor(np.stack(Jintj, axis=0)[None], dtype=torch.float, device='cuda')
    eint = torch.tensor(np.stack(eint, axis=0)[None], dtype=torch.float, device='cuda').unsqueeze(-1)
    chi2 = preint_scale * eint.transpose(2,3) @ info @ eint

    wJintiT = preint_scale * torch.matmul(Jinti.transpose(2,3), info)
    wJintjT = preint_scale * torch.matmul(Jintj.transpose(2,3), info)

    Hintii = torch.matmul(wJintiT, Jinti)
    Hintij = torch.matmul(wJintiT, Jintj)
    Hintji = torch.matmul(wJintjT, Jinti)
    Hintjj = torch.matmul(wJintjT, Jintj)

    vinti = torch.matmul(wJintiT, eint).squeeze(-1)
    vintj = torch.matmul(wJintjT, eint).squeeze(-1)

    if GDir:
        Jgs = torch.tensor(np.stack(Jgs, axis=0)[None], dtype=torch.float, device='cuda')
        wJgsT = preint_scale * torch.matmul(Jgs.transpose(2,3), info)
        Hgs = torch.sum(torch.matmul(wJgsT, Jgs), dim=1)
        vgs = torch.sum(torch.matmul(wJgsT, eint), dim=1)
        Hi_gs = torch.matmul(wJintiT, Jgs)
        Hj_gs = torch.matmul(wJintjT, Jgs)
        return Hintii, Hintij, Hintji, Hintjj, vinti, vintj, Hgs, vgs, Hi_gs, Hj_gs, chi2
    return Hintii, Hintij, Hintji, Hintjj, vinti, vintj, chi2


def get_preint_factors_cpp(poses_bw, velos_w, biass_w, integrators, Rwg, ii_cpu, jj_cpu, GDir=False, wo_pose=False, scale=1, preint_scale=1e-5, transport_columns=False):
    """C++-backed preintegration factors over the edge list `integrators` (aligned with
    ii_cpu/jj_cpu): avoids redundant device sync, maps unique indices via vectorized
    np.searchsorted, transfers non-blocking. transport_columns=True emits the 17-wide
    [pose6 | vel3 | bias6 | u@15 | h@16] blocks of the in-ride bundle adjustment
    (elevator.transport.assemble_factors_17w) -- the two transport columns of the paper's
    Eq. (5) and Eq. (6) -- built in C++."""
    from imu_integrator_cpp import batched_preint_factors_cpu

    # Optimization: compute inverse only for unique indices, then index
    # np.unique returns sorted array, use searchsorted for O(K log U) vectorized mapping
    all_indices = np.concatenate([ii_cpu, jj_cpu])
    unique_indices = np.unique(all_indices)
    ii_mapped = np.searchsorted(unique_indices, ii_cpu)
    jj_mapped = np.searchsorted(unique_indices, jj_cpu)

    # Invert only unique poses once on GPU, then transfer (saves ~K/U GPU operations)
    poses_inv_unique = poses_bw[0, unique_indices].inv().matrix().detach().cpu().numpy()

    # CPU-side data preparation
    vel = velos_w[0].detach().cpu().numpy()
    bias = biass_w[0].detach().cpu().numpy()

    G0s = poses_inv_unique[ii_mapped]
    G1s = poses_inv_unique[jj_mapped]
    V0s = vel[ii_cpu]
    V1s = vel[jj_cpu]
    B0s = bias[ii_cpu]

    out = batched_preint_factors_cpu(
        integrators, G0s, G1s, V0s, V1s, B0s, Rwg,
        wo_pose=wo_pose, GDir=GDir, scale=float(scale), transport_columns=transport_columns
    )
    if transport_columns:
        assert out["Jinti"].shape[-1] == TRANSPORT_D and out["Jintj"].shape[-1] == TRANSPORT_D, \
            f"transport_columns=True Jinti/Jintj must be 17-wide, got {out['Jinti'].shape[-1]}/{out['Jintj'].shape[-1]}"

    # Transfer results to GPU with non_blocking
    info  = torch.from_numpy(out["info"]).to("cuda", torch.float32, non_blocking=True)[None]
    Jinti = torch.from_numpy(out["Jinti"]).to("cuda", torch.float32, non_blocking=True)[None]
    Jintj = torch.from_numpy(out["Jintj"]).to("cuda", torch.float32, non_blocking=True)[None]
    eint  = torch.from_numpy(out["eint"]).to("cuda", torch.float32, non_blocking=True)[None].unsqueeze(-1)

    chi2 = preint_scale * eint.transpose(2,3) @ info @ eint

    wJintiT = preint_scale * torch.matmul(Jinti.transpose(2,3), info)
    wJintjT = preint_scale * torch.matmul(Jintj.transpose(2,3), info)

    Hintii = torch.matmul(wJintiT, Jinti)
    Hintij = torch.matmul(wJintiT, Jintj)
    Hintji = torch.matmul(wJintjT, Jinti)
    Hintjj = torch.matmul(wJintjT, Jintj)

    vinti = torch.matmul(wJintiT, eint).squeeze(-1)
    vintj = torch.matmul(wJintjT, eint).squeeze(-1)

    if GDir:
        Jgs = torch.from_numpy(out["Jgs"]).to("cuda", torch.float32, non_blocking=True)[None]
        wJgsT = preint_scale * torch.matmul(Jgs.transpose(2,3), info)
        Hgs = torch.sum(torch.matmul(wJgsT, Jgs), dim=1)
        vgs = torch.sum(torch.matmul(wJgsT, eint), dim=1)
        Hi_gs = torch.matmul(wJintiT, Jgs)
        Hj_gs = torch.matmul(wJintjT, Jgs)
        return Hintii, Hintij, Hintji, Hintjj, vinti, vintj, Hgs, vgs, Hi_gs, Hj_gs, chi2
    return Hintii, Hintij, Hintji, Hintjj, vinti, vintj, chi2

def get_bias_factors(biass_w, preints, ii, jj, wo_pose=False, preint_scale=1e-5):
    """Random-walk bias factors per IMU edge, assembled in Python (the IMU-init path;
    get_bias_factors_cuda is the in-loop one)."""
    Jinti, Jintj = [], []
    eint, info = [], []
    for i, j in zip(ii, jj):
        id0, id1 = i.item(), j.item()
        inter = preints[(id0,id1)]
        Bias0 = biass_w[0,id0]
        Bias1 = biass_w[0,id1]
        err = Bias1 - Bias0
        eint.append(err)
        info.append(inter.info2)
        if wo_pose:
            Ji = np.zeros((6,9))
            Ji[:,3:] = np.eye(6)
        else:
            Ji = np.zeros((6,15))
            Ji[:,9:] = np.eye(6)
        Jj = -Ji
        Jinti.append(Ji)
        Jintj.append(Jj)
    info = torch.tensor(np.stack(info, axis=0)[None], dtype=torch.float, device='cuda')
    Jinti = torch.tensor(np.stack(Jinti, axis=0)[None], dtype=torch.float, device='cuda')
    Jintj = torch.tensor(np.stack(Jintj, axis=0)[None], dtype=torch.float, device='cuda')
    eint = torch.stack(eint, dim=0)[None].unsqueeze(-1)

    preint_scale_prior = preint_scale
    wJintiT = preint_scale_prior * torch.matmul(Jinti.transpose(2,3), info)
    wJintjT = preint_scale_prior * torch.matmul(Jintj.transpose(2,3), info)

    Hintii = torch.matmul(wJintiT, Jinti)
    Hintij = torch.matmul(wJintiT, Jintj)
    Hintji = torch.matmul(wJintjT, Jinti)
    Hintjj = torch.matmul(wJintjT, Jintj)

    vinti = torch.matmul(wJintiT, eint).squeeze(-1)
    vintj = torch.matmul(wJintjT, eint).squeeze(-1)
    return Hintii, Hintij, Hintji, Hintjj, vinti, vintj

def get_bias_factors_cuda(biass_w, ii, jj, preint_scale=1e-5, info2s=None, D=TRANSPORT_D):
    """CUDA-accelerated get_bias_factors: constrains consecutive bias states to a random-walk
    model. biass_w: (1,N,6) [bg(3),ba(3)]. Blocks come out at state width D -- TRANSPORT_D=17 in a
    ride, BASE_D=15 outside one -- the kernel writes only the 9:15 bias sub-block of a
    zero-initialised output, so at 17 the u@15/h@16 columns are already zero and no pad is needed.
    Returns Hintii/Hintij/Hintji/Hintjj (1,num,D,D) and vinti/vintj (1,num,D)."""
    biass = biass_w[0].contiguous()
    ii_long = ii.long().contiguous()
    jj_long = jj.long().contiguous()

    Hii, Hij, Hji, Hjj, vi, vj = vigs_backends.bias_factors(
        biass, info2s, ii_long, jj_long, preint_scale, TRANSPORT_D)
    if D != TRANSPORT_D:
        # the kernel is pinned to TRANSPORT_D (TORCH_CHECK) and writes only the 9:15 block into
        # zeros, so the narrower state is its leading D x D block
        Hii, Hij, Hji, Hjj = (H[..., :D, :D].contiguous() for H in (Hii, Hij, Hji, Hjj))
        vi, vj = vi[..., :D].contiguous(), vj[..., :D].contiguous()

    return Hii, Hij, Hji, Hjj, vi, vj

def get_bias_prior_factors_cuda(biass_w, ii, preint_scale=1e-5, D=TRANSPORT_D):
    """Bias prior: constrains all biases to the reference bias (biass_w[0,0]).
    biass_w: (1,N,6) [bg(3),ba(3)]. Emitted at state width D (TRANSPORT_D=17 in a ride, BASE_D=15
    outside one); the kernel writes only the 9:15 diagonal, leaving any u@15/h@16 at the zero
    it allocates. Returns Hprior (1,num,D,D), vprior (1,num,D)."""
    biass = biass_w[0].contiguous()
    ii_long = ii.long().contiguous()

    Hprior, vprior = vigs_backends.bias_prior_factors(biass, ii_long, preint_scale, TRANSPORT_D)
    if D != TRANSPORT_D:   # kernel pinned to TRANSPORT_D, writes only the 9:15 diagonal: slice (see above)
        Hprior, vprior = Hprior[..., :D, :D].contiguous(), vprior[..., :D].contiguous()

    return Hprior, vprior


def InitializeFullInertialBA(t0, t1, poses_bw, velos_w, biass_w, disps, ii, preints, Rwg, scale, H, E, C, v, w, fix_front, imu_init_fix_scale, bias_scale=1.0):
    """ Full Bundle Adjustment with both reprojection and inertial factors"""

    B, _, ht, wd = disps.shape
    D = poses_bw.manifold_dim + 3 + 6   # Pose(6) + Vel(3) + Bias gyr and acc(6)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]     # number of depth maps to optimize

    # 2b: add preint imu factors
    P = t1 - t0     # number of poses to optimize
    ii = torch.arange(t0  , t1-1, device='cuda') - t0 
    jj = ii + 1
    Hintii, Hintij, Hintji, Hintjj, vinti, vintj, Hgdir, vgdir, Hi_gdir, Hj_gdir, chi2 = get_preint_factors(poses_bw, velos_w, biass_w, preints, Rwg, ii+t0, jj+t0, GDir=True, scale=scale)
    H += safe_scatter_add_mat(Hintii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hintij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hintji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hintjj, jj, jj, P, P)
    v += safe_scatter_add_vec(vinti, ii, P) + \
        safe_scatter_add_vec(vintj, jj, P)

    # 2c: add bias consistency factors
    Hbii, Hbij, Hbji, Hbjj, vbi, vbj = get_bias_factors(biass_w, preints, ii+t0, jj+t0)
    H += safe_scatter_add_mat(Hbii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hbij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hbji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hbjj, jj, jj, P, P)
    v += safe_scatter_add_vec(vbi, ii, P) + \
        safe_scatter_add_vec(vbj, jj, P)

    # 3: solve the system
    H = H.view(B, P, P, D, D)
    if imu_init_fix_scale:
        Hgdir = None
        dxv, dz = schur_solve_imu(H, E, C, v, w, Hgdir, vgdir, Hi_gdir, Hj_gdir, fix_front=fix_front)
    else:
        dxv, dz, ds = schur_solve_imu(H, E, C, v, w, Hgdir, vgdir, Hi_gdir, Hj_gdir, fix_front=fix_front)

    # 4: apply retraction
    poses_bw = pose_retr(poses_bw, dxv[..., :6], torch.arange(P) + t0)
    velos_w = velo_retr(velos_w, dxv[..., 6:9], torch.arange(P) + t0)
    biass_w = bias_retr(biass_w, dxv[..., 9:]*bias_scale, torch.arange(P) + t0)
    disps = disp_retr(disps, dz[:,:M].view(B,-1,ht,wd), kx)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    if not imu_init_fix_scale:
        scale *= math.exp(ds)
    return poses_bw, velos_w, biass_w, disps, scale


def InitializeVeloBiasGdir(t0, t1, poses_bw, velos_w, biass_w, preints, Rwg, scale, imu_init_fix_scale, fix_front=3, bias_scale=1.0):
    """ Initialize velocities, biases and gravity direction via inertial preintegrate/prior factor """
    B, P = poses_bw.shape
    D = 3 + 6   # Vel(3) + Bias gyr and acc(6)

    # 1: compute jacobians and residuals
    # 2b: add preint imu factors
    P = t1 - t0
    ii = torch.arange(0, t1-1 - t0, device='cuda')
    jj = ii + 1
    Hintii, Hintij, Hintji, Hintjj, vinti, vintj, Hgdir, vgdir, Hi_gdir, Hj_gdir, chi2 = get_preint_factors(
                                                                    poses_bw, velos_w, biass_w, preints, Rwg, ii+t0, jj+t0, GDir=True, wo_pose=True, scale=scale)
    H = safe_scatter_add_mat(Hintii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hintij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hintji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hintjj, jj, jj, P, P)
    v = safe_scatter_add_vec(vinti, ii, P) + \
        safe_scatter_add_vec(vintj, jj, P)

    # 2c: add bias consistency factors
    Hbii, Hbij, Hbji, Hbjj, vbi, vbj = get_bias_factors(biass_w, preints, ii+t0, jj+t0, wo_pose=True)
    H += safe_scatter_add_mat(Hbii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hbij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hbji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hbjj, jj, jj, P, P)
    v += safe_scatter_add_vec(vbi, ii, P) + \
        safe_scatter_add_vec(vbj, jj, P)

    # 3: solve the system
    H = H.view(B, P, P, D, D)
    if imu_init_fix_scale:
        Hgdir = Hgdir[:, :-1, :-1]
    dx, dgs = block_solve_imu(H, v, Hgdir, vgdir, Hi_gdir, Hj_gdir, fix_front=fix_front)
    # 4: apply retraction
    velos_w = velo_retr(velos_w, dx[..., :3], torch.arange(P) + t0)
    biass_w = bias_retr(biass_w, dx[..., 3:]*bias_scale, torch.arange(P) + t0)
    Rwg = Rwg @ sp.SO3.exp(dgs[0,:3].cpu().numpy()).matrix()
    if not imu_init_fix_scale:
        scale *= math.exp(dgs[0,3])

    return velos_w, biass_w, Rwg, scale

def InitializeGravityDirectionDynamic(t0, t1, poses_bw, velos_w, biass_w, preints, Rwg):
    """ Initialize gravity direction via inertial preintegrate factor """
    ii = torch.arange(0, t1-1 - t0, device='cuda') 
    jj = ii + 1
    _, _, _, _, _, _, Hgdir, vgdir, _, _, chi2 = get_preint_factors(poses_bw, velos_w, biass_w, preints, Rwg, ii+t0, jj+t0, GDir=True, wo_pose=True)

    dR = solve_dR(Hgdir, vgdir)
    return Rwg @ sp.SO3.exp(dR[0,:3].cpu().numpy()).matrix()

