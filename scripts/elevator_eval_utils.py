#!/usr/bin/env python3
"""The elevator benchmarks' metrics, numpy-only, called in-process by eval_elevator_mono.py.

  * the real height metric: heights read on the still windows of a recording's control_points.json
    along the run's OWN gravity, nothing aligned -- `score_pairs` below is the single
    statement of the metric every real number is read with;
  * the sim ATE: Sim(3) (Umeyama with scale, the public sets' `evo_ape -vas` protocol) fitted
    on the whole trajectory -- `compute_ate`.
"""
import numpy as np


# control-point labels ("control point" is the paper's word and the released dataset's)
# One label per set-down, in recording order: PLACE letter (A = recording start/end, B =
# departure floor, C = arrival floor, D.. = further plates in order of first visit) + VISIT
# count (B1 before the ride, B2 on the return) -- same letter = same plate.
def place(label):
    """'B2' -> 'B'."""
    return label.rstrip("0123456789")


def visit(label):
    """'B2' -> 2 ('B' alone counts as visit 1)."""
    return int(label[len(place(label)):] or 1)


def control_point_chain(points):
    """Every scored control point in time order, i.e. all plates except A (A1 is the init
    still-head and A2 the power-off set-down; neither bounds a ride)."""
    return sorted((l for l in points if place(l) != "A"), key=lambda l: points[l][0])


def is_round_trip(points):
    """A revisited plate in the chain closes the loop (B2); otherwise the sequence is
    one-way. Decided by the labels alone, never by the sequence name."""
    return any(visit(l) >= 2 for l in control_point_chain(points))


# gravity
def gravity_up(Rwg, init_g):
    """Unit gravity-up vector in the SLAM world frame: -Rwg @ init_g, normalized."""
    g_w = np.asarray(Rwg, dtype=np.float64) @ np.asarray(init_g, dtype=np.float64)
    return -g_w / np.linalg.norm(g_w)


# the real height metric on the control-point windows -- one trajectory, one table
# The single statement of the metric every real number is read with. Rules:
#   * the windows come from the RAW IMU (control_points.json, shipped with the recording), so every
#     method is read at the SAME instants and nothing is aligned to anything;
#   * `h` at a window = the MEDIAN over every pose the run has inside it, along the run's
#     OWN gravity (gravity_up above). The device is at rest, so the true height is constant
#     and the median estimates it with less noise than any single frame, immune to one bad
#     keyframe and free of a "which frame" choice;
#   * no interpolation onto a common instant -- an interpolator silently bridges a
#     tracking gap, which is the failure the coverage test exists to catch;
#   * a window with NO pose inside is read from the NEAREST pose, if that pose is within
#     VNEAR_S of the window, since a run that tracks most of the time should not simply
#     fail here. The device is on the same floor for seconds on either side of a set-down,
#     so a pose that close reads the plate to within the carry height. Nothing within
#     VNEAR_S -> fail: the run really did not reach the plate. Only the CONTROL-POINT
#     windows (B1/C1, B1/B2) get this; A1/A2 stay strict, see score_pairs.
# Labels are PLACE letter + VISIT count (see place/visit above): on a one-way sequence
# (A1/B1/C1) the only number is the rise h(C1)-h(B1) vs the laser +H -- the metric never reads
# the recording's first and last pose, so a run that initialises after A1 is still scored.
# On a round trip: e_z^B = h(B2)-h(B1) (closes on the departure plate B, brackets the rides;
# late-init runs still cover it, so this is the primary), e_z^A = h(A2)-h(A1) (adds the
# approach and return walk; blank for a run that initialises after A1 -- an expected
# property, NOT a fail), and when the arrival plate C1 exists the per-ride split
# up = h(C1)-h(B1) vs +H, down = h(B2)-h(C1) vs -H (a `downup` sequence rides down first:
# pass the laser H with a minus sign).
# Yaw is irrelevant here: rotating the device about gravity between two control points leaves
# lever arm's vertical component untouched, so no cam/imu extrinsic conversion is needed.
# Only tilt matters, and control_points.json's dtilt column audits it.
VNEAR_S = 5.0           # s outside an empty window within which the nearest pose still reads it


def h_at(t, h, lo, hi, near=VNEAR_S):
    """(height, pose count inside, covered fraction of the window).

    Poses inside the window: their MEDIAN. None inside: the single pose nearest to the
    window, if it lies within `near` seconds of it. Nothing that close: nan, and the pair
    is a fail. `n` counts poses INSIDE only, so a fallback reading is recognisable as one.

    `cov` guards the case the count alone misses: a run that dies a second into a 44 s
    window still has poses in it, but they all sit at one edge and the last of them may
    be the garbage it died on. A healthy run at rest spans the window."""
    m = (t >= lo) & (t <= hi)
    if m.any():
        return (float(np.median(h[m])), int(m.sum()),
                float((t[m].max() - t[m].min()) / (hi - lo)))
    cand = []
    b, a = t < lo, t > hi
    if b.any():
        i = int(np.argmax(np.where(b, t, -np.inf)))
        cand.append((lo - t[i], float(h[i])))
    if a.any():
        i = int(np.argmin(np.where(a, t, np.inf)))
        cand.append((t[i] - hi, float(h[i])))
    cand = [c for c in cand if c[0] <= near]
    if not cand:
        return float("nan"), 0, 0.0
    return min(cand)[1], 0, 0.0


def score_pairs(points, t, h, H=None, echo=lambda *_: None):
    """The metric of Sec. 4.3, once, for one trajectory. Returns ({label: cm or None},
    {control point: (h, n, cov)}).

    Pairing follows the control-point set, so one function serves both sequence kinds:
      no revisit  -> one-way: `rise` between the chain's ends (B1, C1), against +H
      B2 present  -> round trip: e_z^B (B1,B2) and e_z^A (A1,A2); with the
                     arrival plate C1 present, the first and last rides against +/-H
    A1 is never scored on a one-way sequence: it is the first window, at t=0, where the
    late-initialising rows have no pose -- moving the control points off the recording ends
    is the whole point, so it stays the init still-head."""
    # nearest-pose fallback serves the CONTROL POINTS only (chain: B1/C1 one-way, B1/B2
    # round trip); A1/A2 stay strict since a late-initialising row's first pose after A1 is
    # its init at carry height, not a reading of the plate.
    chain = control_point_chain(points)
    val = {}
    for k, (lo, hi) in points.items():
        val[k] = h_at(t, h, lo, hi, near=VNEAR_S if k in chain else 0.0)
    out = {}

    def pair(x, y, name, ref=0.0):
        if x not in val or y not in val:
            return
        hx, hy = val[x][0], val[y][0]
        if np.isnan(hx) or np.isnan(hy):
            out[name] = None
            echo(f"{name:>10}: fail (init/lost) -- no pose within {VNEAR_S:.0f} s of "
                 f"{x if np.isnan(hx) else y}")
            return
        d = (hy - hx) * 100.0
        out[name] = (abs(d) - ref * 100.0) if ref else d
        echo(f"{name:>10}: {d:+8.1f} cm   (ref {ref * 100:+.1f})"
             + ("" if not ref else f"   err {abs(d) - ref * 100:+8.1f} cm"))

    ch = control_point_chain(points)
    if not is_round_trip(points):
        if len(ch) >= 2:
            pair(ch[0], ch[-1], "rise", H or 0.0)
    else:
        pair(ch[0], ch[-1], "e_z B")                  # B1 -> B2, the departure plate
        pair("A1", "A2", "e_z A")                     # A1 -> A2, the start/end plate
        rides = list(zip(ch, ch[1:]))
        if len(rides) >= 2 and not any(np.isnan(val[k][0]) for k in ch):
            pair(*rides[0], "up ride", H or 0.0)
            pair(*rides[-1], "down ride", -(H or 0.0))
    return out, val


def umeyama_sim3(src, dst):
    """Similarity s,R,t (with scale) s.t. dst ~= s*R@src + t. src,dst: (N,3).
    The alignment of the sim ATE (compute_ate); real tables are unaligned (score_pairs)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = ((dst - mu_d).T @ (src - mu_s)) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    Rot = U @ D @ Vt
    var_src = ((src - mu_s) ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_src)
    return s, Rot, mu_d - s * Rot @ mu_s


# Sim only: every method, ours included, is read under one Sim(3) (Umeyama w/ scale,
# the base system's public-set protocol, `evo_ape -vas`); real tables are unaligned
# control-point reads (score_pairs) -- there is no GT trajectory to align to on real data.
def compute_ate(kf_t, kf_pos, gt_t, gt_pos):
    """Whole-trajectory ATE [cm] for sim seqs with full GT trajectory.

    The one sim protocol: Sim(3) (Umeyama with scale) fitted on EVERY pose, the same one
    the base system runs on the public sets. `scale` is the fitted scale (|1-s| = the scale
    error); `final_cm` is the last keyframe's error, the drift the run ends on."""
    g_at = np.stack([np.interp(kf_t, gt_t, gt_pos[:, k]) for k in range(3)], axis=1)
    est = np.asarray(kf_pos, float)
    s, Rot, t = umeyama_sim3(est, g_at)
    e = np.linalg.norm(s * (Rot @ est.T).T + t - g_at, axis=1)
    return dict(ate_cm=float(np.sqrt((e ** 2).mean()) * 100),
                final_cm=float(e[-1] * 100),
                scale=float(s),
                n=len(kf_t))
