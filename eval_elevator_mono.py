#!/usr/bin/env python3
"""Elevator evaluation on our two benchmarks, real and sim, one entry point.

Usage:  python eval_elevator_mono.py                        # both benchmarks, every sequence
        python eval_elevator_mono.py --gsmapping            # + train the Gaussian map (online)
        python eval_elevator_mono.py --gsmapping --offline  # + final BA / GS refinement

Every sequence of both benchmarks is run and scored, then the summary table
is printed. Runs land in outputs/output_elevator/opensource/<Sequence>/, the layout of the base
system's eval_*_mono.py; one whose trajectory is already there is scored, not re-run (rm its run
dir), so a second call is the table on its own. Both domains always run the elevator path; for the
original VIGS-SLAM, run the upstream code (https://github.com/cvg/VIGS-SLAM) on the same
sequences. Public datasets are eval_public_mono.py. The dataset is one tree,
data/Elevator-VIGS/{real,sim}/<Sequence>/ (README, Elevator-VIGS Dataset), one folder per domain.

real  a recording: images/, imu.txt, calib.txt, MT-Pose.txt, control_points.json
    (+ the rig's MT-Cloud.ply / MT-Traj.ply and lidar/, unused here), one rig across every
    site. The metric reads heights on control_points.json's still windows along the run's OWN
    gravity, nothing aligned (scripts/elevator_eval_utils.py, score_pairs): one-way ->
    the rise vs the laser-measured H in config/elevator_real.yaml
    Elevator.sequences[<Sequence>].gt_rise, round trip -> the closure error e_z on the
    departure plate. In the set: any sequence whose control_points.json carries labelled plates.

sim  an Isaac Sim capture: images/, imu.txt, calib.txt, MT-Pose.txt, events.json.
    MT-Pose.txt is exact body-frame ground truth on the epoch clock, so the metric is a
    Sim(3)-aligned ATE (Umeyama with scale, the public sets' `evo_ape -vas` protocol) of the
    keyframe trajectory against the camera-frame GT (MT-Pose through the config's Tcb_np). A
    `-Walk` capture has no ride in events.json -- the elevator-free control row. In the set:
    any sequence with an events.json.
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

from scripts.elevator_eval_utils import (
    compute_ate, gravity_up, control_point_chain, is_round_trip, score_pairs,
)

# user config: what is formatted into the demo.py command line. Sensor and IMU-init keyframe
# (IMU noise/extrinsics, imu_late_init_from, pgba) are the config's, the shared algorithm
# hyper-parameters vigs/config_defaults.py's.
CONFIG = {"real": "config/elevator_real.yaml", "sim": "config/elevator_sim.yaml"}
BUFFER             = 2500   # sim only: demo.py's 1200 default overflows the long captures (Mall-M hits KF 1200 at frame 3745/4051)
output_folder      = "outputs/output_elevator/opensource"   # one run dir per sequence, <output_folder>/<Sequence>/
DATA_ROOT          = "data/Elevator-VIGS"   # data/Elevator-VIGS/{real,sim}/<Sequence>/
REPORT_JSON        = f"{output_folder}/real_results.json"   # the real-domain table, as printed
DIVERGED_X         = 10.0   # spatial extent, as a multiple of the rig's own, before a run's number is refused


def demo_cmd(out, config, data_dir, args, extra=""):
    """The demo.py invocation both domains run. Its default is the ONLINE state (no final BA, no
    GS refinement) and the metric reads the same traj_kf_beforeBA.txt in every mode: --gsmapping
    trains the map alongside, --offline adds the final BA + GS colour refinement. The run
    leaves init_dump.json, the real metric's own-gravity source."""
    return (f"python demo.py"
            f" --calib {data_dir}/calib.txt"
            f" --imagedir {data_dir}/images"
            f" --config {config}"
            f" --imufile {data_dir}/imu.txt"
            f" --output {out}"
            f"{extra}"
            + (" --gsmapping" if args.gsmapping else "")
            + (" --offline" if args.offline else "")
            + f" > {out}/log.txt 2>&1")


def run_demo_if_needed(cmd, out, args):
    """Run the demo unless the run dir already holds the scored trajectory (as eval_public_mono.py).
    rm the run dir, or pass --force, to re-run it regardless. False when no trajectory came out
    of it."""
    traj_path = f"{out}/traj_kf_beforeBA.txt"
    print("RUNNING COMMAND: ", cmd)
    if args.force and os.path.exists(traj_path):
        os.remove(traj_path)
    if args.force or not os.path.exists(traj_path):
        os.system(cmd)
    if not os.path.exists(traj_path):
        print(f"[ERROR] {traj_path} not produced; see {out}/log.txt")
        return False
    return True


# domain real
def real_sequences():
    """{sequence: dir} for every recording with labelled control points -- any
    data/Elevator-VIGS/real/<Sequence>/control_points.json carrying them, so a new recording joins
    the table by being on disk."""
    out = {}
    for cj in sorted(glob.glob(f"{DATA_ROOT}/real/*/control_points.json")):
        if json.load(open(cj)).get("control_points"):
            out[os.path.basename(os.path.dirname(cj))] = os.path.dirname(cj)
    return out


def extent_ratio(pos, d):
    """How far the estimate wandered from its start, over how far the rig ever did (MT-Pose.txt).
    The divergence guard: a run that left the building by tens of times the rig's extent (a
    false loop closure) would otherwise report hundreds of metres as a height reading. It
    refuses the NUMBER, not the trajectory."""
    g = np.loadtxt(f"{d}/MT-Pose.txt")[:, 1:4]
    ref = float(np.linalg.norm(g - g[0], axis=1).max())
    if ref <= 0:
        return None
    return float(np.linalg.norm(pos - pos[0], axis=1).max()) / ref


def score_real(key, d, out, points, H):
    """Score one finished real run: heights on control_points.json's still windows along the run's
    OWN gravity (init_dump.json), nothing aligned to MT-Pose. One-way -> rise h(C1)-h(B1) vs +H;
    round trip -> e_z^B (B1->B2) and e_z^A (A1->A2), plus the per-ride split when the arrival
    plate C1 exists. Prints the per-control-point table, writes <out>/control_point_metrics.json, returns the
    corpus-table fields."""
    dump = f"{out}/init_dump.json"
    if not os.path.exists(dump):
        print(f"[ERROR] {dump} missing: the run never initialised its gravity")
        return {}
    g = np.loadtxt(f"{out}/traj_kf_beforeBA.txt")
    t, pos = g[:, 0], g[:, 1:4]
    m = json.load(open(dump))
    up = gravity_up(np.array(m["Rwg"]).reshape(3, 3), np.array(m["init_g"]))
    h = pos @ up

    print("=" * 64)
    print(f"{out}/traj_kf_beforeBA.txt   up = [{', '.join(f'{v:+.4f}' for v in up)}]")
    span = (min(v[0] for v in points.values()), max(v[1] for v in points.values()))
    print(f"       traj {t[0]:.1f}..{t[-1]:.1f} s  vs points {span[0]:.1f}..{span[1]:.1f} s")
    scores, val = score_pairs(points, t, h, H, echo=lambda s: print("\n" + s))
    print(f"{'point':>6} {'t0':>10} {'t1':>10} {'n':>6} {'cov':>6} {'h [m]':>9}")
    for k, (lo, hi) in points.items():
        hk, n, cov = val[k]
        print(f"{k:>6} {lo:10.3f} {hi:10.3f} {n:6d} {cov:5.0%} "
              + ("      ---" if np.isnan(hk) else f"{hk:9.3f}"))
    if len(control_point_chain(points)) < 2:
        print("\n(fewer than two chain control points labelled -- nothing to pair)")
    json.dump(scores, open(f"{out}/control_point_metrics.json", "w"), indent=1)

    status = None
    ext = extent_ratio(pos, d)
    if ext is not None and ext > DIVERGED_X:
        print(f"{key}: diverged, spatial extent {ext:.0f}x MT-Pose's -- trajectory only, "
              "no number")
        scores = {k: None for k in scores}
        status = f"diverged x{ext:.0f}"
    print("=" * 64)
    return dict(run=out, scores=scores, status=status,
                control_points={k: dict(h=v[0], n=v[1], cov=v[2]) for k, v in val.items()})


def run_real(args):
    cfg_path = CONFIG["real"]
    gt_rise = {k: (v or {}).get("gt_rise") for k, v in
               (yaml.safe_load(open(cfg_path))["Elevator"].get("sequences") or {}).items()}
    seqs = real_sequences()
    if not seqs:
        raise SystemExit(f"no recording with a control_points.json under {DATA_ROOT}/")
    rows, failed = {}, []
    for key, d in seqs.items():
        points = json.load(open(f"{d}/control_points.json"))["control_points"]
        H = gt_rise.get(key)
        row = rows[key] = dict(kind="round-trip" if is_round_trip(points) else "one-way",
                               H=H, scores=None, status=None)
        print(f"{'=' * 60}\n{key}\n{'=' * 60}")
        if row["kind"] == "one-way" and H is None:
            print(f"[warn] {key}: no laser gt_rise in {os.path.basename(cfg_path)} -- "
                  "reporting the raw rise, not the error")
        out = f"{output_folder}/{key}"
        os.makedirs(out, exist_ok=True)
        if run_demo_if_needed(demo_cmd(out, cfg_path, d, args), out, args):
            row.update(score_real(key, d, out, points, H))
        else:
            failed.append(key)

    # the corpus table, one row per recording: one-way the rise error vs the laser H, round trip
    # e_z^B (score_pairs, as above). A cell with no number names which failure it was -- the run
    # diverged, so its heights read nothing (score_real), or it never had a pose at one of the
    # pair's control points (score_pairs' `fail (init/lost)`); `-` is a run with no trajectory.
    print("\n" + "=" * 60)
    for key, r in rows.items():
        what = "rise" if r["kind"] == "one-way" else "e_z B"
        v = (r["scores"] or {}).get(what, "missing")
        if v is None and not r["status"]:
            r["status"] = "no track"
        cell = (f"{v:+9.1f} cm" if v is not None and v != "missing"
                else f"{r['status'] if v is None else '-':>9}")
        print(f"{key:52s} {r['kind']:9s} H={r['H'] if r['H'] else float('nan'):7.3f}  "
              f"{what:>9} {cell}")
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    json.dump(rows, open(REPORT_JSON, "w"), indent=1, default=float)
    print(f"\n{len(rows)} sequence(s) -> {REPORT_JSON}")
    return failed


# domain sim
def sim_sequences():
    """{capture: dir} for every Isaac Sim capture -- any data/Elevator-VIGS/sim/<Sequence>/events.json."""
    return {os.path.basename(os.path.dirname(ej)): os.path.dirname(ej)
            for ej in sorted(glob.glob(f"{DATA_ROOT}/sim/*/events.json"))}


def quat_wxyz_to_R(q):
    """(N, 4) unit quaternions in `w x y z` order (MT-Pose.txt's) -> (N, 3, 3)."""
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def gt_cam(d, Tcb):
    """(t, xyz) of the CAMERA in the sim world, the frame the run's trajectory is in: MT-Pose.txt
    (body pose, `t x y z qw qx qy qz`, epoch seconds) carried through the rig's cam->body
    offset inv(Tcb)[:3, 3]."""
    g = np.loadtxt(f"{d}/MT-Pose.txt")
    t_bc = np.linalg.inv(np.asarray(Tcb, dtype=np.float64))[:3, 3]
    return g[:, 0], g[:, 1:4] + quat_wxyz_to_R(g[:, 4:8]) @ t_bc


def score_sim(d, out, Tcb):
    """The one sim number: whole-trajectory ATE, Sim(3) fitted on every KF (the public
    sets' `evo_ape -vas` protocol). A `-Walk` capture is scored exactly the same way."""
    kf = np.loadtxt(f"{out}/traj_kf_beforeBA.txt")            # TUM, epoch seconds
    gt_t, gt_pos = gt_cam(d, Tcb)                              # epoch seconds as well
    ate = compute_ate(kf[:, 0], kf[:, 1:4], gt_t, gt_pos)
    scale = json.load(open(f"{out}/init_dump.json"))["scale"]
    print("=" * 64)
    print(f"  init scale={scale:.4f}")
    print(f"  ATE [cm], Sim(3) fitted on ALL {ate['n']} KFs (scale {ate['scale']:.4f}):"
          f"  {ate['ate_cm']:6.1f}  final-KF {ate['final_cm']:.1f}")
    print("=" * 64)
    return dict(ate_cm=round(ate["ate_cm"], 1), ate_final_cm=round(ate["final_cm"], 1))


def run_sim(args):
    cfg_path = CONFIG["sim"]
    Tcb = yaml.safe_load(open(cfg_path))["IMU"]["Tcb_np"]
    seqs = sim_sequences()
    if not seqs:
        raise SystemExit(f"no capture with an events.json under {DATA_ROOT}/")
    rows, failed = [], []
    for seq, d in seqs.items():
        events = json.load(open(f"{d}/events.json"))
        # no ride in the capture = the `walk` CONTROL capture, the elevator-free row
        row = dict(seq=seq, route=events.get("route", "?"), walk=not events["legs"])
        rows.append(row)
        print(f"{'=' * 60}\n{seq}  route={row['route']}\n{'=' * 60}")
        out = f"{output_folder}/{seq}"
        os.makedirs(out, exist_ok=True)
        if run_demo_if_needed(demo_cmd(out, cfg_path, d, args, extra=f" --buffer {BUFFER}"),
                              out, args):
            row.update(score_sim(d, out, Tcb))
        else:
            failed.append(seq)

    if len(rows) > 1:
        print("\n" + "=" * 60)
        print(f"{'seq':22s} {'route':7s} {'ATE':>8s} {'final':>7s}")
        for r in rows:
            if "ate_cm" not in r:
                print(f"{r['seq']:22s} {r['route']:7s}  ERROR: no traj")
                continue
            print(f"{r['seq']:22s} {r['route']:7s} {r['ate_cm']:8.1f} {r['ate_final_cm']:7.1f}"
                  + ("   CONTROL (no ride)" if r["walk"] else ""))
        print("=" * 60)
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gsmapping", action="store_true",
                    help="also train the Gaussian map in this run (demo.py's own flag, passed through)")
    ap.add_argument("--offline", action="store_true",
                    help="run the offline post-process after tracking: final BA, and with "
                         "--gsmapping the GS colour refinement "
                         "(default off: the run stops at the online state)")
    ap.add_argument("--force", action="store_true",
                    help="re-run the demo even when the run dir already holds a trajectory "
                         "(its outputs are overwritten in place)")
    args = ap.parse_args()
    failed = []
    for run in (run_real, run_sim):
        failed += run(args) or []

    # a sequence whose demo.py produced no trajectory must not look like a clean run
    if failed:
        raise SystemExit(f"\n{len(failed)} sequence(s) produced no trajectory: "
                         + ", ".join(failed))
