#!/usr/bin/env python3
"""Batch tracking evaluation over the four public datasets.

Usage:  python eval_public_mono.py --dataset euroc
        python eval_public_mono.py --dataset rpng --seqs table_01,table_03
        python eval_public_mono.py --dataset euroc --gsmapping     # + train the Gaussian map
        python eval_public_mono.py --dataset rpng --no-gsmapping   # tracking only, no map
        python eval_public_mono.py --dataset rpng --offline        # + final BA / GS refinement

Gaussian mapping is per dataset (DATASETS[...]["gsmapping"]): on for rpng, utmm and fastlivo,
off for euroc. --gsmapping / --no-gsmapping override that for one call. The map is trained and
saved (the aligned PLY too), not scored: as the base system's eval_*_mono.py, the script reports
the trajectory metrics only.

Where the knobs live: config/<dataset>.yaml owns what differs between datasets (IMU
noise/extrinsics/time offset, imu_late_init_from, pgba.active), vigs/config_defaults.py the
algorithm hyper-parameters shared by all of them, the DATASETS table below the demo.py command
line (buffer, gsmapping and paths -- templates take {root}, {seq} = root/sequence dir, {name}),
and the flags what varies between two runs of the same sequence. Nothing is read from the
environment.

Protocol is the base system's own, unchanged: ONE run per sequence, ATE on the keyframe
trajectory before the final BA, set score = the mean over the sequences.
"""
import argparse
import os

import numpy as np

from vigs.util.compute_recall import compute_recall_from_file
from scripts.align_ply_from_ape_log import align_ply_from_ape_log

# user config
traj_name        = "traj_kf_beforeBA.txt"  # or traj_{kf,full}_{before,after}BA.txt
stride           = 1
recall_thresh_cm = 10

DATASETS = {
    "euroc": dict(
        # Flat layout: data/euroc/MH_01_easy/... (see README)
        root="data/euroc", config="config/euroc.yaml",
        calib="calib/euroc.txt", imagedir="{seq}/mav0/cam0/data",
        imufile="{seq}/mav0/imu0/data.csv", gt="euroc_groundtruth/{name}_sec.txt",
        ply_align=False,
        seqs=["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
              "V1_01_easy", "V1_02_medium", "V1_03_difficult",
              "V2_01_easy", "V2_02_medium", "V2_03_difficult"],
    ),
    "rpng": dict(
        root="data/rpngar", config="config/rpng.yaml",
        calib="calib/rpngar.txt", imagedir="{seq}/rgb",
        imufile="{seq}/imu.txt", gt="{seq}/gt.txt",
        ply_align=True, buffer=700, gsmapping=True,
        seqs=["table_01", "table_02", "table_03", "table_04",
              "table_05", "table_06", "table_07", "table_08"],
    ),
    "utmm": dict(
        root="data/UTMM_Dataset", config="config/utmm.yaml",
        calib="{seq}/intrinsics_ours.txt", imagedir="{seq}/rgb_timestamp",
        imufile="{seq}/imu_ours.txt", gt="{seq}/groundtruth.txt",
        ply_align=True, gsmapping=True,
        seqs=["ego-centric-1", "ego-centric-2", "ego-drive",
              "fast-straight", "slow-straight-1", "slow-straight-2",
              "square-1", "square-2"],
    ),
    "fastlivo": dict(
        root="data/fast-livo2-dataset", config="config/fastlivo.yaml",
        calib="{seq}/intrinsics.txt", imagedir="{seq}/rgb",
        imufile="{seq}/imu.txt", gt="{root}/pgt/{name}.txt",
        ply_align=True, gsmapping=True,
        seqs=["CBD_Building_01", "CBD_Building_02",
              "HKU_Campus", "Retail_Street", "SYSU_01"],
    ),
}


def _fmt(ds, name):
    """Substitutions for the DATASETS path templates."""
    return dict(root=ds["root"], seq=os.path.join(ds["root"], name), name=name)


def build_cmd(ds, name, out, gsmapping, offline):
    fmt = _fmt(ds, name)
    buffer = ds.get("buffer")
    return (f"python demo.py"
            f" --calib {ds['calib'].format(**fmt)}"
            f" --imagedir {ds['imagedir'].format(**fmt)}"
            f" --config {ds['config']}"
            f" --stride {stride}"
            f" --imufile {ds['imufile'].format(**fmt)}"
            f" --output {out}"
            + (f" --buffer {buffer}" if buffer else "")
            + " --undistort"
            + (" --gsmapping" if gsmapping else "")
            + (" --offline" if offline else "")
            + f" > {out}/log.txt 2>&1")   # 2>&1: stderr carries the tracebacks AND the tqdm rate


def score_seq(ds, name, out, gsmapping):
    """Score one finished run dir: evo ATE / scale error / recall on traj_name, plus the aligned
    PLY when the map was trained. Returns the row dict (None when the APE log cannot be parsed).
    Run from the repo root."""
    fmt = _fmt(ds, name)
    gt_file   = ds["gt"].format(**fmt)
    traj_stem = traj_name.split(".")[0]
    log_ape   = f"{out}/log_ape_{traj_stem}.txt"
    os.system(f"evo_ape tum -vas --no_warnings --plot_mode xy"
              f" --save_plot {out}/ape_se3_{traj_stem}.png"
              f" --save_results {out}/ape_results.zip"
              f" {gt_file} {out}/{traj_name}"
              f" > {log_ape}")

    try:
        lines = open(log_ape).readlines()
        ATE   = float([l for l in lines if "rmse"             in l][-1].split("\t")[-1])
        scale = float([l for l in lines if "Scale correction" in l][-1].split(" ")[-1])
    except Exception as e:
        print(f"  [WARN] could not parse APE log: {e}")
        return None

    _, __, recall = compute_recall_from_file(gt_file, f"{out}/{traj_name}", thresh_cm=recall_thresh_cm)
    row = dict(name=name, ate_cm=ATE * 100, scale_err_pct=abs(1 - scale) * 100, recall=recall)
    print(f"  ATE={row['ate_cm']:.2f} cm  scale_err={row['scale_err_pct']:.2f}%  "
          f"recall@{recall_thresh_cm}cm={recall:.2f}%")

    if gsmapping and ds["ply_align"]:   # the PLY only exists when the map was trained
        try:
            if "beforeBA" in traj_name:
                align_ply_from_ape_log(ape_log_path=log_ape,
                                       ply_in=f"{out}/3dgs_before_final.ply",
                                       ply_out=f"{out}/3dgs_before_final_aligned.ply")
            elif "afterBA" in traj_name:
                align_ply_from_ape_log(ape_log_path=log_ape,
                                       ply_in=f"{out}/3dgs_final.ply",
                                       ply_out=f"{out}/3dgs_final_aligned.ply")
        except Exception as e:
            print(f"  [WARN] PLY alignment failed: {e}")
    return row


def print_summary(rows):
    """The CSV + LaTeX lines for a list of score_seq() rows (set score = the mean)."""
    ate_values   = [r["ate_cm"] for r in rows]
    scale_errors = [r["scale_err_pct"] for r in rows]
    recalls      = [r["recall"] for r in rows]

    if ate_values:
        print("\n--- CSV ---")
        print("," + ",".join(f"{v:.2f}" for v in ate_values))
        print("," + ",".join(f"{v:.2f}" for v in scale_errors))
        print("," + ",".join(f"{v:.2f}" for v in recalls))
        print("\n--- LaTeX ---")
        print("ATE [cm] & "          + " & ".join(f"{v:.2f}" for v in ate_values)   + f" & {np.mean(ate_values):.2f} \\\\")
        print("Scale error [\\%] & " + " & ".join(f"{v:.2f}" for v in scale_errors) + f" & {np.mean(scale_errors):.2f} \\\\")
        print(f"Recall@{recall_thresh_cm}cm [\\%] & " + " & ".join(f"{v:.2f}" for v in recalls) + f" & {np.mean(recalls):.2f} \\\\")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--seqs", default=None,
                    help="comma-separated subset of sequences (default: all in the table)")
    # demo.py's own flag, passed through; traj_kf_beforeBA.txt is untouched by the mapping.
    # Unset = the dataset's own setting.
    ap.add_argument("--gsmapping", action=argparse.BooleanOptionalAction, default=None,
                    help="also train the Gaussian map in this run; --no-gsmapping for tracking "
                         "only (default: the dataset's setting -- on for rpng, utmm and "
                         "fastlivo, off for euroc)")
    ap.add_argument("--offline", action="store_true",
                    help="run the offline post-process after tracking: final BA, and with "
                         "the map the GS colour refinement "
                         "(default off: the run stops at the online state)")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    gsmapping = ds.get("gsmapping", False) if args.gsmapping is None else args.gsmapping
    seqs = args.seqs.split(",") if args.seqs else ds["seqs"]
    output_folder = f"outputs/output_{args.dataset}/opensource"
    os.makedirs(output_folder, exist_ok=True)

    rows, failed = [], []
    for name in seqs:
        out = os.path.join(output_folder, name)
        os.makedirs(out, exist_ok=True)
        print(f"{'='*60}\n{name}\n{'='*60}")

        cmd = build_cmd(ds, name, out, gsmapping, args.offline)
        print("RUNNING COMMAND: ", cmd)
        if not os.path.exists(f"{out}/{traj_name}"):
            os.system(cmd)
        if not os.path.exists(f"{out}/{traj_name}"):
            print(f"  [ERROR] {out}/{traj_name} not produced; see {out}/log.txt")
            failed.append(name)
            continue

        row = score_seq(ds, name, out, gsmapping)
        if row is None:
            failed.append(name)
        else:
            rows.append(row)

    print_summary(rows)

    # a sequence that produced no trajectory, or no parsable APE log, must not look like a
    # clean run: the set mean above is then an average over the survivors only.
    if failed:
        raise SystemExit(f"\n{len(failed)}/{len(seqs)} sequence(s) produced no result: "
                         + ", ".join(failed))


if __name__ == "__main__":
    main()
