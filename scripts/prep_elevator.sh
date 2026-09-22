#!/usr/bin/env bash
set -e
# Elevator-VIGS dataset (https://huggingface.co/datasets/Rui5125/Elevator-VIGS): 16 handheld
# recordings (real/) and 16 Isaac Sim captures (sim/), named as in the paper, one <Sequence>.zip
# each (camera, IMU, reference trajectory, labels; 42 GB in total) plus one <Sequence>_lidar.zip
# that this code does not read. The zips land in data/Elevator-VIGS/zips/ and unpack into
# data/Elevator-VIGS/{real,sim}/<Sequence>/, the tree eval_elevator_mono.py and demo.py read.
# Needs the `hf` CLI (huggingface_hub, in the conda env); a re-run resumes the download.
#   bash scripts/prep_elevator.sh                    # every sequence
#   bash scripts/prep_elevator.sh Campus3 Mall-E     # only these
REPO=Rui5125/Elevator-VIGS
DIR=data/Elevator-VIGS
if [ $# -gt 0 ]; then
  INCLUDE=(); for s in "$@"; do INCLUDE+=(--include "*/$s.zip"); done
else
  INCLUDE=(--include "*/*.zip" --exclude "*_lidar.zip")
fi
hf download "$REPO" --repo-type dataset --local-dir "$DIR/zips" "${INCLUDE[@]}"
for z in "$DIR"/zips/*/*.zip; do
  unzip -q -n "$z" -d "$DIR"
done
