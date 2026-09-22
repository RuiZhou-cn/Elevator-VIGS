#!/usr/bin/env bash
# Demo sequence: Campus3 of our Elevator-VIGS dataset -- a walk to the lift, a five-floor
# ride, a walk out on the arrival floor (1456 frames, 142 s, 0.6 GB). Same tree as
# scripts/prep_elevator.sh, so skip this if you already downloaded the full dataset.
exec bash "$(dirname "$0")/prep_elevator.sh" Campus3
