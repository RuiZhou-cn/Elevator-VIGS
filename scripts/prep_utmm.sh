#!/usr/bin/env bash
set -e
mkdir -p data
cd data
wget https://cvg-data.inf.ethz.ch/vigs-slam/UTMM_Dataset.zip
unzip UTMM_Dataset.zip
cd ..
