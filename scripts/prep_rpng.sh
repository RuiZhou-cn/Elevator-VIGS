#!/usr/bin/env bash
set -e
mkdir -p data
cd data
wget https://cvg-data.inf.ethz.ch/vigs-slam/rpngar.zip
unzip rpngar.zip
cd ..
