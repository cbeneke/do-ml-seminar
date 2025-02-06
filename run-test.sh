#!/usr/bin/env bash

DATE="$(date +%Y%m%d-%H%M)"
ITERATIONS=${1-5}

## TF
#source ~/.virtualenvs/nif-tf/bin/activate
#for r in $(seq ${ITERATIONS}); do
#  for i in functional upstream ; do
#    for o in adam adabelief ; do
#      save_path="results/${i}/low-frequency-${o}-${DATE}-${r}"
#      NIF_IMPLEMENTATION="${i}" OPTIMIZER="${o}" python3 src/01_simple_1d_wave.py
#      mkdir ${save_path}
#      mv saved_weights vis.pdf log loss.pdf ${save_path}
#    done
#  done
#done
#deactivate

# Torch
source ~/.virtualenvs/nif-torch/bin/activate
for r in $(seq ${ITERATIONS}); do
  for i in pytorch ; do
    for o in adam adabelief ; do
      save_path="results/${i}/low-frequency-${o}-${DATE}-${r}"
      NIF_IMPLEMENTATION="${i}" OPTIMIZER="${o}" python3 src/01_simple_1d_wave.py
      mkdir ${save_path}
      mv saved_weights vis.pdf log loss.pdf ${save_path}
    done
  done
done
deactivate
