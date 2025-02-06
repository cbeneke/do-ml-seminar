#!/usr/bin/env bash

DATE="$(date +%Y%m%d-%H%M)"
ITERATIONS=${1-5}
export DATASET="low-frequency"
#export DATASET="high-frequency"

## TF
#source ~/.virtualenvs/nif-tf/bin/activate
#for r in $(seq ${ITERATIONS}); do
#  for i in functional ; do
#  #for i in functional upstream ; do
#    for o in adam adabelief ; do
#      save_path="results/${i}/${DATASET}-${o}-${DATE}-${r}"
#      NIF_IMPLEMENTATION="${i}" OPTIMIZER="${o}" python3 src/experiment.py
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
      save_path="results/${i}/${DATASET}-${o}-${DATE}-${r}"
      NIF_IMPLEMENTATION="${i}" OPTIMIZER="${o}" python3 src/experiment.py
      mkdir ${save_path}
      mv saved_weights vis.pdf log loss.pdf ${save_path}
    done
  done
done
deactivate
