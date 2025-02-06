#!/usr/bin/env bash

for i in functional upstream pytorch ; do
  for data in high-frequency low-frequency ; do
    for optim in adabelief adam ; do
      cat $i/$data-$optim-*/log >/dev/null || continue

      best="$(cat $i/$data-$optim-*/log |cut -d= -f2 |cut -d, -f1 |sort -g | head -n1)"
      for d in $i/$data-$optim-* ; do
        grep $best $d/log && echo "${d}: $best" > $i/$data-$optim.best
      done
    done
  done
done
