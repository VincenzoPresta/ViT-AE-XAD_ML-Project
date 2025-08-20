#!/bin/bash

for i in {0..9}
do
  for ld in 64 128 256;
  do
    python parameters.py -ds fmnist -c $i -patr 0.02 -pate 0.1 -s 29 -size 5 -ld $ld
  done
done