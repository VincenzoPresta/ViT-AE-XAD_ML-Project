#!/bin/bash

for i in {0..9}
do
  python lauch_experiments.py -ds mnist -c $i -patr 0.02 -pate 0.1 -s 38 -size 5 -i rand
done