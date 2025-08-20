#!/bin/bash

for i in {0..9}
do
  python test_lenet_ae.py -ds mnist -c $i -patr 0.02 -pate 0.1 -s 2 -size 5
done