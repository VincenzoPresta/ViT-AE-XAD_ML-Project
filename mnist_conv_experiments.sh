#!/bin/bash

for i in {0..9}
do
  python test_conv_ae.py -ds mnist -c $i -patr 0.02 -pate 0.1 -s 90 -size 5 -i rand -f 1 -net conv
done

