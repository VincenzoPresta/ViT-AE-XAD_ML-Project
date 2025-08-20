#!/bin/bash

for i in {0..14}
do
  echo "Executing object type $i"
  aexad_v2_venv/bin/python test_conv_ae.py -ds mvtec_all -c $i -s 40 -i rand -f 1 -net conv_deep_v2 -na 2 -e 500
done

#for i in {8..14}
#do
#  echo "Executing object type $i"
#  aexad_v2_venv/bin/python test_conv_ae.py -ds mvtec -c $i -s 40 -i rand -f 1 -net conv_deep -na 3 -e 500
#done
