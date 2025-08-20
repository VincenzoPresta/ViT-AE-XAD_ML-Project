for i in {0..14}
#for i in {2..2}
do
  echo "Executing object type $i"
  PYTHONPATH=".:./competitors/deviation:./competitors/fcdd" aexad_v2_venv/bin/python launch_experiments.py -ds mvtec_all -c $i -s 40 -i rand -na 2
done
