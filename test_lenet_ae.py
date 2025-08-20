import argparse
import os
import shutil

import numpy as np
import torch

from tools.create_dataset import square
from aexad_script import launch as launch_aexad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-i', help='Modification intesity', default='rand')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
        square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
               intensity=args.i, DATASET=args.ds, seed=args.s)

    data_path = os.path.join('test_data', args.ds, str(args.c), str(args.s))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)


    def f(x):
        return 1 - x

    times = []

    heatmaps, scores, _, _, _ = launch_aexad(data_path, 2000, 16, 32, (28 * 28) / 25, None, f, 'lenet',
                                                    save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_lenet.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_lenet.npy'), 'wb'), scores)


    shutil.rmtree(data_path)


