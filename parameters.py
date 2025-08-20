import argparse
import os
import shutil
import pickle

import numpy as np
import torch

from tools.create_dataset import square
from run_fcdd import launch as launch_fcdd
from run_deviation import launch as launch_dev
from aexad_script import launch as launch_aexad

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-i', help='Modification intesity', default='rand')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    # Parametri
    parser.add_argument('-ld', type=int, help='Latent space dimension', default=32)
    parser.add_argument('-b', type=int, help='Batch size', default=16)
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
    ret_path = os.path.join('results', 'params', f'{args.ld}_{args.b}', args.ds, str(args.c), str(args.s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    #times = []

    def f(x):
        return 1-x

    # heatmaps, scores, _, _, _ = launch_aexad(data_path, 2000, args.b, args.ld, (28*28) / 25, None, f, 'shallow',
    #                                                 save_intermediate=True, save_path=ret_path)
    # np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    # np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)
    # #times.append(tot_time)
    # del heatmaps, scores
    # torch.cuda.empty_cache()

    heatmaps, scores, _, _, _ = launch_aexad(data_path, 2000,  args.b, args.ld, (28*28) / 25, None, f, 'deep',
                                                    save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_deep.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_deep.npy'), 'wb'), scores)
    #times.append(tot_time)
    del heatmaps, scores
    torch.cuda.empty_cache()

    #np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))
    #print(times)

    shutil.rmtree(data_path)


