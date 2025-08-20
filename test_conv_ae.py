import argparse
import os
import shutil
import pickle

import numpy as np
import torch

from tools.create_dataset import square, square_diff, mvtec, mvtec_only_one, mvtec_only_one_augmented, mvtec_aug, \
    mvtec_personalized, load_dataset, mvtec_all_classes
from aexad_script import launch as launch_aexad

from codecarbon import EmissionsTracker


def compute_weights(Y, GT):
    #weights = np.ones_like(Y)
    tot_pixels = GT[Y == 1].sum(axis=(1, 2))
    #a_max = tot_pixels.max()
    w_anoms = 1 / tot_pixels
    n_norm = np.sum(1 - Y)
    norm_f = n_norm / w_anoms.sum()
    #norm_f = w_anoms * norm_f
    #weights[Y == 1] = (wa / wa.sum()) * (np.sum(1 - Y))
    #return weights
    return float(norm_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-e', type=int, help='Number of epochs', default=2000)
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-ac', type=int, help='Considered anomaly type', default=0)
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-na', type=int, default=1, help='Number of anomalies for anomaly class (only for real ds)')
    parser.add_argument('-i', help='Modification intesity')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    parser.add_argument('-f', type=int, help='Function to use for mapping')
    parser.add_argument('-dp', type=str, help='Dataset path', default=None)
    parser.add_argument('-net', type=str, choices=['shallow', 'deep', 'conv', 'conv_deep', 'conv_deep_v2', 'conv_deep_norm', 'conv_f2'], help='Network to use')
    parser.add_argument('-l', type=str, choices=['aexad', 'mse', 'aexad_norm'], default='aexad', help='Loss to use. Available options are aexad and mse')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--hist', dest='si', action='store_true', help='If specified, saves the weights of the model every 100 epochs')
    args = parser.parse_args()

    if args.i != 'rand' and (args.ds == 'mnist' or args.ds == 'mnist_diff'):
        args.i = float(args.i)

    data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s)) if args.dp is None else args.dp

    if args.ds == 'mnist' or args.ds == 'fmnist':
        dataset = args.ds
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=args.ds, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s)) if args.dp is None else args.dp
        ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.s)) if args.dp is None else \
            os.path.join('results', args.dp[9:])

    elif args.ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=dataset, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s)) if args.dp is None else args.dp
        ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.s)) if args.dp is None else \
            os.path.join('results', args.dp[9:])

    elif args.ds == 'mvtec':
        dataset = args.ds
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec(args.c, 'datasets/mvtec', args.na, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s)) if args.dp is None else args.dp
        ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.s), str(args.na)) if args.dp is None else \
            os.path.join('results', args.dp[9:])

    elif args.ds == 'mvtec_o_a':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_only_one(args.c, 'datasets/mvtec', args.na, a_cls=args.ac,
                                                                             seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac),
                                 str(args.s)) if args.dp is None else args.dp
        ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.ac),
                                str(args.s), str(args.na)) if args.dp is None else \
        os.path.join('results', args.dp[9:])
    elif args.ds == 'mvtec_o_a_aug':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_only_one_augmented(args.c, 'datasets/mvtec', args.na,
                                                                             a_cls=args.ac,
                                                                             seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac),
                                 str(args.s)) if args.dp is None else args.dp
        ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.ac),
                                str(args.s), str(args.na)) if args.dp is None else \
        os.path.join('results', args.dp[9:])
    elif args.ds == 'mvtec_our':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test, _, _, files_train, files_test = mvtec_personalized(args.c, 'datasets/mvtec',
                                                                                       n_anom_per_cls=args.na,
                                                                                       seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.ac), str(args.s), str(args.na))
        if not os.path.exists(ret_path):
            os.makedirs(ret_path)
        np.save(open(os.path.join(ret_path, 'files_train_aexad.npy'), 'wb'), files_train)
        np.save(open(os.path.join(ret_path, 'files_test_aexad.npy'), 'wb'), files_test)

    elif args.ds == 'mvtec_all':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test, _, _, files_train, files_test = mvtec_all_classes(args.c, 'datasets/mvtec',
                                                                                       n_anom_per_cls=args.na,
                                                                                       seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', 'only_anom', args.ds, str(args.c), str(args.s), str(args.na))
        if not os.path.exists(ret_path):
            os.makedirs(ret_path)
        np.save(open(os.path.join(ret_path, 'files_train_aexad.npy'), 'wb'), files_train)
        np.save(open(os.path.join(ret_path, 'files_test_aexad.npy'), 'wb'), files_test)

    elif args.ds == 'hazelnut' or 'road_inspection':
        dataset = args.ds
        data_path = os.path.join('datasets', args.ds, 'files')
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = load_dataset(f'datasets/{args.ds}')
        ret_path = os.path.join('results', args.ds)

    elif args.ds == 'aebad' and args.dp is not None:
        dataset = args.ds
        X_train = np.load(open(os.path.join(data_path, 'X_train.npy'), 'rb'))
        X_test = np.load(open(os.path.join(data_path, 'X_test.npy'), 'rb'))
        Y_train = np.load(open(os.path.join(data_path, 'Y_train.npy'), 'rb'))
        Y_test = np.load(open(os.path.join(data_path, 'Y_test.npy'), 'rb'))
        GT_train = np.load(open(os.path.join(data_path, 'GT_train.npy'), 'rb'))
        GT_test = np.load(open(os.path.join(data_path, 'GT_test.npy'), 'rb'))
        data_path = args.dp
        ret_path = os.path.join('results', args.dp[9:])


    if args.dp is None:
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if args.l == 'mse':
            # If we use the standard autoencoder the train set have no anomalies
            np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train[Y_train==0])
            np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train[Y_train==0])
            np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train[Y_train==0])
        else:
            np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train[Y_train==1])
            np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train[Y_train==1])
            np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train[Y_train==1])

        np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
        # f_o funzione vecchia


    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args_aexad'), 'wb'))

    if args.f == 0:
        # f_0
        def f(x):
           return 1 - x

    elif args.f == 1:
        #f_1
        def f(x):
            return torch.where(x >= 0.5, 0.0, 1.0)

    elif args.f == 2:
        def f(x):
            return x + 2

    elif args.f == 3:
        print('f3')
        def f(x):
            return torch.where(x >= 0.5, x-0.5, x+0.5)

    elif args.f == 4:
        print('f4')
        def f(x):
            return torch.where(x >= 0.5, x-2., x+2.)

    elif args.f == 5:
        print('f4')
        def f(x):
            return torch.where(x >= 0, x-1.5, x+1.5)


    if args.l == 'aexad_norm':
        lambda_s = compute_weights(Y_train, GT_train)
    else:
        lambda_s = None


    # TODO: salvare rete per prossimi test

    tracker = EmissionsTracker()
    tracker.start()

    heatmaps, scores, gtmaps, labels, time = launch_aexad(data_path, args.e, 16, 64, None, lambda_s, f, args.net,
                                                    save_intermediate=args.si, save_path=ret_path, use_cuda=args.cuda,
                                                dataset=dataset, loss=args.l) # 2000
    emissions = tracker.stop()
    print('EMISSIONS: ', emissions)
    pickle.dump(emissions, open(os.path.join(ret_path, 'emissions_aexad.pkl'), 'wb'))

    np.save(open(os.path.join(ret_path, f'{args.l}_htmaps_{args.net}.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, f'{args.l}_scores_{args.net}.npy'), 'wb'), scores)
    np.save(open(os.path.join(ret_path, 'aexad_gtmaps_norm.npy'), 'wb'), gtmaps)
    np.save(open(os.path.join(ret_path, 'aexad_labels_norm.npy'), 'wb'), np.array(labels))

    np.save(open(os.path.join(ret_path, f'{args.l}_times_{args.net}.npy'), 'wb'), np.array([time]))

    shutil.rmtree(data_path)