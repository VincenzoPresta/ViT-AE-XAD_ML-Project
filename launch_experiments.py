import argparse
import os
import shutil
import pickle

import numpy as np
import torch
from codecarbon import EmissionsTracker

from tools.create_dataset import square, square_diff, mvtec, mvtec_only_one, mvtec_only_one_augmented, \
    mvtec_personalized, load_dataset, extract_dataset, mvtec_all_classes
#from run_fcdd import launch as launch_fcdd - questo non funziona
#from run_deviation import launch as launch_dev - idem


from competitors.fcdd.run_fcdd import launch as launch_fcdd
from competitors.deviation.run_deviation import launch as launch_dev

from aexad_script import launch as launch_aexad

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-ac', type=int, help='Considered anomaly type', default=0)
    parser.add_argument('-na', type=int, help='Number of anomalies', default=1)
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-i', help='Modification intesity')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    args = parser.parse_args()

    if args.i != 'rand':
        args.i = float(args.i)

    if args.ds == 'mnist'  or args.ds == 'fmnist':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=args.ds, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
    elif args.ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=dataset, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
    elif args.ds == 'mvtec':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec(args.c, 'datasets/mvtec', args.na, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.s), str(args.na))
        print(ret_path)
    elif args.ds == 'mvtec_o_a':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_only_one(args.c, 'datasets/mvtec', args.na,
                                                                             a_cls=args.ac,
                                                                             seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.ac), str(args.s), str(args.na))
    elif args.ds == 'mvtec_o_a_aug':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_only_one_augmented(args.c, 'datasets/mvtec', args.na,
                                                                             a_cls=args.ac,
                                                                             seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.ac), str(args.s), str(args.na))
    elif args.ds == 'mvtec_our':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test, _, _, files_train, files_test = mvtec_personalized(args.c, 'datasets/mvtec',
                                                                                       n_anom_per_cls=args.na,
                                                                                       seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.ac), str(args.s), str(args.na))
        np.save(open(os.path.join(ret_path, 'files_train_comp.npy'), 'wb'), files_train)
        np.save(open(os.path.join(ret_path, 'files_test_comp.npy'), 'wb'), files_test)
    elif args.ds == 'mvtec_all':
        dataset = 'mvtec'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test, _, _, files_train, files_test = mvtec_all_classes(args.c, 'datasets/mvtec',
                                                                                       n_anom_per_cls=args.na,
                                                                                       seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.ac), str(args.s))
        ret_path = os.path.join('results', 'only_anom', args.ds, str(args.c), str(args.ac), str(args.s), str(args.na))
        if not os.path.exists(ret_path):
            os.makedirs(ret_path)
        np.save(open(os.path.join(ret_path, 'files_train_comp.npy'), 'wb'), files_train)
        np.save(open(os.path.join(ret_path, 'files_test_comp.npy'), 'wb'), files_test)
    elif args.ds == 'hazelnut' or args.ds == 'road_inspection':
        dataset = args.ds
        data_path = os.path.join('datasets', args.ds, 'files')
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = load_dataset(f'datasets/{args.ds}')
        ret_path = os.path.join('results', args.ds)
    elif 'btad' in args.ds:
        dataset = args.ds
        data_path = os.path.join('datasets', args.ds, 'files')
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = extract_dataset(f'datasets/{args.ds}', args.na, seed=args.s)
        ret_path = os.path.join('results', args.ds)

    X_train = X_train.swapaxes(2, 3).swapaxes(1, 2)
    X_test = X_test.swapaxes(2, 3).swapaxes(1, 2)
    GT_train = GT_train.swapaxes(2, 3).swapaxes(1, 2)
    GT_test = GT_test.swapaxes(2, 3).swapaxes(1, 2)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train[Y_train==1])
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train[Y_train==1])
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train[Y_train==1])
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

    if not os.path.exists(ret_path):
        os.makedirs(ret_path)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []
    print(torch.cuda.is_available())
    del X_train, Y_train, X_test, Y_test, GT_train, GT_test

    tracker = EmissionsTracker()
    tracker.start()

    htmaps, scores, gtmaps, labels, tot_time = launch_fcdd(data_path, epochs=200, batch_size=16)  # 400
    np.save(open(os.path.join(ret_path, 'fcdd_gt.npy'), 'wb'), gtmaps)
    np.save(open(os.path.join(ret_path, 'fcdd_labels.npy'), 'wb'), labels)
    np.save(open(os.path.join(ret_path, 'fcdd_htmaps.npy'), 'wb'), htmaps)
    np.save(open(os.path.join(ret_path, 'fcdd_scores.npy'), 'wb'), np.array(scores))
    times.append(tot_time)

    emissions = tracker.stop()
    print('EMISSIONS: ', emissions)
    pickle.dump(emissions, open(os.path.join(ret_path, 'emissions_fcdd.pkl'), 'wb'))

    del htmaps, scores
    torch.cuda.empty_cache()

    #htmaps, scores, gtmaps, labels, tot_time = launch_dev(dataset_root=data_path, epochs=50)  # 50
    #np.save(open(os.path.join(ret_path, 'deviation_htmaps.npy'), 'wb'), htmaps)
    #np.save(open(os.path.join(ret_path, 'deviation_scores.npy'), 'wb'), np.array(scores))
    #np.save(open(os.path.join(ret_path, 'deviation_gtmaps.npy'), 'wb'), gtmaps)
    #np.save(open(os.path.join(ret_path, 'deviation_labels.npy'), 'wb'), np.array(labels))
    #times.append(tot_time)
    #del htmaps, scores
    #torch.cuda.empty_cache()


    #def f(x):
    #    return 1-x

    #heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1000, 16, 32, (28*28) / 25, None, f, 'shallow',
    #                                                save_intermediate=True, save_path=ret_path)
    #np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    #np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)

    #heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1000, 16, 32, (28*28) / 25, None, f, 'conv',
    #                                                save_intermediate=True, save_path=ret_path)
    #np.save(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'wb'), heatmaps)
    #np.save(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'wb'), scores)

    #times.append(tot_time)
    #times = np.array(times)
    #np.save(open(os.path.join(ret_path, 'times_competitors.npy'), 'wb'), np.array(times))
    #print(times)

    shutil.rmtree(data_path)


