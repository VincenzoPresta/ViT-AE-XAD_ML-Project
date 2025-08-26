import argparse
import os
import shutil
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from codecarbon import EmissionsTracker

from torchvision.transforms import Resize

from tools.create_dataset import square, square_diff, mvtec, mvtec_only_one, mvtec_only_one_augmented, \
    mvtec_personalized, load_dataset, extract_dataset, mvtec_all_classes, mvtec_ViT

    
#per ora commento, non mi interessa usare i competitor    
'''from run_fcdd import launch as launch_fcdd
from run_deviation import launch as launch_dev'''

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

    if args.ds == 'mnist' or args.ds == 'fmnist':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                intensity=args.i, DATASET=args.ds, seed=args.s)
        
        # converti in tensori float
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        GT_train = torch.tensor(GT_train, dtype=torch.float32)
        GT_test  = torch.tensor(GT_test, dtype=torch.float32)

        # se manca il canale, aggiungilo
        if X_train.ndim == 3:
            X_train = X_train.unsqueeze(1)
            X_test  = X_test.unsqueeze(1)

        # Immagini -> bilinear
        X_train = F.interpolate(X_train, size=(224,224), mode="bilinear")
        X_test  = F.interpolate(X_test,  size=(224,224), mode="bilinear")

        # Maschere -> nearest (per mantenere binarie)
        GT_train = F.interpolate(GT_train, size=(224,224), mode="nearest")
        GT_test  = F.interpolate(GT_test,  size=(224,224), mode="nearest")


        # Conversione finale a numpy
        X_train = X_train.numpy()
        X_test  = X_test.numpy()
        GT_train = GT_train.numpy()
        GT_test  = GT_test.numpy()


        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
        
        #-----DEBUG
        # Salva dataset generato, così non sparisce a fine run -> questo attualmente mi serve per debug
        os.makedirs(data_path, exist_ok=True)
        
        
        print("[DEBUG] Shapes:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        save_path = os.path.join(data_path, f"ad{args.ds}_{X_train.shape[2]}x{X_train.shape[3]}.pt")
        print("[DEBUG] Absolute save path:", os.path.abspath(save_path))

        anom_idx_train = np.where(Y_train == 1)[0][:50]
        norm_idx_train = np.where(Y_train == 0)[0][:50]
        sel_idx_train = np.concatenate([anom_idx_train, norm_idx_train])

        anom_idx_test = np.where(Y_test == 1)[0][:50]
        norm_idx_test = np.where(Y_test == 0)[0][:50]
        sel_idx_test = np.concatenate([anom_idx_test, norm_idx_test])

        torch.save({
            "X_train": X_train[sel_idx_train],
            "Y_train": Y_train[sel_idx_train],
            "X_test": X_test[sel_idx_test],
            "Y_test": Y_test[sel_idx_test],
            "GT_train": GT_train[sel_idx_train],
            "GT_test": GT_test[sel_idx_test],
        }, save_path)

        print(f"Dataset salvato in {data_path}")
        print("[DEBUG] File exists after save?", os.path.exists(save_path))
        
    elif args.ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=dataset, seed=args.s)
        data_path = os.path.join('datasets', args.ds, str(args.c), str(args.s))
        ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
        
    elif args.ds == 'mvtec':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_ViT(args.c, 'datasets/mvtec', args.na, seed=args.s)
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

    #FCDD
    '''htmaps, scores, gtmaps, labels, tot_time = launch_fcdd(data_path, epochs=200, batch_size=16)  # 400
    np.save(open(os.path.join(ret_path, 'fcdd_gt.npy'), 'wb'), gtmaps)
    np.save(open(os.path.join(ret_path, 'fcdd_labels.npy'), 'wb'), labels)
    np.save(open(os.path.join(ret_path, 'fcdd_htmaps.npy'), 'wb'), htmaps)
    np.save(open(os.path.join(ret_path, 'fcdd_scores.npy'), 'wb'), np.array(scores))
    times.append(tot_time)

    emissions = tracker.stop()
    print('EMISSIONS: ', emissions)
    pickle.dump(emissions, open(os.path.join(ret_path, 'emissions_fcdd.pkl'), 'wb'))

    del htmaps, scores
    torch.cuda.empty_cache()'''

    #htmaps, scores, gtmaps, labels, tot_time = launch_dev(dataset_root=data_path, epochs=50)  # 50
    #np.save(open(os.path.join(ret_path, 'deviation_htmaps.npy'), 'wb'), htmaps)
    #np.save(open(os.path.join(ret_path, 'deviation_scores.npy'), 'wb'), np.array(scores))
    #np.save(open(os.path.join(ret_path, 'deviation_gtmaps.npy'), 'wb'), gtmaps)
    #np.save(open(os.path.join(ret_path, 'deviation_labels.npy'), 'wb'), np.array(labels))
    #times.append(tot_time)
    #del htmaps, scores
    #torch.cuda.empty_cache()


    def f(x):
       return 1-x
   
   
    #SHALLOW
    '''heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1000, 16, 32, (28*28) / 25, None, f, 'shallow',
                                                   save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)'''

    #CONV
    '''heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1000, 16, 32, (28*28) / 25, None, f, 'conv',
                                                   save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'wb'), scores)

    times.append(tot_time)
    times = np.array(times)
    np.save(open(os.path.join(ret_path, 'times_competitors.npy'), 'wb'), np.array(times))
    print(times)'''
    
    # ViT
    heatmaps, scores, gtmaps, labels, tot_time = launch_aexad(
        data_path, 
        70,              # epoche 
        16,              # batch size
        32,             # latent dim
        (224*224) / 25, # radius adattato al 224x224, come da paper
        None, 
        f, 
        'vit',         
        save_intermediate=True, 
        save_path=ret_path
    )
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_vit.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_vit.npy'), 'wb'), scores)
    np.save(os.path.join(ret_path, "aexad_labels.npy"), labels)
    np.save(os.path.join(ret_path, "aexad_gt.npy"), gtmaps)

    times.append(tot_time)
    times = np.array(times)
    np.save(open(os.path.join(ret_path, 'times_competitors.npy'), 'wb'), np.array(times))
    print(times)


    shutil.rmtree(data_path)



