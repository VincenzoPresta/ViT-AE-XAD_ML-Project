import os

import PIL.Image as Image
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from math import ceil


def extract_dataset(path, n_anom_per_cls, seed=None):
    '''
    Method to arrange the data. Train and test images are supposed to be stored in two different folders.
    The train folder consists of only normal images stored in the 'good' subfolder. The test folder contains normal data
    in the 'good' subfolder and anomalous data into different subfolders representing different types of anomalies.
    A number of anomalies equals to n_anom_per_cls is taken from the test folder and included into the training set.
    :param path: str, path in which the dataset is stored
    :param n_anom_per_cls: int, number of anomalies to be included into the training set
    :param seed: seed to use for reproducibility, if None a random seed is selected, defaults to None
    '''
    np.random.seed(seed=seed)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(path, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(path, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(path, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            print(file)
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

    print('GT ', len(GT_train))
    X_train = np.array(X_train).astype(np.uint8)

    X_test = np.array(X_test).astype(np.uint8)

    print(X_train.shape)
    print(X_test.shape)

    GT_train = np.array(GT_train)

    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    print(f'Training anomalies: {Y_train.sum()}')
    print(f'Training anomalies: {Y_test.sum()}')

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def load_dataset(path):
    X_train = []
    X_test = []
    Y_train = []
    GT_train = []
    GT_test = []
    Y_test = []

    f_path = os.path.join(path, 'train', 'good')
    files_tr = os.listdir(f_path)
    for file in files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            image = np.array(img)
            gt = np.zeros_like(image, dtype=np.uint8)
            X_train.append(image)
            GT_train.append(gt)
            Y_train.append(0)

            ## Augment data
            ## Rotate 90
            #X_train.append(np.array(img.transpose(Image.ROTATE_90)))
            #GT_train.append(gt)
            #Y_train.append(0)
            ## Rotate 180
            #X_train.append(np.array(img.transpose(Image.ROTATE_180)))
            #GT_train.append(gt)
            #Y_train.append(0)
            ## Rotate 270
            #X_train.append(np.array(img.transpose(Image.ROTATE_270)))
            #GT_train.append(gt)
            #Y_train.append(0)

            ## Simmetria sulla diagonale principale
            #X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
            #GT_train.append(gt)
            #Y_train.append(0)
            ## Simmetria sulla diagonale secondaria
            #X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
            #GT_train.append(gt)
            #Y_train.append(0)
            ## Horizontal flip
            #X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
            #GT_train.append(gt)
            #Y_train.append(0)
            ## Vertical flip
            #X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
            #GT_train.append(gt)
            #Y_train.append(0)

    f_path = os.path.join(path, 'train', 'crack')
    gt_path = os.path.join(path, 'train', 'gt')
    files_tr = os.listdir(f_path)
    for file in files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            image = np.array(img)
            X_train.append(image)
            gt = Image.open(os.path.join(gt_path, file)).convert('RGB')
            GT_train.append(np.array(gt))
            Y_train.append(1)

            ## Augment data
            ## Rotate 90
            #X_train.append(np.array(img.transpose(Image.ROTATE_90)))
            #GT_train.append(np.array(gt.transpose(Image.ROTATE_90)))
            #Y_train.append(1)
            ## Rotate 180
            #X_train.append(np.array(img.transpose(Image.ROTATE_180)))
            #GT_train.append(np.array(gt.transpose(Image.ROTATE_180)))
            #Y_train.append(1)
            ## Rotate 270
            #X_train.append(np.array(img.transpose(Image.ROTATE_270)))
            #GT_train.append(np.array(gt.transpose(Image.ROTATE_270)))
            #Y_train.append(1)

            ## Simmetria sulla diagonale principale
            #X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
            #GT_train.append(np.array(gt.transpose(Image.TRANSPOSE)))
            #Y_train.append(1)
            ## Simmetria sulla diagonale secondaria
            #X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
            #GT_train.append(np.array(gt.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
            #Y_train.append(1)
            ## Horizontal flip
            #X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
            #GT_train.append(np.array(gt.transpose(Image.FLIP_TOP_BOTTOM)))
            #Y_train.append(1)
            ## Vertical flip
            #X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
            #GT_train.append(np.array(gt.transpose(Image.FLIP_LEFT_RIGHT)))
            #Y_train.append(1)

    f_path = os.path.join(path, 'test', 'good')
    files_te = os.listdir(f_path)
    for file in files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))
            Y_test.append(0)

    f_path = os.path.join(path, 'test', 'crack')
    gt_path = os.path.join(path, 'test', 'gt')
    files_te = os.listdir(f_path)
    for file in files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.array(Image.open(os.path.join(gt_path, file)).convert('RGB')))
            Y_test.append(1)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    GT_train = np.array(GT_train)
    GT_test = np.array(GT_test)
    GT_train = np.where(GT_train>(255*0.5), 255, 0).astype(np.uint8)
    GT_test = np.where(GT_test>(255*0.5), 255, 0).astype(np.uint8)

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test

def extract_dataset(path, n_anom_per_cls, seed=None):
    '''
    Method to arrange the data. Train and test images are supposed to be stored in two different folders.
    The train folder consists of only normal images stored in the 'good' subfolder. The test folder contains normal data
    in the 'good' subfolder and anomalous data into different subfolders representing different types of anomalies.
    A number of anomalies equals to n_anom_per_cls is taken from the test folder and included into the training set.
    :param path: str, path in which the dataset is stored
    :param n_anom_per_cls: int, number of anomalies to be included into the training set
    :param seed: seed to use for reproducibility, if None a random seed is selected, defaults to None
    '''
    np.random.seed(seed=seed)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(path, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(path, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(path, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            print(file)
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

    print('GT ', len(GT_train))
    X_train = np.array(X_train).astype(np.uint8)

    X_test = np.array(X_test).astype(np.uint8)

    print(X_train.shape)
    print(X_test.shape)

    GT_train = np.array(GT_train)

    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    print(f'Training anomalies: {Y_train.sum()}')
    print(f'Test anomalies: {Y_test.sum()}')

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def extract_dataset_btad(base_path, n_anom_per_cls, seed=None, class_id=0):
    """
    Loader specifico per BTAD: gestisce le sottocartelle 01, 02, 03
    con struttura:
      train/ok, test/ok, test/ko, ground_truth/ko
    """
    np.random.seed(seed)

    # Mappa classi 0,1,2 -> "01","02","03"
    class_map = {0: "01", 1: "02", 2: "03"}
    cls = class_map[class_id]

    path = os.path.join(base_path, cls)

    X_train, X_test, GT_train, GT_test = [], [], [], []

    # --- Normal train ---
    f_path = os.path.join(path, 'train', 'ok')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if file.lower().endswith(("png","jpg","npy","bmp")):
            img = Image.open(os.path.join(f_path, file)).convert('RGB').resize((224,224))
            img = np.array(img, dtype=np.uint8)
            X_train.append(img)
            GT_train.append(np.zeros((1,224,224), dtype=np.uint8))  # maschere vuote

    # --- Normal test ---
    f_path = os.path.join(path, 'test', 'ok')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if file.lower().endswith(("png","jpg","npy","bmp")):
            img = Image.open(os.path.join(f_path, file)).convert('RGB').resize((224,224))
            img = np.array(img, dtype=np.uint8)
            X_test.append(img)
            GT_test.append(np.zeros((1,224,224), dtype=np.uint8))

    # --- Anomalies (ko) ---
    f_path = os.path.join(path, 'test', 'ko')
    anomal_files = os.listdir(f_path)
    idxs = np.random.permutation(len(anomal_files))

    # --- Train anomalies ---
    for file in np.array(anomal_files)[idxs[:n_anom_per_cls]]:
        if file.lower().endswith(("png","jpg","npy","bmp")):
            # Immagine
            img = Image.open(os.path.join(f_path, file)).convert('RGB').resize((224,224))
            img = np.array(img, dtype=np.uint8)
            X_train.append(img)

            # Maschera
            mask_path = os.path.join(path, 'ground_truth', 'ko', file).replace(f'.{file.split(".")[-1]}', '.png')
            mask = Image.open(mask_path).convert("L").resize((224,224))
            mask = np.array(mask, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=0)   # (1,224,224)
            GT_train.append(mask)

    # --- Test anomalies ---
    for file in np.array(anomal_files)[idxs[n_anom_per_cls:]]:
        if file.lower().endswith(("png","jpg","npy","bmp")):
            # Immagine
            img = Image.open(os.path.join(f_path, file)).convert('RGB').resize((224,224))
            img = np.array(img, dtype=np.uint8)
            X_test.append(img)

            # Maschera
            mask_path = os.path.join(path, 'ground_truth', 'ko', file).replace(f'.{file.split(".")[-1]}', '.png')
            mask = Image.open(mask_path).convert("L").resize((224,224))
            mask = np.array(mask, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=0)
            GT_test.append(mask)

    # --- Convert to arrays ---
    X_train = np.array(X_train).astype(np.uint8)
    X_test  = np.array(X_test).astype(np.uint8)
    GT_train = np.array(GT_train)
    GT_test  = np.array(GT_test)

    # --- Labels ---
    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr):] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te):] = 1

    print(f"BTAD-{cls} | Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Training anomalies: {Y_train.sum()}, Test anomalies: {Y_test.sum()}")
    print(f"[DEBUG] BTAD loader attivato: class_id={class_id}, train={len(X_train)}, test={len(X_test)}")

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test




def square(dig,perc_anom_train = 0.2,perc_anom_test = 0.2,size = 5,intensity = 'rand',DATASET = 'mnist', seed=None):
    '''
    :param dig: Selected dataset class
    :param perc_anom_train: Anomalies percentage in th training set (# of anomalies)/(# of sample in the training set)
    :param perc_anom_test: Anomalies percentage in th test set (# of anomalies)/(# of sample in the test set)
    :param size: Dimension of the square
    :param intensity: Pixel value for anomalous square
    :param DATASET: Dataset to use, possible choices: mnist, fmnist and cifar
    :return: X_train, Y_train, X_test, Y_test, GT_train, GT_test
    '''

    np.random.seed(seed=seed)

    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.copy()
        x_test = x_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    width,height = x_train.shape[1:3]
    edge = size//2

    num_anomalies_train = int(perc_anom_train*(np.where(y_train==dig)[0].size))


    id_an_train = np.random.choice(np.where(y_train==dig)[0],num_anomalies_train, replace=False)
    GT_train = np.zeros((x_train.shape[0],width,height))

    pos_intensities = [0.2, 0.4, 0.6, 0.8]

    for id in id_an_train:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)
        if intensity == 'rand':
            #intens = np.random.randint(0,255,3)/255.
            c = np.random.randint(0, 4)
            intens = np.full(3, fill_value=pos_intensities[c])
        if len(x_train.shape) == 4:
            x_train[id, center_x - edge:center_x+edge+1,center_y-edge:center_y+edge+1,0] = intens[0]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]

        GT_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1

    num_anomalies_test = int(perc_anom_test*(np.where(y_test==dig)[0].size))

    id_an_test = np.random.choice(np.where(y_test == dig)[0], num_anomalies_test, replace=False)
    GT_test = np.zeros((x_test.shape[0], width, height))


    for id in id_an_test:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        if intensity == 'rand':
            #intens = np.random.randint(0, 255, 3) / 255.
            c = np.random.randint(0, 4)
            intens = np.full(3, fill_value=pos_intensities[c])

        if len(x_test.shape) == 4:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 0] = intens[0]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]


        GT_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1
    id_train = np.where(y_train == dig)
    id_test = np.where(y_test == dig)



    X_train = x_train[id_train]
    X_test = x_test[id_test]
    GT_train = GT_train[id_train]
    GT_test = GT_test[id_test]
    y_train = y_train.copy()
    y_train[id_train] = 0
    y_train[id_an_train] = 1
    Y_train = y_train[id_train]
    y_test = y_test.copy()
    y_test[id_test] = 0
    y_test[id_an_test] = 1
    Y_test = y_test[id_test]


    if len(X_test.shape) < 4:
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.swapaxes(X_test, 1, 2)

    GT_train = GT_train.reshape(GT_train.shape[0], 1, GT_train.shape[1], GT_train.shape[2])
    GT_test = GT_test.reshape(GT_test.shape[0], 1, GT_test.shape[1], GT_test.shape[2])


    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def square_diff(dig,perc_anom_train = 0.2,perc_anom_test = 0.2,size = 5,intensity = 0.2,DATASET = 'mnist', seed=None):
    '''
    :param dig: Selected dataset class
    :param perc_anom_train: Anomalies percentage in th training set (# of anomalies)/(# of sample in the training set)
    :param perc_anom_test: Anomalies percentage in th test set (# of anomalies)/(# of sample in the test set)
    :param size: Dimension of the square
    :param intensity: Pixel value for anomalous square
    :param DATASET: Dataset to use, possible choices: mnist, fmnist and cifar
    :return: X_train, Y_train, X_test, Y_test, GT_train, GT_test
    '''

    np.random.seed(seed=seed)

    if DATASET == 'mnist' or DATASET == 'mnist_diff':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    width,height = x_train.shape[1:3]
    edge = size//2

    num_anomalies_train = int(perc_anom_train*(np.where(y_train==dig)[0].size))


    id_an_train = np.random.choice(np.where(y_train==dig)[0],num_anomalies_train, replace=False)
    GT_train = np.zeros((x_train.shape[0],width,height))


    for id in id_an_train:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = \
            np.where(x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity <=1,
                     x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity,
                     x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] - intensity)

        GT_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1

    num_anomalies_test = int(perc_anom_test*(np.where(y_test==dig)[0].size))

    id_an_test = np.random.choice(np.where(y_test == dig)[0], num_anomalies_test, replace=False)
    GT_test = np.zeros((x_test.shape[0], width, height))


    for id in id_an_test:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = \
            np.where(
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity <= 1,
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity,
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] - intensity)


        GT_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1
    id_train = np.where(y_train == dig)
    id_test = np.where(y_test == dig)



    X_train = x_train[id_train]
    X_test = x_test[id_test]
    GT_train = GT_train[id_train]
    GT_test = GT_test[id_test]
    y_train[id_train] = 0
    y_train[id_an_train] = 1
    Y_train = y_train[id_train]
    y_test[id_test] = 0
    y_test[id_an_test] = 1
    Y_test = y_test[id_test]


    if len(X_test.shape) < 4:
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.swapaxes(X_test, 1, 2)

    GT_train = GT_train.reshape(GT_train.shape[0], 1, GT_train.shape[1], GT_train.shape[2])
    GT_test = GT_test.reshape(GT_test.shape[0], 1, GT_test.shape[1], GT_test.shape[2])


    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def mvtec(cl, path, n_anom_per_cls, seed=None):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = sorted(os.listdir(f_path))
    for file in normal_files_tr:
        if file.lower().endswith(("png","jpg","npy")):
            image = Image.open(os.path.join(f_path, file)).convert("RGB").resize((224,224))
            image = np.array(image, dtype=np.uint8)
            X_train.append(image)
            GT_train.append(np.zeros((1,224,224), dtype=np.uint8))  # maschere vuote

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = sorted(os.listdir(f_path))
    for file in normal_files_te:
        if file.lower().endswith(("png","jpg","npy")):
            image = Image.open(os.path.join(f_path, file)).convert("RGB").resize((224,224))
            image = np.array(image, dtype=np.uint8)
            X_test.append(image)
            GT_test.append(np.zeros((1,224,224), dtype=np.uint8))  # maschere vuote
            
    print("DEBUG GT_train shapes:", [gt.shape for gt in GT_train[:5]])

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = sorted(os.listdir(outlier_data_dir))
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = sorted(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for i in idxs[: n_anom_per_cls]: #for file in outlier_file[idxs[: n_anom_per_cls]]:
            file = outlier_file[i]
            if file.lower().endswith(("png","jpg","npy")):
                img_path = os.path.join(root, 'test', cl_a, file)
                image = Image.open(img_path).convert("RGB").resize((224,224))
                image = np.array(image, dtype=np.uint8)
                X_train.append(image)

                # Train mask
                mask_path = os.path.join(root, 'ground_truth', cl_a, file).replace(".png","_mask.png")
                mask = Image.open(mask_path).convert("L").resize((224,224))
                mask = np.array(mask, dtype=np.uint8)
                mask = np.expand_dims(mask, axis=0)  # (1,224,224)
                GT_train.append(mask)

        # Test
        for i in idxs[: n_anom_per_cls]: #for file in outlier_file[idxs[: n_anom_per_cls]]:
            if file.lower().endswith(("png","jpg","npy")):
                img_path = os.path.join(root, 'test', cl_a, file)
                image = Image.open(img_path).convert("RGB").resize((224,224))
                image = np.array(image, dtype=np.uint8)
                X_test.append(image)

                # Test mask
                mask_path = os.path.join(root, 'ground_truth', cl_a, file).replace(".png","_mask.png")
                mask = Image.open(mask_path).convert("L").resize((224,224))
                mask = np.array(mask, dtype=np.uint8)
                mask = np.expand_dims(mask, axis=0)
                GT_test.append(mask)



    X_train = np.array(X_train).astype(np.uint8)# / 255.0).astype(np.float32)
    #X_train = np.swapaxes(X_train, 2, 3)
    #X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.array(X_test).astype(np.uint8)
    #X_test = np.swapaxes(X_test, 2, 3)
    #X_test = np.swapaxes(X_test, 1, 2)


    GT_train = np.array(GT_train)#.astype(np.uint8)
    #GT_train = np.swapaxes(GT_train, 2, 3)
    #GT_train = np.swapaxes(GT_train, 1, 2)

    GT_test = np.array(GT_test)#.astype(np.uint8)
    #GT_test = np.swapaxes(GT_test, 2, 3)
    #GT_test = np.swapaxes(GT_test, 1, 2)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1
    
    print("DEBUG X_train:", np.array(X_train).shape)
    print("DEBUG GT_train:", np.array(GT_train).shape)
    print("DEBUG X_test:", np.array(X_test).shape)
    print("DEBUG GT_test:", np.array(GT_test).shape)


    return X_train, Y_train, X_test, Y_test, GT_train, GT_test

def mvtec_all_classes(cl, path, n_anom_per_cls, n_norm=None, seed=None, augmentation=False):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    normal_files_tr.sort()

    files_train = []
    files_test = []

    if n_norm is None:
        n_norm = len(normal_files_tr)

    inserted = 0
    i = 0
    while inserted < n_norm and i < len(normal_files_tr):
        file = normal_files_tr[i]
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            image = np.array(img)
            gt = np.zeros_like(image)
            X_train.append(np.array(image))
            GT_train.append(gt)
            inserted += 1

            if augmentation:
                # Augment data
                # Rotate 90
                X_train.append(np.array(img.transpose(Image.ROTATE_90)))
                GT_train.append(gt)
                # Rotate 180
                X_train.append(np.array(img.transpose(Image.ROTATE_180)))
                GT_train.append(gt)
                # Rotate 270
                X_train.append(np.array(img.transpose(Image.ROTATE_270)))
                GT_train.append(gt)

                # Simmetria sulla diagonale principale
                X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
                GT_train.append(gt)
                # Simmetria sulla diagonale secondaria
                X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                GT_train.append(gt)
                # Horizontal flip
                X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
                GT_train.append(gt)
                # Vertical flip
                X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
                GT_train.append(gt)

            files_train.append(os.path.join(f_path, file))
        i += 1

    # Add normal data to test set
    # Eventually remaining normal files
    while i < len(normal_files_tr):
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))
            files_test.append(os.path.join(f_path, file))
        i += 1

    out_labels = []

    # Normal Test files
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    normal_files_te.sort()

    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))
            out_labels.append('good')
            files_test.append(os.path.join(f_path, file))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    outlier_classes.sort()
    outlier_classes.remove('good')

    for cl_a in outlier_classes:
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl_a))
        outlier_file.sort()
        outlier_file = np.array(outlier_file)
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                img = Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')
                gt = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')
                X_train.append(np.array(img))
                GT_train.append(np.array(gt))

                if augmentation:
                    # Augment data
                    # Rotate 90
                    X_train.append(np.array(img.transpose(Image.ROTATE_90)))
                    GT_train.append(np.array(gt.transpose(Image.ROTATE_90)))
                    # Rotate 180
                    X_train.append(np.array(img.transpose(Image.ROTATE_180)))
                    GT_train.append(np.array(gt.transpose(Image.ROTATE_180)))
                    # Rotate 270
                    X_train.append(np.array(img.transpose(Image.ROTATE_270)))
                    GT_train.append(np.array(gt.transpose(Image.ROTATE_270)))

                    # Simmetria sulla diagonale principale
                    X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
                    GT_train.append(np.array(gt.transpose(Image.TRANSPOSE)))
                    # Simmetria sulla diagonale secondaria
                    X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                    GT_train.append(np.array(gt.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                    # Horizontal flip
                    X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
                    GT_train.append(np.array(gt.transpose(Image.FLIP_TOP_BOTTOM)))
                    # Vertical flip
                    X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
                    GT_train.append(np.array(gt.transpose(Image.FLIP_LEFT_RIGHT)))

                files_train.append(os.path.join(root, 'test/' + cl_a + '/' + file))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                out_labels.append(cl_a)
                files_test.append(os.path.join(root, 'test/' + cl_a + '/' + file))


    X_train = np.array(X_train).astype(np.uint8)
    X_test = np.array(X_test).astype(np.uint8)
    GT_train = np.array(GT_train)
    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test, np.array(out_labels), outlier_classes, np.array(files_train), np.array(files_test)


def mvtec_personalized(cl, path, n_anom_per_cls, n_norm=None, seed=None):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    normal_files_tr.sort()

    files_train = []
    files_test = []

    if n_norm is None:
        n_norm = len(normal_files_tr)

    inserted = 0
    i = 0
    while inserted < n_norm and i < len(normal_files_tr):
        file = normal_files_tr[i]
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            image = np.array(img)
            gt = np.zeros_like(image)
            X_train.append(np.array(image))
            GT_train.append(gt)
            inserted += 1

            # Augment data
            # Rotate 90
            X_train.append(np.array(img.transpose(Image.ROTATE_90)))
            GT_train.append(gt)
            # Rotate 180
            X_train.append(np.array(img.transpose(Image.ROTATE_180)))
            GT_train.append(gt)
            # Rotate 270
            X_train.append(np.array(img.transpose(Image.ROTATE_270)))
            GT_train.append(gt)

            # Simmetria sulla diagonale principale
            X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
            GT_train.append(gt)
            # Simmetria sulla diagonale secondaria
            X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
            GT_train.append(gt)
            # Horizontal flip
            X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
            GT_train.append(gt)
            # Vertical flip
            X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
            GT_train.append(gt)

            files_train.append(os.path.join(f_path, file))
        i += 1

    # Add normal data to test set
    # Eventually remaining normal files
    while i < len(normal_files_tr):
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))
            files_test.append(os.path.join(f_path, file))
        i += 1

    out_labels = []

    # Normal Test files
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    normal_files_te.sort()

    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))
            out_labels.append('good')
            files_test.append(os.path.join(f_path, file))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    outlier_classes.sort()
    outlier_classes.remove('good')

    # Randomly permute anomalous classes
    outlier_classes = np.random.permutation(outlier_classes)
    n_anom_classes = ceil(len(outlier_classes)/2)


    for cl_a in outlier_classes[:n_anom_classes]:
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl_a))
        outlier_file.sort()
        outlier_file = np.array(outlier_file)
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                img = Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')
                gt = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')
                X_train.append(np.array(img))
                GT_train.append(np.array(gt))

                # Augment data
                # Rotate 90
                X_train.append(np.array(img.transpose(Image.ROTATE_90)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_90)))
                # Rotate 180
                X_train.append(np.array(img.transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_180)))
                # Rotate 270
                X_train.append(np.array(img.transpose(Image.ROTATE_270)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_270)))

                # Simmetria sulla diagonale principale
                X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE)))
                # Simmetria sulla diagonale secondaria
                X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                # Horizontal flip
                X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_TOP_BOTTOM)))
                # Vertical flip
                X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_LEFT_RIGHT)))

                files_train.append(os.path.join(root, 'test/' + cl_a + '/' + file))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                out_labels.append(cl_a)
                files_test.append(os.path.join(root, 'test/' + cl_a + '/' + file))


    for cl_a in outlier_classes[n_anom_classes:]:
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl_a))
        outlier_file.sort()
        outlier_file = np.array(outlier_file)

        # Test
        for file in outlier_file:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                out_labels.append(cl_a)
                files_test.append(os.path.join(root, 'test/' + cl_a + '/' + file))


    X_train = np.array(X_train).astype(np.uint8)
    X_test = np.array(X_test).astype(np.uint8)
    GT_train = np.array(GT_train)
    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test, np.array(out_labels), outlier_classes[n_anom_classes:], np.array(files_train), np.array(files_test)



def mvtec_aug(cl, path, n_anom_per_cls, seed=None):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                img = Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')
                gt = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')
                X_train.append(np.array(img))
                GT_train.append(np.array(gt))

                # Augment data
                # Rotate 90
                X_train.append(np.array(img.transpose(Image.ROTATE_90)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_90)))
                # Rotate 180
                X_train.append(np.array(img.transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_180)))
                # Rotate 270
                X_train.append(np.array(img.transpose(Image.ROTATE_270)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_270)))

                # Simmetria sulla diagonale principale
                X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE)))
                # Simmetria sulla diagonale secondaria
                X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                # Horizontal flip
                X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_TOP_BOTTOM)))
                # Vertical flip
                X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_LEFT_RIGHT)))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))


    X_train = np.array(X_train).astype(np.uint8)
    X_test = np.array(X_test).astype(np.uint8)
    GT_train = np.array(GT_train)
    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1


    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def mvtec_ac(cl, path, n_anom_per_cls, seed=None):
    # Uguale a quello di sopra ma restituisce il vettore delle classi anomale
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []
    anomaly_type = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                anomaly_type.append(cl_a)

    X_train = np.array(X_train).astype(np.uint8)# / 255.0).astype(np.float32)
    #X_train = np.swapaxes(X_train, 2, 3)
    #X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.array(X_test).astype(np.uint8)
    #X_test = np.swapaxes(X_test, 2, 3)
    #X_test = np.swapaxes(X_test, 1, 2)


    GT_train = np.array(GT_train)#.astype(np.uint8)
    #GT_train = np.swapaxes(GT_train, 2, 3)
    #GT_train = np.swapaxes(GT_train, 1, 2)

    GT_test = np.array(GT_test)#.astype(np.uint8)
    #GT_test = np.swapaxes(GT_test, 2, 3)
    #GT_test = np.swapaxes(GT_test, 1, 2)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1


    return X_train, Y_train, X_test, Y_test, GT_train, GT_test, np.array(anomaly_type)


def mvtec_only_one(cl, path, n_anom_per_cls, a_cls, seed=None, return_at=False):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    #anom_groups = (
    #    3, 8, 5, 5, 5, 4, 5, 4, 7, 5, 5, 1, 4, 5, 7
    #)

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []
    anomaly_type = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    normal_files_tr.sort()
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    normal_files_te.sort()
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    outlier_classes.sort()
    ac_id = 0
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl_a))
        outlier_file.sort()
        outlier_file = np.array(outlier_file)
        idxs = np.random.permutation(outlier_file.shape[0])

        # Train
        n_anom_per_cls_cl = n_anom_per_cls if a_cls == ac_id else 0
        if a_cls == ac_id:
            print(cl_a )
        ac_id += 1
        for file in outlier_file[idxs[: n_anom_per_cls_cl]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls_cl:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                anomaly_type.append(cl_a)


    X_train = np.array(X_train).astype(np.uint8)# / 255.0).astype(np.float32)
    #X_train = np.swapaxes(X_train, 2, 3)
    #X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.array(X_test).astype(np.uint8)
    #X_test = np.swapaxes(X_test, 2, 3)
    #X_test = np.swapaxes(X_test, 1, 2)


    GT_train = np.array(GT_train)#.astype(np.uint8)
    #GT_train = np.swapaxes(GT_train, 2, 3)
    #GT_train = np.swapaxes(GT_train, 1, 2)

    GT_test = np.array(GT_test)#.astype(np.uint8)
    #GT_test = np.swapaxes(GT_test, 2, 3)
    #GT_test = np.swapaxes(GT_test, 1, 2)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1


    if return_at:
        return X_train, Y_train, X_test, Y_test, GT_train, GT_test, np.array(anomaly_type)
    else:
        return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def mvtec_only_one_augmented(cl, path, n_anom_per_cls, a_cls, seed=None, return_at=False):
    print('Creating results with augmentation...')
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )


    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []
    anomaly_type = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    ac_id = 0
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        n_anom_per_cls_cl = n_anom_per_cls if a_cls == ac_id else 0
        ac_id += 1
        for file in outlier_file[idxs[: n_anom_per_cls_cl]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                img = Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')
                gt = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')
                X_train.append(np.array(img))
                GT_train.append(np.array(gt))

                # Augment data
                # Rotate 90
                X_train.append(np.array(img.transpose(Image.ROTATE_90)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_90)))
                # Rotate 180
                X_train.append(np.array(img.transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_180)))
                # Rotate 270
                X_train.append(np.array(img.transpose(Image.ROTATE_270)))
                GT_train.append(np.array(gt.transpose(Image.ROTATE_270)))

                # Simmetria sulla diagonale principale
                X_train.append(np.array(img.transpose(Image.TRANSPOSE)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE)))
                # Simmetria sulla diagonale secondaria
                X_train.append(np.array(img.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                GT_train.append(np.array(gt.transpose(Image.TRANSPOSE).transpose(Image.ROTATE_180)))
                # Horizontal flip
                X_train.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_TOP_BOTTOM)))
                # Vertical flip
                X_train.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))
                GT_train.append(np.array(gt.transpose(Image.FLIP_LEFT_RIGHT)))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls_cl:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))
                anomaly_type.append(cl_a)


    X_train = np.array(X_train).astype(np.uint8)# / 255.0).astype(np.float32)
    #X_train = np.swapaxes(X_train, 2, 3)
    #X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.array(X_test).astype(np.uint8)
    #X_test = np.swapaxes(X_test, 2, 3)
    #X_test = np.swapaxes(X_test, 1, 2)


    GT_train = np.array(GT_train)#.astype(np.uint8)
    #GT_train = np.swapaxes(GT_train, 2, 3)
    #GT_train = np.swapaxes(GT_train, 1, 2)

    GT_test = np.array(GT_test)#.astype(np.uint8)
    #GT_test = np.swapaxes(GT_test, 2, 3)
    #GT_test = np.swapaxes(GT_test, 1, 2)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1


    if return_at:
        return X_train, Y_train, X_test, Y_test, GT_train, GT_test, np.array(anomaly_type)
    else:
        return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def aebad_s(cl, sg, path, n_anom_per_cls, seed=None):
    np.random.seed(seed=seed)

    labels = ('breakdown', 'ablation', 'groove', 'fracture')
    subgroups = ('background', 'illumination', 'view', 'same')

    class_o = labels[cl]
    subgr_o = subgroups[sg]

    root = path #os.path.join(path, class_o, subgr_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    for subdir in subgroups:
        if subdir != 'same':
            f_path = os.path.join(root, 'train', 'good', subdir)
            normal_files_tr = os.listdir(f_path)
            for file in normal_files_tr:
                if not file.split('/')[-1].startswith('._') and 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    image = Image.open(os.path.join(f_path, file)).convert('RGB')
                    image = image.resize((448, 448))
                    image = np.array(image)
                    X_train.append(image)
                    GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good', 'same')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if not file.split('/')[-1].startswith('._') and 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = Image.open(os.path.join(f_path, file)).convert('RGB')
            image = image.resize((448, 448))
            image = np.array(image)
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        f_path = os.path.join(root, 'test', cl_a, 'same')
        outlier_file = np.array(os.listdir(f_path))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            if not file.split('/')[-1].startswith('._') and 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                image = Image.open(os.path.join(f_path, file)).convert('RGB')
                image = image.resize((448, 448))
                image = np.array(image)
                X_train.append(np.array(image))
                image = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/same/' + file)).convert('RGB')
                image = image.resize((448, 448))
                image = np.array(image)
                GT_train.append(np.array(image))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if not file.split('/')[-1].startswith('._') and 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                image = Image.open(os.path.join(f_path, file)).convert('RGB')
                image = image.resize((448, 448))
                image = np.array(image)
                X_test.append(image)
                image = Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/same/' + file)).convert('RGB')
                image = image.resize((448, 448))
                image = np.array(image)
                GT_test.append(image)

    print('GT ', len(GT_train))
    X_train = np.array(X_train).astype(np.uint8)# / 255.0).astype(np.float32)
    #X_train = np.swapaxes(X_train, 2, 3)
    #X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.array(X_test).astype(np.uint8)
    #X_test = np.swapaxes(X_test, 2, 3)
    #X_test = np.swapaxes(X_test, 1, 2)

    print(X_train.shape)
    print(X_test.shape)

    GT_train = np.array(GT_train)#.astype(np.uint8)
    #GT_train = np.swapaxes(GT_train, 2, 3)
    #GT_train = np.swapaxes(GT_train, 1, 2)

    GT_test = np.array(GT_test)#.astype(np.uint8)
    #GT_test = np.swapaxes(GT_test, 2, 3)
    #GT_test = np.swapaxes(GT_test, 1, 2)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    print(Y_train.sum())
    print(Y_test.sum())

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test