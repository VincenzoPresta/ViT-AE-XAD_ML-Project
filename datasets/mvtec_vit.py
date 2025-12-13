import os
import numpy as np
import random
import torchvision.transforms as T
from PIL import Image, ImageOps
from datasets.transforms_vit import get_vit_augmentation

import os
import numpy as np
import random
import torchvision.transforms as T
from PIL import Image
from datasets.transforms_vit import get_vit_augmentation  # la versione FIXATA
# Assicura che get_vit_augmentation sia quella senza Normalize e senza jitter/rotazioni.


def mvtec_ViT(cl, path, n_anom_per_cls, seed=None, use_copy_paste=False):

    print("Loaded MVTec for ViT (CORRECTED AE-XAD LOADER)")

    np.random.seed(seed)

    # SOLO: Resize + ToTensor + Noise leggerissimo
    aug_train = get_vit_augmentation(224)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]
    root = os.path.join(path, class_o)

    X_train, X_test, GT_train, GT_test = [], [], [], []

    # ============================================================
    #                 NORMAL TRAIN (NO AUGMENTATION)
    # ============================================================
    normal_tr_path = os.path.join(root, 'train', 'good')
    normal_files_tr = sorted(os.listdir(normal_tr_path))

    for file in normal_files_tr:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img = Image.open(os.path.join(normal_tr_path, file)).convert('RGB')
            img = img.resize((224, 224), Image.NEAREST)

            X_train.append(np.array(img, dtype=np.uint8))
            GT_train.append(np.zeros((224, 224, 1), dtype=np.uint8))

    # ============================================================
    #                 NORMAL TEST (NO AUGMENTATION)
    # ============================================================
    normal_te_path = os.path.join(root, 'test', 'good')
    normal_files_te = sorted(os.listdir(normal_te_path))

    for file in normal_files_te:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img = Image.open(os.path.join(normal_te_path, file)).convert('RGB')
            img = img.resize((224, 224), Image.NEAREST)

            X_test.append(np.array(img, dtype=np.uint8))
            GT_test.append(np.zeros((224, 224, 1), dtype=np.uint8))

    # ============================================================
    #                 ANOMALIE (SELEZIONE GLOBALE CORRETTA)
    # ============================================================
    outlier_path = os.path.join(root, 'test')
    outlier_classes = sorted(os.listdir(outlier_path))

    anom_pool = []  # lista di (cl_a, filename)

    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_files = [
            f for f in os.listdir(os.path.join(outlier_path, cl_a))
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        outlier_files.sort()

        for f in outlier_files:
            anom_pool.append((cl_a, f))

    # shuffle deterministico
    rng = np.random.RandomState(seed)
    rng.shuffle(anom_pool)

    # n_anom_per_cls anomalie TOTALI per training
    train_anoms = anom_pool[:n_anom_per_cls]
    test_anoms  = anom_pool[n_anom_per_cls:]


    # ========================================================
    #        TRAIN ANOMALIES (leggero noise, no rotazioni)
    # ========================================================

    for (cl_a, file) in train_anoms:
        img = Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB')
        mask = Image.open(
            os.path.join(root, 'ground_truth', cl_a, file.replace('.png', '_mask.png'))
        ).convert('L')

        img = img.resize((224, 224), Image.NEAREST)
        mask = mask.resize((224, 224), Image.NEAREST)

        img_aug = aug_train(img)  # -> Tensor [0,1]
        img_aug = (img_aug.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        X_train.append(img_aug)
        GT_train.append(np.array(mask, dtype=np.uint8)[..., None])
        
        # +5 copie dirette (senza copy-paste)
        for _ in range(5):
            img_aug = aug_train(img)
            img_aug = (img_aug.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            X_train.append(img_aug)
            GT_train.append(np.array(mask, dtype=np.uint8)[..., None])


        # ========================================================
        #                COPY-PASTE (versione corretta)
        # ========================================================

        if use_copy_paste:
            for _ in range(10):
                idx_n = np.random.randint(len(normal_files_tr))
                normal_file = normal_files_tr[idx_n]

                base = Image.open(os.path.join(root, 'train', 'good', normal_file)).convert('RGB')
                base = base.resize((224, 224), Image.NEAREST)

                # versione corretta del copy-paste (nessuna rotate, nessun jitter)
                new_img, new_mask = copy_paste_defect_clean(base, img, mask)

                if new_img is None:
                    continue

                # leggero rumore
                t_img = aug_train(new_img)
                t_img = (t_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                X_train.append(t_img)
                GT_train.append(np.array(new_mask, dtype=np.uint8)[..., None])

    # ========================================================
    #                     TEST ANOMALIES
    # ========================================================

    for (cl_a, file) in test_anoms:
        img = Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB')
        mask = Image.open(
            os.path.join(root, 'ground_truth', cl_a, file.replace('.png', '_mask.png'))
        ).convert('L')

        img = img.resize((224,224), Image.NEAREST)
        mask = mask.resize((224,224), Image.NEAREST)

        X_test.append(np.array(img, dtype=np.uint8))
        GT_test.append(np.array(mask, dtype=np.uint8)[..., None])

    # ========================================================
    #               CONVERSIONE FINALE
    # ========================================================
    X_train = np.array(X_train, dtype=np.uint8)
    X_test  = np.array(X_test , dtype=np.uint8)
    GT_train = np.array(GT_train, dtype=np.uint8)
    GT_test  = np.array(GT_test , dtype=np.uint8)

    # Labels (0 = good, 1 = anomaly)
    Y_train = np.zeros(X_train.shape[0], dtype=np.uint8)
    Y_train[len(normal_files_tr):] = 1

    Y_test = np.zeros(X_test.shape[0], dtype=np.uint8)
    Y_test[len(normal_files_te):] = 1

    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"GT_train {GT_train.shape}, GT_test {GT_test.shape}")
    print(f"Training anomalies: {Y_train.sum()}, Test anomalies: {Y_test.sum()}")

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def copy_paste_defect_clean(base_img, anomaly_img, anomaly_mask):
    """
    Versione pulita del copy-paste per AE-XAD.
    Nessuna rotazione, nessun flip, nessun jitter.
    Mantiene coerenza pixel-wise.
    """

    mask_np = np.array(anomaly_mask) > 127
    if mask_np.sum() == 0:
        return None, None

    ys, xs = np.where(mask_np)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    defect = anomaly_img.crop((x1, y1, x2, y2))
    defect_mask = anomaly_mask.crop((x1, y1, x2, y2))

    # riposiziona al suo posto, senza distorsioni
    base_img = base_img.copy()
    base_img.paste(defect, (x1, y1), defect_mask)

    new_mask = Image.new('L', (224,224), 0)
    new_mask.paste(defect_mask, (x1, y1))

    return base_img, new_mask
