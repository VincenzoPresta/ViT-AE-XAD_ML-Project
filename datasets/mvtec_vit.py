import os
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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

        # --- replicate "as-is" 5 volte (paper) ---
        for rep in range(5):
            img_aug = aug_train(img)  # solo noise leggero come già fai
            img_aug = (img_aug.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            X_train.append(img_aug)
            GT_train.append(np.array(mask, dtype=np.uint8)[..., None])
        
        cp_ok = 0
        cp_fail = 0

        if use_copy_paste:
            for _ in range(10):
                idx_n = np.random.randint(len(normal_files_tr))
                normal_file = normal_files_tr[idx_n]

                base = Image.open(os.path.join(root, 'train', 'good', normal_file)).convert('RGB')
                base = base.resize((224, 224), Image.NEAREST)

                new_img, new_mask = copy_paste_defect_affine(base, img, mask)

                if new_img is None:
                    cp_fail += 1 
                    continue
                cp_ok += 1

                t_img = aug_train(new_img)
                t_img = (t_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                X_train.append(t_img)
                GT_train.append(np.array(new_mask, dtype=np.uint8)[..., None])
                
        print(f"[DBG CP] ok={cp_ok} fail={cp_fail}")


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


def copy_paste_defect_affine(base_img, anomaly_img, anomaly_mask,
                             rot_deg=20, scale_range=(0.7, 1.3), max_trans_frac=0.10):
    """
    AE-XAD Arrays compliant cut-paste:
    - crop difetto via bbox mask
    - random affine (rot+scale+translation) su difetto e maschera
    - paste su immagine normale in posizione random valida
    """

    base_w, base_h = base_img.size

    mask_np = np.array(anomaly_mask) > 127
    if mask_np.sum() == 0:
        return None, None

    ys, xs = np.where(mask_np)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    defect = anomaly_img.crop((x1, y1, x2 + 1, y2 + 1))
    defect_mask = anomaly_mask.crop((x1, y1, x2 + 1, y2 + 1))

    # --- random affine params ---
    angle = float(np.random.uniform(-rot_deg, rot_deg))
    scale = float(np.random.uniform(scale_range[0], scale_range[1]))

    pw, ph = defect.size
    max_dx = int(max_trans_frac * pw)
    max_dy = int(max_trans_frac * ph)
    trans = (int(np.random.uniform(-max_dx, max_dx)), int(np.random.uniform(-max_dy, max_dy)))

    # apply same affine to defect & mask (NEAREST for mask)
    defect_t = TF.affine(defect, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0], interpolation=Image.BILINEAR)
    mask_t   = TF.affine(defect_mask, angle=angle, translate=trans, scale=scale, shear=[0.0, 0.0], interpolation=Image.NEAREST)

    # bbox after transform (re-crop tight bbox to avoid huge transparent area)
    mask_np2 = np.array(mask_t) > 127
    if mask_np2.sum() == 0:
        return None, None
    ys2, xs2 = np.where(mask_np2)
    y1b, y2b = ys2.min(), ys2.max()
    x1b, x2b = xs2.min(), xs2.max()

    defect_t = defect_t.crop((x1b, y1b, x2b + 1, y2b + 1))
    mask_t   = mask_t.crop((x1b, y1b, x2b + 1, y2b + 1))

    dw, dh = defect_t.size
    if dw <= 0 or dh <= 0:
        return None, None

    max_x = base_w - dw
    max_y = base_h - dh
    if max_x <= 0 or max_y <= 0:
        return None, None

    rx = np.random.randint(0, max_x + 1)
    ry = np.random.randint(0, max_y + 1)

    base_img2 = base_img.copy()
    base_img2.paste(defect_t, (rx, ry), mask_t)

    new_mask = Image.new("L", (base_w, base_h), 0)
    new_mask.paste(mask_t, (rx, ry))

    return base_img2, new_mask
