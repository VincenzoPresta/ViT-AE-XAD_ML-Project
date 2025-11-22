import os
import numpy as np
import random
import torchvision.transforms as T
from PIL import Image, ImageOps
from datasets.transforms_vit import get_vit_augmentation

def mvtec_ViT(cl, path, n_anom_per_cls, seed=None):

    print ("loaded mvtec for ViT - new loader")

    np.random.seed(seed)
    
    aug_train = get_vit_augmentation(224)   # augmentation ViT SOLO per train anomalies
    
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]
    root = os.path.join(path, class_o)

    X_train, X_test, GT_train, GT_test = [], [], [], []

    # === NORMAL TRAIN (NO AUGMENTATION) ===
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = sorted(os.listdir(f_path))
    for file in normal_files_tr:
        if file.lower().endswith(('png','jpg','jpeg')):
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            img = img.resize((224,224), Image.NEAREST)
            X_train.append(np.array(img, dtype=np.uint8))
            GT_train.append(np.zeros((224,224,1), dtype=np.uint8))

    # === NORMAL TEST (NO AUGMENTATION) ===
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = sorted(os.listdir(f_path))
    for file in normal_files_te:
        if file.lower().endswith(('png','jpg','jpeg')):
            img = Image.open(os.path.join(f_path, file)).convert('RGB')
            img = img.resize((224,224), Image.NEAREST)
            X_test.append(np.array(img, dtype=np.uint8))
            GT_test.append(np.zeros((224,224,1), dtype=np.uint8))

    # === ANOMALIES (AUG ON TRAIN, NO AUG ON TEST) ===
    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = sorted(os.listdir(outlier_data_dir))

    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_files = [
            f for f in os.listdir(os.path.join(outlier_data_dir, cl_a))
            if f.lower().endswith(('png','jpg','jpeg'))
        ]
        outlier_files.sort()
        idxs = np.random.permutation(len(outlier_files))

        # ---------- TRAIN ANOMALIES WITH AUG + COPY-PASTE ----------
        for file in [outlier_files[i] for i in idxs[:n_anom_per_cls]]:

            # load anomaly + mask
            img = Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB')
            mask = Image.open(os.path.join(root, 'ground_truth', cl_a, file.replace('.png','_mask.png'))).convert('L')

            img = img.resize((224,224), Image.NEAREST)
            mask = mask.resize((224,224), Image.NEAREST)

            # 1) ADD ORIGINAL ANOMALY (augmented)
            img_aug = aug_train(img)
            img_aug = (img_aug * 0.5 + 0.5).clamp(0,1)
            img_aug = (img_aug.permute(1,2,0).numpy()*255).astype(np.uint8)

            X_train.append(img_aug)
            GT_train.append(np.array(mask, dtype=np.uint8)[...,None])

            # 2) COPY-PASTE: generate 10 synthetic anomalies
            # pick normal base images
            for _ in range(10):
                idx_n = np.random.randint(len(normal_files_tr))
                normal_file = normal_files_tr[idx_n]

                base = Image.open(os.path.join(root, 'train', 'good', normal_file)).convert('RGB')
                base = base.resize((224,224), Image.NEAREST)

                new_img, new_mask = copy_paste_defect(base, img, mask)

                if new_img is None:
                    continue

                # optional: apply ViT augmentation after paste
                t_img = aug_train(new_img)
                t_img = (t_img * 0.5 + 0.5).clamp(0,1)
                t_img = (t_img.permute(1,2,0).numpy()*255).astype(np.uint8)

                X_train.append(t_img)
                GT_train.append(np.array(new_mask, dtype=np.uint8)[...,None])

        # ---------- TEST ANOMALIES (NO AUG) ----------
        for file in [outlier_files[i] for i in idxs[n_anom_per_cls:]]:
            img = Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB')
            mask = Image.open(os.path.join(root, 'ground_truth', cl_a, file.replace('.png','_mask.png'))) \
                         .convert('L')

            img = img.resize((224,224), Image.NEAREST)
            mask = mask.resize((224,224), Image.NEAREST)

            X_test.append(np.array(img, dtype=np.uint8))
            GT_test.append(np.array(mask, dtype=np.uint8)[...,None])

    # Convert arrays
    X_train = np.array(X_train, dtype=np.uint8)
    X_test  = np.array(X_test, dtype=np.uint8)
    GT_train = np.array(GT_train, dtype=np.uint8)
    GT_test  = np.array(GT_test, dtype=np.uint8)

    # Labels
    Y_train = np.zeros(X_train.shape[0]); Y_train[len(normal_files_tr):] = 1
    Y_test  = np.zeros(X_test.shape[0]);  Y_test[len(normal_files_te):] = 1

    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"GT_train {GT_train.shape}, GT_test {GT_test.shape}")
    print(f"Training anomalies: {Y_train.sum()}, Test anomalies: {Y_test.sum()}")

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def copy_paste_defect(normal_img, anomaly_img, anomaly_mask):
    """
    normal_img: PIL RGB 224x224 (base)
    anomaly_img: PIL RGB 224x224 (fonte difetto)
    anomaly_mask: PIL L 224x224 (0/255)
    """

    # Convert mask to boolean
    mask_np = np.array(anomaly_mask) > 127
    if mask_np.sum() == 0:
        return None, None  # anomaly too small

    # Bounding box of defect
    ys, xs = np.where(mask_np)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    defect = anomaly_img.crop((x1, y1, x2, y2))
    defect_mask = anomaly_mask.crop((x1, y1, x2, y2))

    # --- geometric light augmentations ---
    if random.random() < 0.5:
        defect = ImageOps.mirror(defect)
        defect_mask = ImageOps.mirror(defect_mask)

    if random.random() < 0.5:
        angle = random.uniform(-25, 25)
        defect = defect.rotate(angle, resample=Image.BILINEAR, expand=True)
        defect_mask = defect_mask.rotate(angle, resample=Image.NEAREST, expand=True)

    # Resize patch back roughly to original bbox size (keeps consistency)
    defect = defect.resize((x2-x1, y2-y1), Image.BILINEAR)
    defect_mask = defect_mask.resize((x2-x1, y2-y1), Image.NEAREST)

    # Paste defect on normal image
    normal_img = normal_img.copy()
    normal_img.paste(defect, (x1, y1), defect_mask)

    # Build the new mask (binary)
    new_mask = Image.new('L', (224,224), 0)
    new_mask.paste(defect_mask, (x1, y1))

    return normal_img, new_mask