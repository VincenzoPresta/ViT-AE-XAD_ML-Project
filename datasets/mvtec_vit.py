import os
import PIL.Image as Image
import numpy as np
import torchvision.transforms as T
from datasets.transforms_vit import get_vit_augmentation

def mvtec_ViT(cl, path, n_anom_per_cls, seed=None):

    print ("loaded mvtec for ViT - new loader")

    np.random.seed(seed)
    
    aug_train = get_vit_augmentation(224)   # augmentation ViT SOLO per train anomalies
    pil_to_tensor = T.ToTensor()

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

        # ---------- TRAIN ANOMALIES WITH AUG ----------
        for file in [outlier_files[i] for i in idxs[:n_anom_per_cls]]:
            img = Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB')
            mask = Image.open(os.path.join(root, 'ground_truth', cl_a, file.replace('.png','_mask.png'))) \
                         .convert('L')

            img = img.resize((224,224), Image.NEAREST)
            mask = mask.resize((224,224), Image.NEAREST)

            # AUGMENTATION APPLIED HERE
            img_aug = aug_train(img)                      # PIL → transforms → tensor normalized
            img_aug = (img_aug * 0.5 + 0.5).clamp(0,1)    # de-normalize back to [0,1]
            img_aug = (img_aug.permute(1,2,0).numpy()*255).astype(np.uint8)

            X_train.append(img_aug)
            GT_train.append(np.array(mask, dtype=np.uint8)[...,None])

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