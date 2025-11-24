import argparse
import os
import numpy as np
import torch
from codecarbon import EmissionsTracker
from datasets.mvtec_vit import mvtec_ViT
from dataloaders.tensor_loader import TensorDatasetAD
from models import ViT_CNN_Attn
from aexad_script import Trainer


def save_dataset(path, X_train, Y_train, X_test, Y_test, GT_train, GT_test):
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "Y_train.npy"), Y_train)
    np.save(os.path.join(path, "GT_train.npy"), GT_train)

    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "Y_test.npy"), Y_test)
    np.save(os.path.join(path, "GT_test.npy"), GT_test)

    print(f"[DATA SAVED] → {path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", type=str, required=True, help="Dataset (mvtec)")
    parser.add_argument("-c", type=int, required=True, help="Class index")
    parser.add_argument("-na", type=int, default=1, help="Number of train anomalies")
    parser.add_argument("-s", type=int, default=0, help="Seed")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # ============================================================
    #                  DATASET: MVTec (default)
    # ============================================================
    if args.ds == "mvtec":

        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec_ViT(
            args.c, "datasets/mvtec", n_anom_per_cls=args.na, seed=args.s
        )

        data_path = os.path.join("datasets/mvtec", str(args.c), str(args.s))
        save_path = os.path.join("results/mvtec", str(args.c), str(args.s), str(args.na))
        os.makedirs(save_path, exist_ok=True)

    else:
        raise ValueError(f"Dataset {args.ds} non supportato ancora.")

    # ============================================================
    #               CONVERSIONE IN NCHW FLOAT32
    # ============================================================
    X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32) / 255.
    X_test  = X_test.transpose(0, 3, 1, 2).astype(np.float32) / 255.

    GT_train = GT_train.transpose(0, 3, 1, 2).astype(np.float32)
    GT_test  = GT_test.transpose(0, 3, 1, 2).astype(np.float32)

    save_dataset(data_path, X_train, Y_train, X_test, Y_test, GT_train, GT_test)

    # ============================================================
    #                 COSTRUZIONE DATASET E DATALOADER
    # ============================================================
    train_set = TensorDatasetAD(data_path, train=True)
    test_set  = TensorDatasetAD(data_path, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False
    )

    # ============================================================
    #                         MODELLO
    # ============================================================
    model = ViT_CNN_Attn((3, 224, 224))

    # ============================================================
    #                         TRAINER
    # ============================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        save_path=save_path,
        cuda=True
    )

    '''tracker = EmissionsTracker()
    tracker.start()'''

    # ============================================================
    #                        TRAINING
    # ============================================================
    trainer.train(epochs=args.epochs)

    # ============================================================
    #                        TESTING
    # ============================================================
    
    print(">>> Running TEST ...")
    heatmaps, scores, gtmaps, labels = trainer.test()

    np.save(os.path.join(save_path, "aexad_htmaps_vit.npy"), heatmaps)
    np.save(os.path.join(save_path, "aexad_scores_vit.npy"), scores)
    np.save(os.path.join(save_path, "aexad_labels.npy"), labels)
    np.save(os.path.join(save_path, "aexad_gt.npy"), gtmaps)

    print("\n=== EXPERIMENT COMPLETED ===")
