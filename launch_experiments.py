import argparse
import os
import numpy as np
import torch
from datasets.mvtec_vit import mvtec_ViT
from dataloaders.tensor_loader import TensorDatasetAD
from models import ViT_CNN_Attn
from aexad_script import Trainer
from torch.utils.data import Sampler
import math
from collections import Counter


def save_dataset(path, X_train, Y_train, X_test, Y_test, GT_train, GT_test):
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "Y_train.npy"), Y_train)
    np.save(os.path.join(path, "GT_train.npy"), GT_train)

    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "Y_test.npy"), Y_test)
    np.save(os.path.join(path, "GT_test.npy"), GT_test)

    print(f"[DATA SAVED] → {path}")
    
class RatioBatchSampler(Sampler):
    """
    Produce batch con ratio fisso: n_anom = floor(B/3), n_norm = B - n_anom.
    Campiona con reshuffle e wrapping (replacement implicito).
    """
    def __init__(self, labels, batch_size, anom_frac=1/3, seed=0):
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.anom_bs = max(1, int(math.floor(self.batch_size * anom_frac)))
        self.norm_bs = self.batch_size - self.anom_bs
        assert self.norm_bs > 0, "batch_size troppo piccolo"
        self.rng = np.random.RandomState(seed)

        self.anom_idx = np.where(self.labels == 1)[0].tolist()
        self.norm_idx = np.where(self.labels == 0)[0].tolist()
        assert len(self.anom_idx) > 0 and len(self.norm_idx) > 0, "servono sia normali che anomalie"

        # numero batch per epoca: usa tutti i normali una volta circa
        self.num_batches = int(math.ceil(len(self.norm_idx) / self.norm_bs))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        anom = self.anom_idx.copy()
        norm = self.norm_idx.copy()
        self.rng.shuffle(anom)
        self.rng.shuffle(norm)

        a_ptr, n_ptr = 0, 0

        for _ in range(self.num_batches):
            if n_ptr + self.norm_bs > len(norm):
                self.rng.shuffle(norm)
                n_ptr = 0
            if a_ptr + self.anom_bs > len(anom):
                self.rng.shuffle(anom)
                a_ptr = 0

            batch = norm[n_ptr:n_ptr+self.norm_bs] + anom[a_ptr:a_ptr+self.anom_bs]
            self.rng.shuffle(batch)

            n_ptr += self.norm_bs
            a_ptr += self.anom_bs

            yield batch
            


def print_dataset_stats(ds, name="dataset"):
    # prova: alcuni dataset espongono ds.labels
    labels = None
    if hasattr(ds, "labels"):
        labels = np.asarray(ds.labels)
    else:
        # fallback: prova a estrarre label chiamando __getitem__ su un subset (costo alto)
        raise RuntimeError(f"{name}: ds.labels non trovato. Aggiungi un attributo labels nel dataset.")

    c = Counter(labels.tolist())
    n0 = c.get(0, 0)
    n1 = c.get(1, 0)
    tot = len(labels)

    print(f"[{name}] N={tot}  normal(0)={n0}  anomal(1)={n1}  anom_ratio={n1/max(1,tot):.4f}")
    

def check_first_batch_ratio(loader, name="train_loader"):
    x, y, gt = next(iter(loader))  # adattalo se il tuo dataset ritorna tuple diverse
    # y può essere shape [B] o [B,1] o [B, ...]
    y_flat = y.view(-1).detach().cpu().numpy()
    print(f"[{name}] batch_size={len(y_flat)}  y.mean={y_flat.mean():.4f}  #anom={int(y_flat.sum())}")


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
            args.c, "datasets/mvtec", n_anom_per_cls=args.na, seed=args.s , use_copy_paste=True
        )

        data_path = os.path.join("datasets/mvtec", str(args.c), str(args.s))
        save_path = os.path.join(
            "results/mvtec", str(args.c), str(args.s), str(args.na)
        )
        os.makedirs(save_path, exist_ok=True)

    else:
        raise ValueError(f"Dataset {args.ds} non supportato ancora.")

    # ============================================================
    #               CONVERSIONE IN NCHW FLOAT32
    # ============================================================
    X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    X_test = X_test.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    GT_train = GT_train.transpose(0, 3, 1, 2).astype(np.float32)
    GT_test = GT_test.transpose(0, 3, 1, 2).astype(np.float32)

    save_dataset(data_path, X_train, Y_train, X_test, Y_test, GT_train, GT_test)

    # ============================================================
    #                 COSTRUZIONE DATASET E DATALOADER
    # ============================================================
    train_set = TensorDatasetAD(data_path, train=True)
    test_set = TensorDatasetAD(data_path, train=False)

    batch_sampler = RatioBatchSampler(train_set.labels, batch_size=args.batch_size, anom_frac=1/3, seed=args.s)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    print_dataset_stats(train_set, "TRAIN_SET")
    print_dataset_stats(test_set,  "TEST_SET")
    
    
    batch = next(iter(train_loader))
    print("[BATCH] type:", type(batch))
    print("[BATCH] len:", len(batch))
    for i, item in enumerate(batch):
        print(f"  item[{i}] type={type(item)}")
        if hasattr(item, "shape"):
            print(f"    shape={tuple(item.shape)} dtype={item.dtype}")
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            print(f"    sample type={type(item[0])}")
        else:
            # stampa un esempio, ma limitato
            try:
                print(f"    example={item}")
            except:
                pass

    
    
    #check_first_batch_ratio(train_loader, "TRAIN_LOADER")


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
        cuda=True,
    )

    """tracker = EmissionsTracker()
    tracker.start()"""

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
