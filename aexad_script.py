import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.testing_tools import aexad_heatmap_and_score
from loss import AEXAD_Loss
from models import ViT_CNN_Attn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score


def compute_XAUC_per_image(gt, hm):
    gt_flat = gt.flatten()
    hm_flat = hm.flatten()
    if np.unique(gt_flat).size == 1:
        return None
    return roc_auc_score(gt_flat, hm_flat)

def compute_IoU_F1_paper(gt, hm):
    gt_flat = gt.flatten()

    mu_h = hm.mean()
    sigma_h = hm.std()
    pred = (hm > (mu_h + sigma_h)).astype(np.uint8).flatten()

    if pred.sum() == 0:
        return 0.0, 0.0

    IoU = jaccard_score(gt_flat, pred, zero_division=0)
    F1  = f1_score(gt_flat, pred, zero_division=0)
    return IoU, F1


class Trainer:
    def __init__(self, model, train_loader, test_loader, save_path=".", cuda=True):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_path = save_path

        if not isinstance(self.model, ViT_CNN_Attn):
            raise RuntimeError("Unexpected model type")

        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        # ----------------------
        # LOSS AE-XAD (paper)
        # ----------------------
        self.criterion = AEXAD_Loss(debug=False)
        if self.cuda:
            self.criterion = self.criterion.cuda()

        # ----------------------
        # OPTIMIZER
        # ----------------------
        
        #ViT frozen
        '''self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,
            weight_decay=1e-5
        )'''
        
        #  ViT full trainable
        decoder_params = []
        vit_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            if "encoder.encoder_vit" in name or "encoder.conv_proj" in name or "encoder.class_token" in name:
                vit_params.append(p)
            else:
                decoder_params.append(p)

        param_groups = [
            {"params": decoder_params, "lr": 5e-4, "weight_decay": 1e-5},
            {"params": vit_params,     "lr": 1e-4, "weight_decay": 1e-5},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999)
        )


            
    def train(self, epochs=200):
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.model.train()

        for epoch in range(epochs):
            tbar = tqdm(self.train_loader)
            epoch_loss = 0.0
                        
            for batch in tbar:
                img = batch["image"]
                gt  = batch["gt_label"]
                y   = batch["label"]

                if self.cuda:
                    img = img.cuda()
                    gt  = gt.cuda()
                    y   = y.cuda()

                # forward
                out = self.model(img)

                # loss
                loss = self.criterion(out, img, gt, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += float(loss.item())

            # scheduler update
            self.scheduler.step()

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"[Epoch {epoch}] Loss={avg_loss:.4f}")
            
           

            # --- periodic evaluation (logging only) ---
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                metrics = self.evaluate_metrics()
                print(f"[Eval @ {epoch+1}] {metrics}")
                self.model.train()
                
                

        torch.save(self.model.state_dict(), os.path.join(self.save_path, "vit_final.pt"))
        print("[Training done] saved vit_final.pt")



    # ============================================
    #                   TEST
    # ============================================
    def test(self):
        self.model.eval()

        heatmaps = []
        scores = []
        gtmaps = []
        labels = []

        results_dir = os.path.join(self.save_path, "test_images")
        os.makedirs(results_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):

                img = batch["image"]  # (1,3,224,224)
                gt = batch["gt_label"]  # (1,1,224,224)
                lab = batch["label"]

                if self.cuda:
                    img = img.cuda()

                # --------------------------------------------
                #               MODEL FORWARD
                # --------------------------------------------
                out_t = self.model(img).cpu()  # tensor (1,3,224,224)
                out = out_t.numpy()[0]  # numpy  (3,224,224)

                img_np = img.cpu().numpy()[0]  # (3,224,224)
                gt_np = gt.numpy()[0]  # (1,224,224)

                # --------------------------------------------
                #      AE-XAD HEATMAP & SCORE UFFICIALI
                # --------------------------------------------
                
                
                lab_val = int(batch["label"].item())
                gt_sum = float(gt.sum().item())

                e_raw, h_filtered, h_bin, score, k_hat = aexad_heatmap_and_score(
                    img_np, out, lab_val
                )

                heatmaps.append(h_filtered[None, ...])
                scores.append(score)
                gtmaps.append(gt_np[None, ...])
                labels.append(lab_val)
                
                print(f"[SAMPLE {i}] LABEL={lab_val} GT_SUM={gt_sum} score={score} k_hat={k_hat}")


                # =====================================================
                #           PLOT 6 IMMAGINI (STILE PAPER)
                # =====================================================

                fig = plt.figure(figsize=(14, 8))

                plt.subplot(2, 3, 1)
                plt.imshow(img_np.transpose(1, 2, 0))
                plt.title("Input image")
                plt.axis("off")

                plt.subplot(2, 3, 2)
                plt.imshow(out.transpose(1, 2, 0))
                plt.title("AE-XAD reconstruction")
                plt.axis("off")

                plt.subplot(2, 3, 3)
                plt.imshow(e_raw, cmap="inferno")
                plt.title("Raw reconstruction error")
                plt.axis("off")

                plt.subplot(2, 3, 4)
                plt.imshow(h_filtered, cmap="inferno")
                plt.title(f"Filtered heatmap")
                plt.axis("off")

                plt.subplot(2, 3, 5)
                plt.imshow(h_bin, cmap="inferno")
                plt.title("Binarized heatmap")
                plt.axis("off")

                plt.subplot(2, 3, 6)
                plt.imshow(gt_np.squeeze(), cmap="gray")
                plt.title("Ground truth mask")
                plt.axis("off")

                plt.savefig(os.path.join(results_dir, f"test_{i}.jpg"))
                plt.close(fig)

        return (
            np.concatenate(heatmaps),
            np.array(scores),
            np.concatenate(gtmaps),
            np.array(labels),
        )
        
        
    #EVAL    
    def evaluate_metrics(self):
        self.model.eval()

        heatmaps = []
        gtmaps = []
        labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                img = batch["image"]
                gt  = batch["gt_label"]
                lab = batch["label"]

                if self.cuda:
                    img = img.cuda()

                out_t = self.model(img).cpu()
                out = out_t.numpy()[0]          # (3,224,224)
                img_np = img.cpu().numpy()[0]   # (3,224,224)
                gt_np  = gt.numpy()[0]          # (1,224,224)

                e_raw, h_filtered, h_bin, score, k_hat = aexad_heatmap_and_score(img_np, out,labels)

                heatmaps.append(h_filtered[None, ...])
                gtmaps.append(gt_np[None, ...])
                labels.append(lab.numpy())

        heatmaps = np.concatenate(heatmaps)
        gtmaps   = np.concatenate(gtmaps)
        labels   = np.array(labels)

        # X-AUC
        per_img_auc = []
        for hm, gt in zip(heatmaps, gtmaps):
            hm = np.squeeze(hm)
            gt = (np.squeeze(gt) > 0.001).astype(np.uint8)
            auc = compute_XAUC_per_image(gt, hm)
            if auc is not None:
                per_img_auc.append(auc)
        X_AUC = float(np.mean(per_img_auc)) if len(per_img_auc) > 0 else float("nan")

        # IoU/F1 (solo anomalie)
        IoUs, F1s = [], []
        for hm, gt, lab in zip(heatmaps, gtmaps, labels):
            if int(lab) == 0:
                continue
            hm = np.squeeze(hm)
            gt = (np.squeeze(gt) > 0.001).astype(np.uint8)
            iou_m, f1_m = compute_IoU_F1_paper(gt, hm)
            IoUs.append(iou_m)
            F1s.append(f1_m)

        IoU_max = float(np.mean(IoUs)) if len(IoUs) > 0 else 0.0
        F1_max  = float(np.mean(F1s))  if len(F1s) > 0 else 0.0

        return {"X_AUC": X_AUC, "IoU_max": IoU_max, "F1_max": F1_max}
    




