import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.gaussian import gaussian_smoothing
from utils.testing_tools import aexad_heatmap_and_score
from loss import  AEXAD_Loss
from models import ViT_CNN_Attn

class Trainer:
    def __init__(self, model, train_loader, test_loader, save_path='.', cuda=True):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_path = save_path
        
        if isinstance(self.model, ViT_CNN_Attn):
            name = "model_vit"
        else:
            raise RuntimeError("Unexpected model type")

        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        # ---- LOSS ----
        self.criterion = AEXAD_Loss(debug=False)
            
        if self.cuda:
            self.criterion = self.criterion.cuda()

        # --- OPTIMIZER  ---
        # enc_unfrozen = ultimi layer del ViT (6–11)
        # decoder = intero decoder AE-XAD
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=1e-3,
            weight_decay=1e-4
        )

        # scheduler sarà definito nel train
        self.scheduler = None

    # ============================================
    #                   TRAIN
    # ============================================
    def train(self, epochs=200):
        
        step = 0
        self.model.train()

        for epoch in range(epochs):
            tbar = tqdm(self.train_loader)
            epoch_loss = 0

            for batch in tbar:
                img = batch["image"]
                if self.cuda:
                    img = img.cuda()
                
                out = self.model(img)
                gt = batch["gt_label"]     
                y  = batch["label"]        
                
                loss = self.criterion(out, img, gt, y)

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                step += 1

            avg_loss = epoch_loss / len(self.train_loader)
            tbar.set_description(f"[Epoch {epoch}] Loss={avg_loss:.4f}")
            print(f"[Epoch {epoch}] Loss={avg_loss:.4f}")

        torch.save(self.model.state_dict(), os.path.join(self.save_path, "vit_final.pt"))

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

                img = batch["image"]      # (1,3,224,224)
                gt  = batch["gt_label"]   # (1,1,224,224)
                lab = batch["label"]

                if self.cuda:
                    img = img.cuda()

                # --------------------------------------------
                #               MODEL FORWARD
                # --------------------------------------------
                out_t = self.model(img).cpu()      # tensor (1,3,224,224)
                out   = out_t.numpy()[0]           # numpy  (3,224,224)

                img_np = img.cpu().numpy()[0]      # (3,224,224)
                gt_np  = gt.numpy()[0]             # (1,224,224)

                # --------------------------------------------
                #      AE-XAD HEATMAP & SCORE UFFICIALI
                # --------------------------------------------
                e_raw, h_filtered, h_bin, score, k_hat = aexad_heatmap_and_score(
                    img_np, out
                )

                # Salviamo per le metriche finali
                heatmaps.append(h_filtered[None, ...])
                scores.append(score)
                gtmaps.append(gt_np[None, ...])
                labels.append(lab.numpy())

                # =====================================================
                #           PLOT 6 IMMAGINI (STILE PAPER)
                # =====================================================

                fig = plt.figure(figsize=(14,8))

                plt.subplot(2,3,1)
                plt.imshow(img_np.transpose(1,2,0))
                plt.title("Input image")
                plt.axis("off")

                plt.subplot(2,3,2)
                plt.imshow(out.transpose(1,2,0))
                plt.title("AE-XAD reconstruction")
                plt.axis("off")

                plt.subplot(2,3,3)
                plt.imshow(e_raw, cmap="inferno")
                plt.title("Raw reconstruction error")
                plt.axis("off")

                plt.subplot(2,3,4)
                plt.imshow(h_filtered, cmap="inferno")
                plt.title(f"Filtered heatmap (k̂={k_hat})")
                plt.axis("off")

                plt.subplot(2,3,5)
                plt.imshow(h_bin, cmap="inferno")
                plt.title("Binarized heatmap")
                plt.axis("off")

                plt.subplot(2,3,6)
                plt.imshow(gt_np.squeeze(), cmap="gray")
                plt.title("Ground truth mask")
                plt.axis("off")

                plt.savefig(os.path.join(results_dir, f"test_{i}_full.jpg"))
                plt.close(fig)

        return (
            np.concatenate(heatmaps),
            np.array(scores),
            np.concatenate(gtmaps),
            np.concatenate(labels)
        )


