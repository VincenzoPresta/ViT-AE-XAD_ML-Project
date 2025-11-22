import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.gaussian import gaussian_smoothing
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
        self.criterion = AEXAD_Loss()
            
        if self.cuda:
            self.criterion = self.criterion.cuda()

        # ---- OPTIMIZER ----
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': 5e-4}
        ], lr=1e-3, weight_decay=1e-5)

        # scheduler sarà definito nel train
        self.scheduler = None


    # ============================================
    #                   TRAIN
    # ============================================
    def train(self, epochs=200):
        
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

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
                
                gt = batch["gt_label"]     # o batch["gt"]
                y  = batch["label"]        # opzionale
                
                loss = self.criterion(out, img, gt, y)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                step += 1

            avg_loss = epoch_loss / step
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

                # ----------------------------
                # MODEL FORWARD
                # ----------------------------
                out_t = self.model(img).cpu()      # tensor (1,3,224,224)
                out   = out_t.numpy()[0]           # numpy  (3,224,224)

                img_np = img.cpu().numpy()[0]      # (3,224,224)
                gt_np  = gt.numpy()[0]             # (1,224,224)

                # ----------------------------
                # HEATMAP GREZZA
                # ----------------------------
                hmap = ((img_np - out)**2).sum(0)     # (224,224)
                hmap = hmap**0.7
                hmap_s = gaussian_smoothing(hmap[None, ...], 21, 4)[0]
                hmap_s /= (hmap_s.max() + 1e-8)

                heatmaps.append(hmap_s[None, ...])
                scores.append(hmap_s.mean())
                gtmaps.append(gt_np[None, ...])
                labels.append(lab.numpy())

                # =====================================================
                #                PLOT 6 FIGURE (STILE PAPER)
                # =====================================================

                # 1) RAW RECON ERROR
                e = ((img_np - out)**2).sum(0)     # (H,W)

                # 2) SCORE MAP raw
                score_raw = e.copy()

                # 3) NORMALIZATION ê
                e_norm = (e - e.mean()) / (e.std() + 1e-6)

                # 4) GAUSSIAN FILTER F_k(ê)
                h = gaussian_smoothing(e_norm[None,...], kernel_size=21, sigma=4)[0]

                # 5) BINARIZATION
                mu_h = h.mean()
                sigma_h = h.std()
                binary_h = (h > (mu_h + sigma_h)).astype(np.uint8)

                # -----------------------------------------------------
                #               PLOTTING (2×3 grid)
                # -----------------------------------------------------
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
                plt.imshow(score_raw, cmap="inferno")
                plt.title("AE-XAD heatmap (raw)")
                plt.axis("off")

                plt.subplot(2,3,4)
                plt.imshow(h, cmap="inferno")
                plt.title("Filter application")
                plt.axis("off")

                plt.subplot(2,3,5)
                plt.imshow(binary_h, cmap="cividis")
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

