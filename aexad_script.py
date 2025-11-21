import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.filters import gaussian_smoothing
from loss import AEXAD_loss_ViT_SSIM, AEXAD_Loss
from AE_architectures import ViT_CNN_Attn


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
                print("[DEBUG train] image:", img.shape, "min:", img.min().item(), "max:", img.max().item())

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

                tbar.set_description(f"Epoch {epoch} | Loss {epoch_loss/(step):.4f}")

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

                img = batch["image"]
                gt  = batch["gt_label"]
                lab = batch["label"]

                if self.cuda:
                    img = img.cuda()

                out = self.model(img).cpu().numpy()
                img_np = img.cpu().numpy()
                gt_np  = gt.numpy()

                # heatmap
                hmap = ((img_np - out)**2).sum(1)
                hmap = hmap**0.7
                hmap = gaussian_smoothing(hmap, 21, 4)
                hmap /= hmap.max(axis=(1,2), keepdims=True) + 1e-8

                score = hmap.reshape(hmap.shape[0], -1).mean(1)

                heatmaps.append(hmap)
                scores.append(score)
                gtmaps.append(gt_np)
                labels.append(lab.numpy())

                # qualitative plot
                fig = plt.figure(figsize=(14,4))
                plt.subplot(1,4,1); plt.imshow(img_np[0].transpose(1,2,0)); plt.axis("off"); plt.title("Input")
                plt.subplot(1,4,2); plt.imshow(out[0].transpose(1,2,0)); plt.axis("off"); plt.title("Reconstruction")
                plt.subplot(1,4,3); plt.imshow(hmap[0], cmap="inferno"); plt.axis("off"); plt.title("Heatmap")
                plt.subplot(1,4,4); plt.imshow(gt_np[0].squeeze(), cmap="gray"); plt.axis("off"); plt.title("GT mask")
                plt.savefig(os.path.join(results_dir, f"test_{i}.jpg"))
                plt.close(fig)

        return (
            np.concatenate(heatmaps),
            np.concatenate(scores),
            np.concatenate(gtmaps),
            np.concatenate(labels)
        )
