import torchvision.transforms as T
import torch


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.02):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

 
def get_vit_augmentation(img_size=224): #for train
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.90, 1.00)),
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.05)
        ], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.RandomRotation(degrees=10),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        AddGaussianNoise(0., 0.02),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_vit_test_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
