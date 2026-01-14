import torchvision.transforms as T
import torch


#PER FARE EVENTUALE AUGMENTATION
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.02):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

 
def get_vit_augmentation(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),            # -> [0,1]
        AddGaussianNoise(0., 0.02)   
    ])

def get_vit_test_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])