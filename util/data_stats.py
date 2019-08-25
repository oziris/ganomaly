import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = ImageFolder(os.path.join('/media/samo/Workspace/ganomaly_pytorch/data/dl4cv', 'train'), transform)
        
    def __getitem__(self, index):
        x = self.dataset[index]
        return x

    def __len__(self):
        return len(self.dataset)

def online_mean_and_std(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        b, c, h, w = data.shape
        print(data.shape)
        exit
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
  

dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False
)

mean, std = online_mean_and_std(loader)
print("Mean {}, std {}".format(mean, std))