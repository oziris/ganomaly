import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

#PATH = '/media/samo/Workspace/ganomaly_pytorch/data/dl4cv'
PATH = '/workspace/data/dl4cv'

class MyDataset(Dataset):
    def __init__(self):
        self.data = ImageFolder(os.path.join(PATH, 'train'), loader=pil_loader_bw, transform=ToTensor())
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def pil_loader_rgb(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_loader_bw(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')        

def online_mean_and_std(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for x, y in loader: 
        b, c, h, w = x.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(x, dim=[0, 2, 3])
        sum_of_square = torch.sum(x ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
  
def online_mean_and_std_bw(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for x, y in loader: 
        b, c, h, w = x.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(x, dim=[0, 2, 3])
        sum_of_square = torch.sum(x ** 2, dim=[0, 2, 3])
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

mean, std = online_mean_and_std_bw(loader)
print("Mean {}, std {}".format(mean, std))