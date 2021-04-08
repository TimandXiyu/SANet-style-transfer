import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torchvision import transforms


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        # transforms.RandomCrop(512),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def eval_transform():
    transform_list = [
        transforms.Resize(size=(1024, 1024)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset2(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset2, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.mask_paths = list(map(lambda x: Path(x), self.paths))
        self.mask_paths = list(map(lambda x: Path(self.root).parent / 'train_mask' / Path(str(x.stem) + '.png'), self.mask_paths))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset2'

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'