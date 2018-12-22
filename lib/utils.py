import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class CelebA(Dataset):
    """
    Celeba dataset with labels
    """
    def __init__(self, data_folder, img_transform=None, attr_transform=None):
        super(CelebA, self).__init__()
        self.img_folder = os.path.join(data_folder, "img_align_celeba")
        self.attrs = pd.read_csv(os.path.join(data_folder, "list_attr_celeba.csv"), 
                                 index_col="image_id")
        self.n_attr = self.attrs.shape[1]

        self.filenames = sorted([fn for fn in os.listdir(self.img_folder) if fn.endswith(".jpg")])
        self.img_transform = img_transform
        self.attr_transform = attr_transform

    def __getitem__(self, index):
        sample = Image.open(os.path.join(self.img_folder, self.filenames[index]))
        attr = torch.Tensor(self.attrs.values[index, :].astype(np.float32))
        if self.img_transform is not None:
            sample = self.img_transform(sample)
        if self.attr_transform is not None:
            attr = self.attr_transform(attr)
        return sample, attr

    def __len__(self):
        return len(self.filenames)

    def get_random_attrs(self, num):
        attrs = torch.Tensor(self.attrs.sample(num).values.astype(np.float32))*0.5 + 0.5
        return attrs


class CroppedCelebA(CelebA):
    """
    Celeba dataset with labels cropped and resized
    """
    def __init__(self, data_folder):
        self.h, self.w = 218, 178
        self.size = 64, 64

        img_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.crop((25, 45, self.w-25, self.h-45))),
            transforms.Resize(size=self.size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        super(CroppedCelebA, self).__init__(data_folder=data_folder,
                                            img_transform=img_transform)


class Plotter(object):
    def __init__(self, viz):
        self.viz = viz
        self.n = 0
        self.n_show = 0
        self.losses = defaultdict(list)
        self.temp_losses = defaultdict(list)
        self.images = dict()
        self.losses_window = None
        self.images_windows = dict()
        self.scatters = dict()
        self.scatters_windows = dict()

    def plot(self):
        self.n_show += 1  # number of points to show on the plot
        for k, v in self.images.items():
            self.images_windows[k] = self.viz.images(v, win=self.images_windows.get(k, None))

        for k, v in self.scatters.items():
            self.scatters_windows[k] = self.viz.scatter(
                v,
                win=self.scatters_windows.get(k, None),
                opts=dict(
                    width=300,
                    height=300,
                    title=k
                )
            )

        x = np.linspace(0, self.n, self.n_show)
        y = np.zeros((self.n_show, len(self.temp_losses)))
        names = []
        for i, (k, v) in enumerate(self.temp_losses.items()):
            names.append(k)
            vl = self.losses[k] 
            vl.append(np.mean(v))
            vl = np.array(vl)
            y[:, i] = vl
            v.clear()

        self.losses_window = self.viz.line(
            Y=y,
            X=x,
            win=self.losses_window,
            opts=dict(
                legend=names,
                width=1000,
                height=400,
                title="losses"
            )
        )

    def save_mpl_img(self, path, epoch, n_rows, n_cols, figsize, frame="z_const"):
        imgs = self.images[frame]
        assert n_rows * n_cols == imgs.shape[0]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        for k in range(n_rows*n_cols):
            i = k // n_cols
            j = k % n_cols

            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].cla()
            ax[i, j].imshow(imgs[k].cpu().data.numpy().transpose(1, 2, 0))

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(path)
