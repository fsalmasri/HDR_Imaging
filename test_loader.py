import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision
from torchvision import transforms


from dloader2 import HDRDataset
from config import get_configs


args = get_configs()

input_transformer = transforms.Compose(
    [transforms.Resize(args.insize),
     transforms.ToTensor()])

output_transformer = transforms.Compose([transforms.ToTensor()])


train_dataset = HDRDataset(args.train_ds_path, "train", args.train_xsl_path, args.outsize, args.insize, input_transformer,
                           output_transformer, flip_prob=0.5)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

def show_transformed_image(loader):
    batch = next(iter(loader))
    all_imgs = torch.concat(batch)
    for i, f in enumerate(all_imgs):
        print(f.max(), f.min())

    grid = torchvision.utils.make_grid(all_imgs, nrow=6)
    plt.figure()
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

show_transformed_image(train_loader)