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

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)

def show_transformed_image(loader):

    for batch in loader:
        batch = [batch[x] for x in batch]
        # batch[-1] = np.log(1 + 5000 * (batch[-1] + 1) / 2.) / np.log(1 + 5000) * 2. - 1

        batch[-1] = np.log(1 + 5000 * batch[-1]) / np.log(1+5000)
        plt.figure()
        for i, f in enumerate(batch):
            print(f.shape)
            # plt.subplot(3,5,i+1)
            # plt.imshow(f.permute(1,2,0).data.numpy())
        exit()
        plt.figure()
        for i, f in enumerate(batch):
            plt.subplot(3, 5, i + 1)
            plt.hist(f.permute(1, 2, 0).data.numpy().flatten(), bins=50)

        plt.show()

    # exit()


show_transformed_image(train_loader)