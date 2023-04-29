import os
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt

import cv2
import imageio
imageio.plugins.freeimage.download()
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


def LDR2HDR(img, expo):
    img = np.array(img) / 255.0
    GAMMA = 2.2
    # print("img shape is ", img.shape)
    # expo =np.power(2.0,float(expo))
    # expo=np.float(expo)
    # if expo==0:
    # expo=0.0001
    # ((img ** GAMMA) * 2.0**(-1*expo))**(1/GAMMA)
    # ((((img+1)/2.)**GAMMA / expo) *2.-1)
    # img=np.power(img, 1.0/GAMMA)
    # invGamma = 1 / GAMMA
    # table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    # table = np.array(table, np.uint8)
    # return cv2.LUT(img,table)
    return (((img ** GAMMA) * 2.0 ** (-1 * expo)) ** (1 / GAMMA)).astype(np.float32)


class HDRDataset(Dataset):
    def __init__(self, root, phase, excel_path, outsize=(214, 214), insize=(214,214), input_transformer=None, output_transformer=None,
                 flip_prob=0.5):
        self.root = root
        self.phase = phase
        self.df = None
        self.outsize = outsize
        self.insize = insize
        self.input_transformer = input_transformer
        self.output_transformer = output_transformer
        self.flip_prob = flip_prob

        if not os.path.isfile(excel_path):
            dicts_lst = []
            if phase == 'train' or phase == 'test':
                subdirs = [join(root,f) for f in os.listdir(root)]
                # files = [join(y,x) for y in subdirs for x in os.listdir(y)]

                for dir in subdirs:
                    files = [join(dir, f) for f in os.listdir(dir)]
                    files = sorted(files)
                    dict = {'TIF': []}
                    for fn in files:
                        if 'lowtoref.png' in fn.lower(): dict['low_to_ref_img'] = fn
                        if 'hightoref' in fn.lower(): dict['high_to_ref_img'] = fn
                        if '.tif' in fn.lower(): dict['TIF'].append(fn)
                        if '.hdr' in fn.lower(): dict['HDR'] = fn
                        if '.txt' in fn.lower(): dict['gammas'] = fn

                    # //TODO check
                    # if high_to_ref_img is None or low_to_ref_img is None or tif1_img is None or tif2_img is None or tif3_img is None or output_img is None or gamma_dims is None:
                    #     continue

                    with open(dict['gammas']) as file:
                        lines = [float(line.rstrip()) for line in file]

                    # //TODO Check
                    if len(lines) != 5:
                        break
                    else:
                        dict['expos'] = lines

                    dicts_lst.append(dict)

                self.df = pd.DataFrame(dicts_lst)
                self.df.to_excel(excel_path, index=None)
        else:
            self.df = pd.read_excel(excel_path, index_col=None)

    def __len__(self):
        return 5 #len(self.df.index)

    def __getitem__(self, index):

        high_to_ref = Image.open(self.df.iloc[index]['high_to_ref_img'])
        low_to_ref = Image.open(self.df.iloc[index]['low_to_ref_img'])


        tif_lst = self.df.iloc[index]['TIF'].replace("'",'').strip('][').split(', ')
        tif1 = Image.open(tif_lst[0])
        tif2 = Image.open(tif_lst[1])
        tif3 = Image.open(tif_lst[2])

        expo_lst = self.df.iloc[index]['expos'].strip('][').split(', ')

        gamma_high_to_ref = LDR2HDR(high_to_ref.resize(self.insize), float(expo_lst[0]))
        gamma_low_to_ref = LDR2HDR(low_to_ref.resize(self.insize), float(expo_lst[1]))
        gamma_tif1 = LDR2HDR(tif1.resize(self.insize), float(expo_lst[2]))
        gamma_tif2 = LDR2HDR(tif2.resize(self.insize), float(expo_lst[3]))
        gamma_tif3 = LDR2HDR(tif3.resize(self.insize), float(expo_lst[4]))

        y_label = imageio.imread(self.df.iloc[index]['HDR'], format='HDR-FI')
        y_label = cv2.resize(y_label, dsize=self.outsize)


        if self.input_transformer:
            high_to_ref = self.input_transformer(high_to_ref)
            low_to_ref = self.input_transformer(low_to_ref)
            tif1 = self.input_transformer(tif1)
            tif2 = self.input_transformer(tif2)
            tif3 = self.input_transformer(tif3)

        if self.output_transformer:
            y_label = self.output_transformer(y_label)
            gamma_high_to_ref = self.output_transformer(gamma_high_to_ref)
            gamma_low_to_ref = self.output_transformer(gamma_low_to_ref)
            gamma_tif1 = self.output_transformer(gamma_tif1)
            gamma_tif2 = self.output_transformer(gamma_tif2)
            gamma_tif3 = self.output_transformer(gamma_tif3)

        # flip = transforms.RandomHorizontalFlip(p=1)
        # if self.flip_prob > 0 and random.random() < self.flip_prob:
        #     imgs = [flip(img) for img in imgs]

        return {"H2R": high_to_ref, "L2R": low_to_ref,
                "TIF1": tif1, "TIF2": tif2, "TIF3": tif3,
                "GH2R": gamma_high_to_ref, "GL2R": gamma_low_to_ref,
                "GTIF1": gamma_tif1, "GTIF2": gamma_tif2, "GTIF3": gamma_tif3, "GT": y_label}
