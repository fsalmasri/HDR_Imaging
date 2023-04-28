from torchvision import transforms

import os
from os.path import join
import random
from PIL import Image
from torch.utils.data.dataset import Dataset

import cv2
import pandas as pd

outsize = (224, 224)
insize = (224, 224)


def LDR2HDR(img, expo):
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
    return ((img ** GAMMA) * 2.0 ** (-1 * expo)) ** (1 / GAMMA)


class HDRDataset(Dataset):
    def __init__(self, root, phase, excel_path, outsize=(214, 214), input_transformer=None, output_transformer=None,
                 flip_prob=0.5):
        self.root = root
        self.phase = phase
        self.df = None
        self.outsize = outsize
        self.input_transformer = input_transformer
        self.output_transformer = output_transformer
        self.flip_prob = flip_prob

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

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        high_to_ref = Image.open(self.df.iloc[index, 0])
        low_to_ref = Image.open(self.df.iloc[index, 1])
        tif1 = Image.open(self.df.iloc[index, 2])
        tif2 = Image.open(self.df.iloc[index, 3])
        tif3 = Image.open(self.df.iloc[index, 4])

        # gamma_high_to_ref_path = LDR2HDR(cv2.imread(self.df.iloc[index, 0]), self.df.iloc[index, 6])
        gamma_high_to_ref = LDR2HDR(
            cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[index, 0]), cv2.COLOR_BGR2RGB), insize),
            self.df.iloc[index, 6])
        gamma_low_to_ref = LDR2HDR(
            cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[index, 1]), cv2.COLOR_BGR2RGB), insize),
            self.df.iloc[index, 7])
        gamma_tif1 = LDR2HDR(
            cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[index, 2]), cv2.COLOR_BGR2RGB), insize),
            self.df.iloc[index, 8])
        gamma_tif2 = LDR2HDR(
            cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[index, 3]), cv2.COLOR_BGR2RGB), insize),
            self.df.iloc[index, 9])
        gamma_tif3 = LDR2HDR(
            cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[index, 4]), cv2.COLOR_BGR2RGB), insize),
            self.df.iloc[index, 10])

        y_label = cv2.imread(self.df.iloc[index, 5], flags=cv2.IMREAD_ANYDEPTH)
        # y_label          = cv2.cvtColor(y_label, cv2.COLOR_BGR2RGB)
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

        # print("dim1 is ", high_to_ref_path.shape)
        # print("dim2 is ", gamma_high_to_ref_path.shape)
        # print("dim3 is ", torch.concat([high_to_ref_path, gamma_high_to_ref_path]).shape)
        # high_to_ref_path = torch.concat([high_to_ref_path, gamma_high_to_ref_path])
        # low_to_ref_path = torch.concat([low_to_ref_path, gamma_low_to_ref_path])
        # tif1_path = torch.concat([tif1_path, gamma_tif1_path])
        # tif2_path = torch.concat([tif2_path, gamma_tif2_path])
        # tif3_path = torch.concat([tif3_path, gamma_tif3_path])
        imgs = [high_to_ref, low_to_ref, tif1, tif2, tif3,
                gamma_high_to_ref, gamma_low_to_ref, gamma_tif1, gamma_tif2, gamma_tif3,
                y_label]
        flip = transforms.RandomHorizontalFlip(p=1)
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            imgs = [flip(img) for img in imgs]

        return imgs