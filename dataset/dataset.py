from PIL import Image
import os
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

path_image = './dataset/CXR8/images/images'

train_split_path = "./dataset/CXR8/train.txt"
test_plit_path = "./dataset/CXR8/val.txt"
val_split_path = "./dataset/CXR8/val.txt"

train_df_path = "./dataset/train_binary.csv"
val_df_path = "./dataset/val_binary.csv"

# train
train_df = pd.read_csv(train_df_path)

y0z0 = train_df[(train_df["Target"] == 0) & (train_df["Sex"] == "F")]
y1z1 = train_df[(train_df["Target"] == 1) & (train_df["Sex"] == "M")]

y0z1 = train_df[(train_df["Target"] == 0) & (train_df["Sex"] == "M")]
y1z0 = train_df[(train_df["Target"] == 1) & (train_df["Sex"] == "F")]

# Make sure all envs have the same number of samples
n = min(len(y0z0), len(y1z1), len(y0z1), len(y1z0))
y0z0 = y0z0.sample(n=n)
y1z1 = y1z1.sample(n=n)
y0z1 = y0z1.sample(n=n)
y1z0 = y1z0.sample(n=n)

train_df_e1 = pd.concat([y0z0, y1z1])
train_df_e2 = pd.concat([y0z1, y1z0])


class NIH_inf(Dataset):
    def __init__(self, mode='train', data_len=None):

        self.mode = mode
        self.input_size = 224
        if mode == 'train':
            print('Loading the training data...')
            train_df = pd.read_csv(train_df_path)
            train_df = train_df[train_df["Sex"] == "M"]
            train_df_size = len(train_df)
            print("Train_df path", train_df_size)
            self.dataframe = train_df

        elif mode == 'test':
            print('Loading the test data...')
            val_df = pd.read_csv(val_df_path)
            val_df = val_df[val_df["Sex"] == "M"] #
            val_df_size = len(val_df)
            print("Validation_df path", val_df_size)
            self.dataframe = val_df

        self.dataset_size = len(self.dataframe)
        self.path_image = path_image


    def __getitem__(self, index):
        item = self.dataframe.iloc[index]

        image_path = os.path.join(self.path_image, item["Image Index"])
        img = Image.open(image_path).convert('RGB')

        target = int(item["Target"])

        if self.mode == 'train':
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.25, contrast=0.25)(img)
            img = transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            #img = transforms.Scale(self.input_size)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, item["Image Index"] #, img_name


    def __len__(self):
        return len(self.dataframe)


class NIH_e1(Dataset):
    def __init__(self, mode='train', data_len=None):


        self.mode = mode
        self.input_size = 224
        if mode == 'train':
            print('Loading the training data...')
            #train_df = pd.read_csv(train_df_path)
            train_df_size = len(train_df_e1)
            print("Train_df path", train_df_size)
            self.dataframe = train_df_e1    #pd.concat([train_df_e1, train_df_e2])

        elif mode == 'test':
            print('Loading the test data...')
            val_df = pd.read_csv(val_df_path)
            #val_df = val_df[val_df["Sex"] == "F"]

            y0z0 = val_df[(val_df["Target"] == 0) & (val_df["Sex"] == "F")]
            y1z1 = val_df[(val_df["Target"] == 1) & (val_df["Sex"] == "M")]

            y0z1 = val_df[(val_df["Target"] == 0) & (val_df["Sex"] == "M")]
            y1z0 = val_df[(val_df["Target"] == 1) & (val_df["Sex"] == "F")]

            n = min(len(y0z0), len(y1z1), len(y0z1), len(y1z0))
            y0z0 = y0z0[:n] #.sample(n=n)
            y1z1 = y1z1[:n] #.sample(n=n)
            y0z1 = y0z1[:n] #.sample(n=n)
            y1z0 = y1z0[:n] #.sample(n=n)


            val_df = pd.concat([y0z0, y1z1, y0z0, y1z1])   # y0z1, y1z0
            val_df = val_df[val_df["Sex"] == "F"]

            val_df_size = len(val_df)
            print("Validation_df path", val_df_size)
            self.dataframe = val_df

        self.dataset_size = len(self.dataframe)
        self.path_image = path_image


    def __getitem__(self, index):
        item = self.dataframe.iloc[index]

        image_path = os.path.join(self.path_image, item["Image Index"])
        img = Image.open(image_path).convert('RGB')

        target = int(item["Target"])

        if self.mode == 'train':
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.25, contrast=0.25)(img)
            img = transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            #img = transforms.Scale(self.input_size)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, item["Sex"] #, img_name


    def __len__(self):
        return len(self.dataframe)

class NIH_e2(Dataset):
    def __init__(self, mode='train', data_len=None):


        self.mode = mode
        self.input_size = 224
        if mode == 'train':
            print('Loading the training data...')
            #train_df = pd.read_csv(train_df_path)
            train_df_size = len(train_df_e2)
            print("Train_df path", train_df_size)
            self.dataframe = train_df_e2

        elif mode == 'test':
            print('Loading the test data...')
            val_df = pd.read_csv(val_df_path)
            val_df_size = len(val_df)
            print("Validation_df path", val_df_size)
            self.dataframe = val_df

        self.dataset_size = len(self.dataframe)
        self.path_image = path_image


    def __getitem__(self, index):
        item = self.dataframe.iloc[index]

        image_path = os.path.join(self.path_image, item["Image Index"])
        img = Image.open(image_path).convert('RGB')

        target = int(item["Target"])

        if self.mode == 'train':
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.25, contrast=0.25)(img)
            img = transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            #img = transforms.Scale(self.input_size)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, item["Sex"] #, img_name


    def __len__(self):
        return len(self.dataframe)

class NIH_erm(Dataset):
    def __init__(self, mode='train', data_len=None):

        self.mode = mode
        self.input_size = 224
        if mode == 'train':
            print('Loading the training data...')
            train_df = pd.concat([train_df_e1, train_df_e2])
            train_df_size = len(train_df)
            print("Train_df path", train_df_size)
            self.dataframe = train_df

        elif mode == 'test':
            print('Loading the test data...')
            val_df = pd.read_csv(val_df_path)
            val_df_size = len(val_df)
            print("Validation_df path", val_df_size)
            self.dataframe = val_df

        self.dataset_size = len(self.dataframe)
        self.path_image = path_image


    def __getitem__(self, index):
        item = self.dataframe.iloc[index]

        image_path = os.path.join(self.path_image, item["Image Index"])
        img = Image.open(image_path).convert('RGB')

        target = int(item["Target"])

        if self.mode == 'train':
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.25, contrast=0.25)(img)
            img = transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            #img = transforms.CenterCrop((800, 800))(img)
            img = transforms.Resize((256, 256), Image.BILINEAR)(img)
            #img = transforms.Scale(self.input_size)(img)
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, item["Sex"] #, img_name


    def __len__(self):
        return len(self.dataframe)
