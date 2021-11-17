from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms
import torch
import os
import numpy as np
import cv2
import Network
import random
import PIL.Image as Image
import datetime
#各个数据集的均值
VGG_MEAN2 = [63.776633, 77.412891, 99.045335 , 65.075620]
VGG_MEAN = [90.930974, 107.563677, 109.988508 , 85.771360]  # RGBN
PAN_MEAN = 92.135761




class gf1_mul_Dataset(Dataset):
    def __init__(self, data_path, transform = None):
        gf1_mul_list = os.listdir(data_path+'/gf1_mul/train')
        self.image_list = []#gf1_mul
        self.label_list = []
        for i in gf1_mul_list:
            path = data_path+'/gf1_mul/train'+'/{}'.format(i)
            name_list = os.listdir(path)
            for _, item in enumerate(name_list):
                self.image_list.append(os.path.join(path,item))
                self.label_list.append(int(i))
        self.transform = transform
        self.len = len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (227, 227))
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.label_list[index]
        return image,label
    def __len__(self):
        return self.len


class gf2_mul_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        gf2_mul_list = os.listdir(data_path+'/gf2_mul/train')
        self.image_list = []  # gf2_mul
        self.label_list = []
        for i in gf2_mul_list:
            path = data_path+'/gf2_mul/train' + '/{}'.format(i)
            name_list = os.listdir(path)
            for _, item in enumerate(name_list):
                self.image_list.append(os.path.join(path, item))
                self.label_list.append(int(i))
        self.transform = transform
        self.len = len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (227, 227))
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.label_list[index]
        return image, label
    def __len__(self):
        return self.len



class gf1_pan_Dataset(Dataset):
    def __init__(self, data_path, transform = None):
        gf1_pan_list = os.listdir(data_path+'/gf1_pan/train')
        self.image_list = []  # gf1_pan
        self.label_list = []
        for i in gf1_pan_list:
            path = data_path+'/gf1_pan/train'+'/{}'.format(i)
            name_list = os.listdir(path)
            for _,item in enumerate(name_list):
                self.image_list.append(os.path.join(path,item))
                self.label_list.append(int(i))
        self.transform = transform
        self.len = len(self.image_list)
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (256, 256))
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.label_list[index]
        return image, label

    def __len__(self):
        return self.len




def get_list(list,data_path):
    image_list = []
    label_list = []

    for i in list:#0123
        path = data_path + '/{}'.format(i)
        name_list = os.listdir(path)
        for _, item in enumerate(name_list):
            image_list.append(os.path.join(path, item))
            label_list.append(int(i))

    return image_list, label_list


def shuffel_list(img_list, img_label):
    randnum = random.randint(0, 1000)
    random.seed(randnum)
    random.shuffle(img_list)
    random.seed(randnum)
    random.shuffle(img_label)


class train_dataset(Dataset):
    def __init__(self, data_path, transform1 = None,transform2 = None):
        gf1_pan_list = os.listdir(data_path+'/gf1_pan/train')
        gf1_pan_list.sort(key = lambda x: int(x))
        gf1_mul_list = os.listdir(data_path+'/gf1_mul/train')
        gf1_mul_list.sort(key = lambda x: int(x))
        gf2_mul_list = os.listdir(data_path + '/gf2_mul/train')
        gf2_mul_list.sort(key = lambda x: int(x))

        self.image1_list, self.label1_list = get_list(gf1_pan_list, data_path + '/gf1_pan/train')
        self.image2_list, self.label2_list = get_list(gf1_mul_list, data_path + '/gf1_mul/train')
        self.image3_list, self.label3_list = get_list(gf2_mul_list, data_path + '/gf2_mul/train')
        self.transform1 = transform1
        self.transform2 = transform2
        self.len = len(self.image1_list)

    def __getitem__(self, index):
        image1 = cv2.imread(self.image1_list[index], cv2.IMREAD_UNCHANGED)
        image1 = cv2.resize(image1, (256, 256))

        label1 = self.label1_list[index]

        image2 = cv2.imread(self.image2_list[index], cv2.IMREAD_UNCHANGED)
        image2 = cv2.resize(image2, (227, 227))
        label2 = self.label2_list[index]

        image3 = cv2.imread(self.image3_list[index], cv2.IMREAD_UNCHANGED)
        image3 = cv2.resize(image3, (227, 227))
        label3 = self.label3_list[index]

        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image3 = Image.fromarray(image3)

        if self.transform2:
            image1 = self.transform2(image1)

        if self.transform1:
            image2 = self.transform1(image2)
            image3 = self.transform1(image3)


        return [image1, image2, image3], (label1, label2, label3)

    def __len__(self):
        return self.len


class validation_dataset(Dataset):
    def __init__(self, data_path, transform1 = None,transform2 = None):
        gf1_pan_list = os.listdir(data_path+'/gf1_pan/val')
        gf1_mul_list = os.listdir(data_path+'/gf1_mul/val')
        gf2_mul_list = os.listdir(data_path + '/gf2_mul/val')

        self.image1_list, self.label1_list = get_list(gf1_pan_list, data_path + '/gf1_pan/val')
        self.image2_list, self.label2_list = get_list(gf1_mul_list, data_path + '/gf1_mul/val')
        self.image3_list, self.label3_list = get_list(gf2_mul_list, data_path + '/gf2_mul/val')
        self.len = len(self.image1_list)
        self.transform1 = transform1
        self.transform2 = transform2
        # self.counter = 0

    def __getitem__(self, index):
        image1 = cv2.imread(self.image1_list[index], cv2.IMREAD_UNCHANGED)
        image1 = cv2.resize(image1, (256, 256))
        label1 = self.label1_list[index]

        image2 = cv2.imread(self.image2_list[index], cv2.IMREAD_UNCHANGED)
        image2 = cv2.resize(image2, (227, 227))
        label2 = self.label2_list[index]

        image3 = cv2.imread(self.image3_list[index], cv2.IMREAD_UNCHANGED)
        image3 = cv2.resize(image3, (227, 227))
        label3 = self.label3_list[index]

        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image3 = Image.fromarray(image3)

        if self.transform1:
            image2 = self.transform1(image2)
            image3 = self.transform1(image3)
        if self.transform2:
            image1 = self.transform2(image1)

        return [image1, image2, image3], (label1, label2, label3)

    def __len__(self):
        return self.len

def init_dataset(path):
    #path = '/media/2T/cc/salayidin/S/gf1gf2'
    train_ds1 = gf1_mul_Dataset(data_path=path)
    # train_loader1 = data.DataLoader(train_ds1, batch_size=64, shuffle=True, num_workers=4)

    train_ds2 = gf2_mul_Dataset(data_path=path)
    # train_loader2 = data.DataLoader(train_ds2, batch_size=64, shuffle=True, num_workers=4)

    train_ds3 = gf1_pan_Dataset(data_path=path)
    # train_loader3 = data.DataLoader(train_ds3, batch_size=64, shuffle=True, num_workers=4)
    datasets = [train_ds1, train_ds2, train_ds3]
    concat_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(concat_dataset, batch_size=1, shuffle=True, num_workers=4)
    return train_loader

if __name__ == '__main__':

    # path = '/media/2T/cc/salayidin/S/gf1gf2'
    path = '/media/2T/cuican/code/Pytorch_RSIR/gf1gf2'
    # tds = train_dataset(data_path=path)
    # transform = transforms.Compose([transforms.Resize(227)])
    tds = gf1_mul_Dataset_beta(data_path=path)
    # vds = validation_dataset(data_path=path)
    train_loader = DataLoader(tds, batch_size=256, shuffle=True, num_workers=4)
    for (img,label) in train_loader:
        img = img.cuda()
        label = label.cuda()
        print("hello")

    print('hello')

