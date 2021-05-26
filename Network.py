import torch
import os
import torch.nn as nn

from torchvision import models
import numpy as np
import random
os.environ['TORCH_HOME'] = 'models'#指定预训练模型下载地址

alexnet_model = models.alexnet(pretrained=True)
print(alexnet_model)
# #


class SpacialNet(nn.Module):
    def __init__(self):
        super(SpacialNet, self).__init__()
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.Linear = nn.Sequential(nn.Linear(9216, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,1024),
                                    nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        return x

class SpacialNet_Pan(nn.Module):
    def __init__(self):
        super(SpacialNet_Pan, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=3, padding=3),
                                      nn.ReLU(),
                                      nn.AvgPool2d(3, stride=2),
                                      nn.Conv2d(32, 64, kernel_size=5, stride=2),
                                      nn.ReLU(),
                                      nn.AvgPool2d(3, stride=1, padding=1),
                                      nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, kernel_size=5, stride=2),
                                      nn.ReLU(),
                                      nn.AvgPool2d(3, stride=1))
        self.Linear = nn.Sequential(nn.Linear(9216, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 1024),
                                     nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        return x

class MixNet(nn.Module):
    def __init__(self):
        super(MixNet, self).__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv1d(256, 256, kernel_size=64, stride=1),
                                      nn.ReLU())
        self.Linear1 = nn.Sequential(nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU())
        self.Linear2 = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
    def forward(self, x, spacial_feature):
        x = self.features(x)
        x = torch.sum(x, dim=2)
        spectral_feature = self.Linear1(x)
        cat_feature = torch.cat((spectral_feature, spacial_feature), dim=1)
        mixed_feature = self.Linear2(cat_feature)
        return mixed_feature#n*1024

def get_spacialinf(img1, img2):
    spacial_img = torch.cat((img1, img2), dim=0)
    spacial_img = spacial_img[:, :, :, 0:3]
    spacial_img = spacial_img.permute(0, 3, 1, 2).float()
    return spacial_img

def get_spectralvector(img1, img2):
    spectral_vector1 = torch.tensor(RandomSelectVector(img1.cpu(), 5))
    spectral_vector2 = torch.tensor(RandomSelectVector(img2.cpu(), 5))
    spectral_vector1 = spectral_vector1.cuda()
    spectral_vector2 = spectral_vector2.cuda()
    spectral_vector = torch.cat((spectral_vector1, spectral_vector2), dim=0)
    spectral_vector = torch.unsqueeze(spectral_vector, dim=1).float()
    return spectral_vector

def RandomSelectVector(image,num):
    _, w, h, _ = image.shape
    vector_list = []
    vector = np.zeros([4 * 4 * num])
    for img in image:
        for n in range(4):  # 把图像分成分成4个块
            for m in range(num):  # 每个块随机选择num个点
                i = random.randint(0, int(w / 2))
                j = random.randint(0, int(h / 2))
                if n ==0:
                    vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[i, j, :]
                elif n ==1:
                    vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[i+int(w/2), j, :]
                elif n ==2:
                    vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[i, j+int(h/2), :]
                elif n ==3:
                    vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[i+int(w/2), j+int(h/2), :]
        vector_list.append(vector)
    return vector_list

class MyModel15(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = SpacialNet()
        self.net2 = MixNet()
        self.net3 = SpacialNet_Pan()
        self.hash = nn.Sequential(nn.Linear(1024, 64), nn.Tanh()).cuda()

    def forward(self, img1, img2, img3):
        spacial_img = get_spacialinf(img1, img2)
        spectral_vector = get_spectralvector(img1, img2)
        img_pan = torch.unsqueeze(img3, dim=1).float()  # 256 * 1 * 256 * 256


        spacial_vector = self.net1(spacial_img)
        mixed_vector = self.net2(spectral_vector, spacial_vector)
        pan_vector = self.net3(img_pan)
        cat_vector = torch.cat((mixed_vector, pan_vector), dim=0)
        hash_code = self.hash(cat_vector)
        return cat_vector, hash_code#(-1,1)

    # def forward(self, spacial_img, spectral_vector, img_pan, state):
    #     if state == 0: #训练
    #         spacial_vector = self.net1(spacial_img)
    #         mixed_vector = self.net2(spectral_vector, spacial_vector)
    #         pan_vector = self.net3(img_pan)
    #         cat_vector = torch.cat((mixed_vector, pan_vector), dim=0)
    #         hash_code = self.hash(cat_vector)
    #     elif state == 1: #校验输入mul
    #         spacial_vector = self.net1(spacial_img)
    #         mixed_vector = self.net2(spectral_vector, spacial_vector)
    #         hash_code = self.hash(mixed_vector)
    #     elif state == 2:#校验输入pan
    #         pan_vector = self.net3(img_pan)
    #         hash_code = self.hash(pan_vector)
    #     return hash_code#(-1,1)


if __name__ == '__main__':
    model = SpacialNet()
    print(model)
    model1 = SpacialNet_Pan()
    print(model1)
    model2 = MixNet()
    print(model2)
    data_path = '/media/2T/cc/salayidin/S/gf1gf2/gf1_mul'
    directory = os.listdir(data_path)
    print(directory)
