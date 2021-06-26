import torch
import os
import torch.nn as nn

from torchvision import models
import numpy as np
import random
os.environ['TORCH_HOME'] = 'models'#指定预训练模型下载地址

alexnet_model = models.alexnet(pretrained=True)
# print(alexnet_model)
# #


class SpacialNet(nn.Module):
    def __init__(self):
        super(SpacialNet, self).__init__()
        self.features = nn.Sequential(*list(alexnet_model.features.children()))

        self.FC = nn.Sequential(nn.Linear(9216, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU())

    def forward(self, x):
        #空间特征
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)

        return x

class SpacialNet_Pan(nn.Module):
    def __init__(self):
        super(SpacialNet_Pan, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, stride=1, padding=0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0)
        )
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size = 3, stride = 1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2,padding=0),#128*128*64
                                      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),#64*64*128
                                      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),#32*32*256
                                      self.block1,
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),# 16*16*512
                                      self.block2,
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(16, stride=1, padding=0)# 1*1*1024
                                      )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        return x

"""
class SpacialNet_Pan(nn.Module):
    def __init__(self):
        super(SpacialNet_Pan, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size = 7, stride = 3, padding=3),
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

"""


class Spectral_Net(nn.Module):
    def __init__(self):
        super(Spectral_Net, self).__init__()
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
        # self.Linear2 = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
    def RandomSelectVector(self,image,num):
        _, w, h, _ = image.shape
        vector_list = []
        vector = np.zeros([4 * 4 * num])
        for img in image:
            for n in range(4):  # 把图像分成分成4个块
                for m in range(num):  # 每个块随机选择num个点
                    i = random.randint(0, int(w / 2))
                    j = random.randint(0, int(h / 2))
                    if n == 0:
                        vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[:, i, j]
                    elif n == 1:
                        vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[:, i + int(w / 2), j]
                    elif n == 2:
                        vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[:, i, j + int(h / 2)]
                    elif n == 3:
                        vector[(4 * n + m) * 4:(4 * n + m + 1) * 4] = img[:, i + int(w / 2), j + int(h / 2)]
            vector_list.append(vector)
        return vector_list

    def forward(self, img):
        x = torch.tensor(self.RandomSelectVector(img.cpu(), 5)).cuda()
        x = self.features(x.unsqueeze(dim=1).float())
        x = torch.sum(x, dim=2)
        spectral_feature = self.Linear1(x)
        # cat_feature = torch.cat((spectral_feature, spacial_feature), dim=1)
        # mixed_feature = self.Linear2(cat_feature)
        # return mixed_feature#n*1024
        return spectral_feature

# def get_spacialinf(img1, img2):
#     spacial_img = torch.cat((img1, img2), dim=0)
#     spacial_img = spacial_img[:, :, :, 0:3]
#     spacial_img = spacial_img.permute(0, 3, 1, 2).float()
#     return spacial_img
#
# def get_spectralvector(img1, img2):
#     spectral_vector1 = torch.tensor(RandomSelectVector(img1.cpu(), 5))
#     spectral_vector2 = torch.tensor(RandomSelectVector(img2.cpu(), 5))
#     spectral_vector1 = spectral_vector1.cuda()
#     spectral_vector2 = spectral_vector2.cuda()
#     spectral_vector = torch.cat((spectral_vector1, spectral_vector2), dim=0)
#     spectral_vector = torch.unsqueeze(spectral_vector, dim=1).float()
#     return spectral_vector

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spacial_Net = SpacialNet()
        self.Spectral_Net1 = Spectral_Net()
        self.Spectral_Net2 = Spectral_Net()
        self.PAN_Net = SpacialNet_Pan()
        self.mixlayer = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        self.hash = nn.Sequential(nn.Linear(1024, 64), nn.Tanh()).cuda()

    def forward(self, img1, img2, img3):#img1 GF1 mul,img2 GF2 mul,img3 GF1 pan

        spacial_feat1 = self.Spacial_Net(img1[:,0:3,:,:])
        spectral_feat1 = self.Spectral_Net1(img1)
        mix_feat1 = self.mixlayer(torch.cat((spacial_feat1, spectral_feat1), dim=1))

        spacial_feat2 = self.Spacial_Net(img2[:, 0:3, :, :])
        spectral_feat2 = self.Spectral_Net1(img2)
        mix_feat2 = self.mixlayer(torch.cat((spacial_feat2, spectral_feat2), dim=1))

        # img_pan = torch.unsqueeze(img3, dim=1).float()  # 256 * 1 * 256 * 256
        pan_feat = self.PAN_Net(img3)

        cat_vector = torch.cat((mix_feat1, mix_feat2, pan_feat), dim=0)
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
    model = MyModel()
    # print(model)