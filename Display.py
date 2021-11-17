import cv2
import numpy as np
import os
import torch.nn as nn
import Network
import torch
from torchvision import transforms
import PIL.Image as Image
import json

class Display():
    def __init__(self,path):
        self.gf1mul_net = Network.gf1mulNet().cuda()
        self.gf1mul_net.load_state_dict(torch.load(path), strict=False)
        self.gf2mul_net = Network.gf2mulNet().cuda()
        self.gf2mul_net.load_state_dict(torch.load(path), strict=False)
        self.gf1pan_net = Network.gf1panNet().cuda()
        self.gf1pan_net.load_state_dict(torch.load(path), strict=False)
        self.switch = {1: self.gf1mul_net,2: self.gf2mul_net,3: self.gf1pan_net}
        self.hash_list = torch.load('./result/train_binary').cpu().numpy()
        self.hash_list = np.asarray(self.hash_list, np.int32)
        # self.hash_list = np.concatenate((self.hash_list[0], self.hash_list[1], self.hash_list[2]), axis=0)
        self.path_list = np.load('./result/Tpath.npy')

        # self.path_list = np.concatenate((self.path_list[0], self.path_list[1], self.path_list[2]), axis=0)
        for i,path_list in enumerate(self.path_list):
            for j, item in enumerate(path_list):
                name = item.split('.')[0].split('/')
                name = name[7:]
                name = os.path.join('/static/new/','/'.join(name))+'.jpg'
                self.path_list[i][j] = name

    def run(self, img, input, output,path):
        if input == 3:
            norm_mean = [0.5]
            norm_std = [0.5]
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)]
            )  # 归一化[-1,1]
        else:
            norm_mean = [0.5, 0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5, 0.5]
            transform = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)]
            )  # 归一化[-1,1]
        img = Image.fromarray(img)
        img = transform(img).cuda()
        _, hash_code = self.switch.get(input)(img.unsqueeze(dim=0))
        hash_code[hash_code > 0] = 1
        hash_code[hash_code < 0] = 0
        hash_code = hash_code.cpu().detach().numpy()
        query_result = np.count_nonzero(hash_code != self.hash_list[output-1], axis=1)
        result = self.path_list[output-1][np.argsort(query_result)]
        result = list(result[0:2000])
        result.append('/'+path)
        result = dict(zip(range(len(result)), result))
        j_result = json.dumps(result)
        # print(hash_code)
        return j_result
if __name__ == '__main__':
    path = './models/06-28-15:10_RSIR/63.pth.tar'
    display = Display(path)
    img = cv2.imread('/media/2T/cuican/code/Pytorch_RSIR/gf1gf2/gf1_mul/val/0/118.tif', cv2.IMREAD_UNCHANGED)
    display.run(img, 'gf1mul')