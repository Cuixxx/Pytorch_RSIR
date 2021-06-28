import argparse
import data
import Network
import math
import os


import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import time

#通过命令行修改超参

parser = argparse.ArgumentParser(description='RSIR')
parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--epoch', type=int, default=64, metavar='epoch',
                    help='epoch')
parser.add_argument('--bits', type=int, default=64, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model2', metavar='P',
                    help='path directory')
args = parser.parse_args()


class my_tensorboarx(object):
    def __init__(self, log_dir, file_name, start_fold_time=0):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.file_name = file_name
        self.epoch = 0
        self.fold_time = start_fold_time
    def draw(self, train_loss, epoch):
        self.epoch = epoch
        self.writer.add_scalars(str(self.file_name), {
            # 'train_acc': train_acc,
            # 'train_prec': train_prec,
            # 'train_rec': train_rec,
            # 'train_f1': train_f1,
            'train_loss': train_loss,
            # 'validation_acc': validation_acc,
            # 'validation_prec': validation_prec,
            # 'validation_rec': validation_rec,
            # 'validation_f1': validation_f1,
        }, self.epoch)


    def close(self):
        self.writer.close()


def init_dataset(path):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)]
    )  # 归一化[-1,1]
    train_ds1 = data.gf1_mul_Dataset(data_path=path, transform=transform)
    train_loader1 = data.DataLoader(train_ds1, batch_size=16, shuffle=True, num_workers=8,drop_last=True)

    train_ds2 = data.gf2_mul_Dataset(data_path=path, transform=transform)
    train_loader2 = data.DataLoader(train_ds2, batch_size=16, shuffle=True, num_workers=8,drop_last=True)

    train_ds3 = data.gf1_pan_Dataset(data_path=path, transform=transform)
    train_loader3 = data.DataLoader(train_ds3, batch_size=16, shuffle=True, num_workers=8,drop_last=True)

    return train_loader1, train_loader2, train_loader3





def loss_function(catlabel,hash_code,gama=5,l = 0.1):#catlabel: 3n*1 hash_code:3n*64
    length = len(hash_code)
    label = torch.zeros(length,4).scatter_(1,catlabel.reshape(-1,1),1)
    label = label.cuda()
    #label = torch.nn.functional.one_hot(torch.tensor(catlabel), num_classes=4)
    A = torch.tensor([torch.matmul(a, a.reshape(-1, 1)) for a in hash_code]).cuda()
    B = torch.matmul(hash_code, hash_code.t()).cuda()
    C = A.expand(length, length).cuda()
    dis_matrix = torch.abs(C+C.t()-2*B)
    # view = (dis_matrix).detach().cpu().numpy()
    mask = torch.triu(torch.ones(length, length), diagonal=1).cuda()#上三角矩阵
    dis_matrix = dis_matrix*mask

    S = torch.matmul(label,label.t()).cuda()
    S_mask = S*mask

    cauchy = lambda x: gama/(x+gama)
    cauchy_matrix1 = cauchy(dis_matrix)*S_mask+(1-S_mask)
    cauchy_matrix1=torch.clamp(cauchy_matrix1, min=0.0001, max=0.9999)
    cauchy_matrix2 = 1-cauchy(dis_matrix*(1-S_mask))+(1-(1-S_mask)*mask)
    cauchy_matrix2 = torch.clamp(cauchy_matrix2, min=0.0001, max=0.9999)
    q_loss = torch.mean((torch.abs(hash_code)-1)*(torch.abs(hash_code)-1))#是hash_code接近-1，1减少舍入误差
    loss = -(torch.sum(torch.log(cauchy_matrix1))+torch.sum(torch.log(cauchy_matrix2)))/(length*(1+length)/2)+l*q_loss
    return loss

def train():
    print('\nEpoch :%d' % epoch)
    train_loss = 0
    #total =0
    with tqdm(total=math.ceil(len(trainloader1)),desc = "training") as pbar:
        for index, ((img1, label1), (img2, label2), (img3, label3)) in enumerate(zip(trainloader1, trainloader2, trainloader3)):
            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            cat_label = torch.cat((label1, label2, label3), dim=0)
            _, hash_code = model(img1, img2, img3)
            loss = loss_function(cat_label, hash_code)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})
            pbar.update(1)
        pbar.close()
        return train_loss/(index+1)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    path = '/media/2T/cuican/code/Pytorch_RSIR/gf1gf2'
    trainloader1, trainloader2, trainloader3 = init_dataset(path)

    model = Network.MyModel()
    # model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    start_epoch = 0
    now = time.strftime("%m-%d-%H:%M", time.localtime(time.time()))
    model_name = now+'_RSIR'
    tensorboard = my_tensorboarx(log_dir='./tensorboard_data', file_name=model_name)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        loss = train()
        scheduler.step(epoch)
        if (epoch+1) % 8 == 0:
            print('saved!')
            if not os.path.isdir('./models/{}'.format(model_name)):
                os.mkdir('./models/{}'.format(model_name))
            torch.save(model.state_dict(), './models/{}/{}.pth.tar'.format(model_name, epoch))
        # if (epoch+1)%10 == 0:
        #     mAP = evaluate()
        tensorboard.draw(train_loss=loss, epoch=epoch)
    tensorboard.close()


    # for index, ((img1, label1), (img2, label2), (img3, label3)) in enumerate(zip(trainloader1, trainloader2, trainloader3)):
    #     spacial_img, label = get_spacialinf(img1, img2, label1, label2)
    #     spectral_vector = get_spectralvector(img1, img2)
    #     img_pan = torch.unsqueeze(img3, dim=1).float()  # 256 * 1 * 256 * 256
    #
    #     pan_vector = net3(img_pan.cuda())
    #     spacial_vector = net1(spacial_img.cuda())
    #     mixed_vector = net2(spectral_vector.cuda(), spacial_vector.cuda())
    #     cat_vector = torch.cat((mixed_vector, pan_vector), dim=0)
    #     cat_label = (*label, *label3)
    #     hash_code = hash_function(cat_vector)
    #     optimizer4nn = torch.optim.SGD(itertools.chain(net1.parameters(), net2.parameters(), net3.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[args.epoch], gamma=0.1)
    #     #loss_function(cat_label, hash_code)
    #     start_epoch = 1
    #     if args.pretrained:
    #         print('none')
    #         # net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
    #         # test()
    #     else:
    #         #if os.path.isdir('{}'.format(args.path)):
    #         #    shutil.rmtree('{}'.format(args.path))
    #         for epoch in range(start_epoch, start_epoch + args.epoch):
    #             train(epoch)
    #             scheduler.step(epoch)