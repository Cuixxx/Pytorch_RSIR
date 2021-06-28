import argparse
import os
from timeit import time

import numpy as np
import torch
import torch.optim.lr_scheduler
from tqdm import tqdm
import data
import Network
import matplotlib.pyplot as plt
import cv2
parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=float, default=1, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=64, metavar='bts',
                    help='binary bits')
parser.add_argument('--model', type=str, default='./models/03-22-22:05_RSIR/63.pth.tar', metavar='bts',
                    help='model path')
args = parser.parse_args()


def load_data(path):
    trainset = data.train_dataset(path)
    tpath_list = np.stack((trainset.image2_list,trainset.image3_list,trainset.image1_list),0)#gf1_mul,gf2_mul,gf1_pan,
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=False, num_workers=4)

    valset = data.validation_dataset(path)
    vpath_list = np.stack((valset.image2_list,valset.image3_list,valset.image1_list),0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                             shuffle=False, num_workers=4)#100
    return trainloader, valloader,tpath_list,vpath_list

def catfunc(hashlike_code,label,vector,input1,input2,input3):
    hashlike_code = torch.cat((hashlike_code, input1), 0)
    label = torch.cat((label, input2), 0)
    vector = torch.cat((vector, input3), 0)

    hashlike_code[hashlike_code <= 0] = 0
    hashlike_code[hashlike_code > 0] = 1
    return hashlike_code, label, vector

def binary_output(dataloader, model_path, model=None):
    #net.load_state_dict(torch.load('./model/{}'.format(8)))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("Use device: " + str(device))
    #net.to(device)
    if model == None:
        model = Network.MyModel15()
        model = model.cuda()
        model.load_state_dict(torch.load(model_path))

    gf1_mul_hashcode = torch.cuda.FloatTensor()
    gf1_mul_label = torch.LongTensor()
    gf1_mul_vector = torch.cuda.FloatTensor()
    gf2_mul_hashcode = torch.cuda.FloatTensor()
    gf2_mul_label = torch.LongTensor()
    gf2_mul_vector = torch.cuda.FloatTensor()
    gf1_pan_hashcode = torch.cuda.FloatTensor()
    gf1_pan_label = torch.LongTensor()
    gf1_pan_vector = torch.cuda.FloatTensor()

    # full_batch_output = torch.cuda.FloatTensor()
    # full_batch_label = torch.cuda.LongTensor()
    # full_batch_vector = torch.cuda.FloatTensor()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            img1 = imgs[1]#gf1_mul
            label1 = labels[1]
            img2 = imgs[2]#gf2_mul
            label2 = labels[2]
            img3 = imgs[0]#gf1_pan
            label3 = labels[0]

            img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
            vectors, outputs = model(img1, img2, img3)

            # gf1_mul_hashcode = torch.cat((gf1_mul_hashcode, outputs[:100].data), 0)
            # gf1_mul_label = torch.cat((gf1_mul_label, label1.data), 0)
            # gf1_mul_vector = torch.cat((gf1_mul_vector, vectors[:100].data), 0)
            #
            # gf1_mul_hashcode[gf1_mul_hashcode <= 0] = 0
            # gf1_mul_hashcode[gf1_mul_hashcode > 0] = 1

            gf1_mul_hashcode, gf1_mul_label, gf1_mul_vector = catfunc(gf1_mul_hashcode, gf1_mul_label, gf1_mul_vector, outputs[:100].data, label1.data,
                    vectors[:100].data)
            gf2_mul_hashcode, gf2_mul_label, gf2_mul_vector = catfunc(gf2_mul_hashcode, gf2_mul_label, gf2_mul_vector, outputs[100:200].data, label2.data,
                    vectors[100:200].data)
            gf1_pan_hashcode, gf1_pan_label, gf1_pan_vector = catfunc(gf1_pan_hashcode, gf1_pan_label, gf1_pan_vector, outputs[200:300].data, label3.data,
                    vectors[200:300].data)

            # gf2_mul_hashcode = torch.cat((gf2_mul_hashcode, outputs[100:200].data), 0)
            # gf2_mul_label = torch.cat((gf2_mul_label, label2.data), 0)
            # gf2_mul_vector = torch.cat((gf2_mul_vector, vectors[100:200].data), 0)
            #
            # gf2_mul_hashcode[gf2_mul_hashcode <= 0] = 0
            # gf2_mul_hashcode[gf2_mul_hashcode > 0] = 1
            #
            #
            #
            # gf1_pan_hashcode = torch.cat((gf1_pan_hashcode, outputs[200:300].data), 0)
            # gf1_pan_label = torch.cat((gf1_pan_label, label3.data), 0)
            # gf1_pan_vector = torch.cat((gf1_pan_vector, vectors[200:300].data), 0)
            #
            # gf1_pan_hashcode[gf1_pan_hashcode <= 0] = 0
            # gf1_pan_hashcode[gf1_pan_hashcode > 0] = 1

            # label = torch.cat((label1, label2, label3), dim=0)#3*100
            # label = label.cuda()
            # full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            # full_batch_vector = torch.cat((full_batch_vector, vectors.data), 0)
            # full_batch_label = torch.cat((full_batch_label, label.data), 0)
            # full_batch_output[full_batch_output < 0] = 0
            # full_batch_output[full_batch_output > 0] = 1

        # full_batch_label = full_batch_label.cpu().numpy()
        # n1 = np.sum(np.equal(0, full_batch_label).astype(int))
        # n2 = np.sum(np.equal(1, full_batch_label).astype(int))
        # n3 = np.sum(np.equal(2, full_batch_label).astype(int))
        # n4 = np.sum(np.equal(3, full_batch_label).astype(int))
        # return full_batch_output, full_batch_label, full_batch_vector
        return gf1_mul_hashcode, gf1_mul_label, gf1_mul_vector,\
               gf2_mul_hashcode, gf2_mul_label, gf2_mul_vector,\
               gf1_pan_hashcode, gf1_pan_label, gf1_pan_vector

def evaluate(trn_binary, trn_label, tst_binary, tst_label, K=10):
    classes = np.max(tst_label) + 1
    #把三个数据集数据合成一个
    trn_binary = np.concatenate((trn_binary[0], trn_binary[1], trn_binary[2]), axis=0)
    trn_label = np.concatenate((trn_label[0], trn_label[1], trn_label[2]), axis=0)
    tst_binary = np.concatenate((tst_binary[0], tst_binary[1], tst_binary[2]), axis=0)
    tst_label = np.concatenate((tst_label[0], tst_label[1], tst_label[2]), axis=0)

    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i+1).permutation(np.where(tst_label == i)[0])[:100]]
            tst_sample_label = np.array([i]).repeat(100)
            continue
        else:
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[np.random.RandomState(seed=i+1).permutation(np.where(tst_label==i)[0])[:100]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(100)])

    query_times = tst_sample_binary.shape[0]#10*100
    trainset_len = trn_binary.shape[0]#50000
    AP = np.zeros(query_times)#一次检索一个AP
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    recall = np.zeros(trainset_len)
    total_time_start = time.time()
    with tqdm(total=query_times, desc="Query") as pbar:
        for i in range(query_times):
            query_label = tst_sample_label[i]
            query_binary = tst_sample_binary[i, :]
            query_result = np.count_nonzero(query_binary != trn_binary, axis=1) #haming distance   # don't need to divide binary length
            sort_indices = np.argsort(query_result)#np.argsort从小到大排序返回索引
            K_sort = sort_indices[0:K]#取前K个
            buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
            x = np.stack((np.sort(query_result),buffer_yes),axis=0)
            n = np.sum(buffer_yes)#9400*3
            P = np.cumsum(buffer_yes) / Ns #累计求和返回数组
            # recall = np.cumsum(buffer_yes)/sum(buffer_yes)
            precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0]-1]
            AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
            sum_tp = sum_tp + np.cumsum(buffer_yes)
            recall = recall + np.cumsum(buffer_yes)/sum(buffer_yes)
            pbar.set_postfix({'Average Precision': '{0:1.5f}'.format(AP[i])})
            pbar.update(1)
    pbar.close()

    precision_at_k = sum_tp / Ns / query_times
    recall = recall / query_times

    plt.plot(recall, precision_at_k)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('rec')
    plt.ylabel('pre')
    plt.title('PR-curve')

    plt.plot()
    plt.savefig('./fig1.png',dpi=300)
    plt.show()

    index = [100, 500, 1000, 2000, 5000]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)
    print('Total query time:', time.time() - total_time_start)

def Display(trn_binary, trn_label,trn_vector, tst_binary, tst_label, tst_vector):
    classes = np.max(tst_label) + 1
    for i in range(classes):
        tst_sample_binary = tst_binary[0][
            np.random.RandomState(seed=10).permutation(np.where(tst_label[0] == i)[0])[0]]
        tst_sample_vector = tst_vector[0][
            np.random.RandomState(seed=10).permutation(np.where(tst_label[0] == i)[0])[0]]

        tst_sample_label = np.array([i])

        query_label = tst_sample_label
        query_binary = tst_sample_binary
        query_result = np.count_nonzero(query_binary != trn_binary[0],
                                        axis=1)  # haming distance   # don't need to divide binary length
        sort_indices = np.where(query_result==0)  # np.argsort从小到大排序返回索引
        gf1_mul_list0 = os.listdir('./gf1gf2/gf1_mul/train/0')
        gf1_mul_list0 = [ './gf1gf2/gf1_mul/train/0/'+x for x in gf1_mul_list0]
        gf1_mul_list1 = os.listdir('./gf1gf2/gf1_mul/train/1')
        gf1_mul_list1 = ['./gf1gf2/gf1_mul/train/1/' + x for x in gf1_mul_list1]
        gf1_mul_list2 = os.listdir('./gf1gf2/gf1_mul/train/2')
        gf1_mul_list2 = ['./gf1gf2/gf1_mul/train/2/' + x for x in gf1_mul_list2]
        gf1_mul_list3 = os.listdir('./gf1gf2/gf1_mul/train/3')
        gf1_mul_list3 = ['./gf1gf2/gf1_mul/train/3/' + x for x in gf1_mul_list3]
        gf1_mul_list = gf1_mul_list0+gf1_mul_list1+gf1_mul_list2+gf1_mul_list3
        print(trn_label[0][sort_indices[0][0]])
        img = cv2.imread(gf1_mul_list[sort_indices[0][0]], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    path = '/media/2T/cuican/code/Pytorch_RSIR/gf1gf2'
    if os.path.exists('./result/train_binary') and os.path.exists('./result/train_label')\
            and os.path.exists('./result/val_binary') and os.path.exists('./result/val_label')\
            and os.path.exists('./result/train_vector')and os.path.exists('./result/val_vector'):

        Tpath = np.load('./result/Tpath.npy')
        Vpath = np.load('./result/Vpath.npy')
        train_binary = torch.load('./result/train_binary')
        train_label = torch.load('./result/train_label')
        train_vector = torch.load('./result/train_vector')
        val_binary = torch.load('./result/val_binary')
        val_label = torch.load('./result/val_label')
        val_vector = torch.load('./result/val_vector')

    else:
        trainloader, valloader,Tpath,Vpath = load_data(path)
        np.save('./result/Tpath', Tpath)
        np.save('./result/Vpath', Vpath)
        t_gf1_mul_hashcode, t_gf1_mul_label, t_gf1_mul_vector, t_gf2_mul_hashcode, t_gf2_mul_label, t_gf2_mul_vector, t_gf1_pan_hashcode, t_gf1_pan_label, t_gf1_pan_vector = binary_output(trainloader, model_path=args.model)
        v_gf1_mul_hashcode, v_gf1_mul_label, v_gf1_mul_vector, v_gf2_mul_hashcode, v_gf2_mul_label, v_gf2_mul_vector, v_gf1_pan_hashcode, v_gf1_pan_label, v_gf1_pan_vector = binary_output(
            valloader, model_path=args.model)
        train_binary = torch.stack((t_gf1_mul_hashcode, t_gf2_mul_hashcode, t_gf1_pan_hashcode), 0)
        train_label = torch.stack((t_gf1_mul_label, t_gf2_mul_label, t_gf1_pan_label), 0)
        train_vector = torch.stack((t_gf1_mul_vector, t_gf2_mul_vector, t_gf1_pan_vector), 0)

        val_binary = torch.stack((v_gf1_mul_hashcode, v_gf2_mul_hashcode, v_gf1_pan_hashcode), 0)
        val_label = torch.stack((v_gf1_mul_label, v_gf2_mul_label, v_gf1_pan_label), 0)
        val_vector = torch.stack((v_gf1_mul_vector, v_gf2_mul_vector, v_gf1_pan_vector), 0)
        # train_binary, train_label, train_vector = binary_output(trainloader, model_path=args.model)
        # val_binary, val_label, val_vector = binary_output(valloader, model_path=args.model)
        if not os.path.isdir('result'):
            os.mkdir('result')
        torch.save(train_binary, './result/train_binary')
        torch.save(train_label, './result/train_label')
        torch.save(train_vector, './result/train_vector')
        torch.save(val_binary, './result/val_binary')
        torch.save(val_label, './result/val_label')
        torch.save(val_vector, './result/val_vector')


    train_binary = train_binary.cpu().numpy()
    train_binary = np.asarray(train_binary, np.int32)
    train_label = train_label.cpu().numpy()

    val_binary = val_binary.cpu().numpy()
    val_binary = np.asarray(val_binary, np.int32)
    val_label = val_label.cpu().numpy()
    Display(trn_binary=train_binary, trn_label=train_label, trn_vector=train_vector, tst_binary=val_binary,
            tst_label=val_label, tst_vector=val_vector)
    evaluate(train_binary, train_label, val_binary, val_label)
    # Display(trn_binary=train_binary,trn_label=train_label,trn_vector=train_vector,tst_binary=val_binary,tst_label=val_label,tst_vector=val_vector)


