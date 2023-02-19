import sys
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from cifar_noisy import CIFAR10_noisy
from sklearn import manifold
from model import Model
import numpy as np
from loss import RkdDistance, RKdAngle,Info_NCE
from hoc import * 
np.random.seed(0)

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_1000_model.pth',
                    help='The pretrained model path')
parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type=int, default=150, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.6)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--reg', type = str, default='rkd_dis')
parser.add_argument('--simclr_pretrain', action='store_true')
args = parser.parse_args()

def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

args.G = 50
args.max_iter = 1500
args.device = set_device()




class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.g = Model().g
        self.fc = nn.Linear(512, num_class, bias=True)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        g_out = self.g(feature)
        out = self.fc(feature)
        return g_out, out

if args.simclr_pretrain:
    model = Net(num_class=args.num_classes, pretrained_path=args.model_path).cuda()
else:
    model = Net(num_class=args.num_classes, pretrained_path=None).cuda()


train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar10_strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



class CIFAR10_noisy_3img(CIFAR10_noisy):

    def __init__(self,root,indexes = None,
                                train=True, 
                                transform = None,
                                strong_transform = None,
                                noise_type='symmetric',noise_rate=0.6, random_state=0):
        super(CIFAR10_noisy_3img, self).__init__(root, indexes=indexes, train=train,
                 transform=transform,noise_type=noise_type,noise_rate = noise_rate,random_state=random_state)
        
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        label = self.train_noisy_labels[index]
        true_label = self.true_labels[index]
        img = self.train_data[index]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.strong_transform(img)
        img3 = self.strong_transform(img)
        return img1,img2,img3,label,true_label,index




train_dataset = CIFAR10_noisy_3img(root='./data/',indexes = None,
                                train=True, 
                                transform = train_cifar10_transform,
                                strong_transform = train_cifar10_strong_transform,
                                noise_type=args.noise_type,noise_rate=args.noise_rate, random_state=0)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,drop_last=True,
                                  shuffle=True,pin_memory=True)


test_dataset = CIFAR10(root='data', train=False, transform=test_cifar10_transform, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)


model.cuda()



reg_factory = {'rkd_dis':RkdDistance(),'rkd_angle':RKdAngle()}
base_loss = nn.CrossEntropyLoss().cuda()
self_criterion = Info_NCE() 
reg_criterion = reg_factory[args.reg]





if args.simclr_pretrain:
    alpha_plan = [0.001] * args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    alpha_plan = [0.01] * 40 + [0.001] * (args.epochs - 40)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

best_acc = [0]
def validate(val_loader, model):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            features, logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 

    return acc

def get_T_global_min(args, record, max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            cnt += 1
        lb += 1
    data_set = {'feature': origin_trans, 'noisy_label': origin_label}

    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes
    # NumTest = 50
    # all_point_cnt = 15000


    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    args.device = set_device()
    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr = lr)

    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    return E_calc, T_init


def find_trans_mat(model):
    # estimate each component of matrix T based on training with noisy labels
    print("estimating transition matrix...")
    #output_ = torch.tensor([]).float().cuda()
    record = [[] for _ in range(args.num_classes)]
    # collect all the outputs
    with torch.no_grad():
        for batch_idx, (data,data1,data2,label,true_label,idx) in enumerate(train_loader):
            data = torch.tensor(data).float().cuda()
            extracted_feature =torch.flatten(model.f(data), start_dim=1)
            for i in range(extracted_feature.shape[0]):
                record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': idx[i]})
    new_estimate_T, _ = get_T_global_min(args, record, max_step=args.max_iter, lr = 0.1, NumTest = args.G)

    return torch.tensor(new_estimate_T).float().to(args.device)

def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    loss = base_loss(outputs, target)
    return loss

model.load_state_dict(torch.load('ce_reg_'+args.noise_type+'_'+str(args.noise_rate)+'.pth')['state_dict'])
trans_mat = find_trans_mat(model)

for epoch in range(args.epochs): 
    model.train()
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    for i, (images1, images2,images3,labels,true_labels,indexes) in enumerate(train_loader):
        images1 = Variable(images1).cuda()
        images2 = Variable(images2).cuda()
        images3 = Variable(images3).cuda()
        labels = Variable(labels).cuda()
        features1, output1 = model(images1)
        features2, output2 = model(images2)
        features3, output3 = model(images3)

        loss = forward_loss(output1, labels,trans_mat) + self_criterion(features2,features3) + reg_criterion(output1,features1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model)
    if acc1 > best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)















