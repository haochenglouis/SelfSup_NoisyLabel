import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions



def loss_peer(epoch, outputs, labels):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(outputs, labels, reduce = False)
    #num_batch = len(loss_numpy)
    #loss_v = np.zeros(num_batch)
    #oss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(outputs) + 1e-8)
   
    loss =  loss - beta*torch.mean(loss_,1)
        
    return torch.mean(loss)

def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=160)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]


def DMI_loss(epoch,output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 10).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

def lq_loss(epoch,outputs, target):
    # loss = (1 - h_j(x)^q) / q
    loss = 0
    outputs = F.softmax(outputs, dim=1)
    for i in range(outputs.size(0)):
        loss += (1.0 - (outputs[i][target[i]]) ** 0.2) / 0.2

    loss = loss / outputs.size(0)
    return loss

def sce_loss(epoch, output, target):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    if epoch <=30:
        return  F.cross_entropy(output, target)
    else:
        alpha = 0.1
        beta = 1.0 
        num_classes = 10
        ce = F.cross_entropy(output, target)

        # RCE
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().to('cuda')
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = alpha * ce + beta * rce.mean()
        return loss



class RkdDistance(nn.Module):
    def forward(self, student, teacher,self_correct = True):
        student = F.softmax(student)
        t_d = pdist(teacher, squared=True)
        mean_td = t_d[t_d>0].mean()
        t_d = t_d / mean_td
       

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss



class RKdAngle(nn.Module):
    def forward(self, student, teacher,self_correct = True):
        # N x C
        # N x N x C
        
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss



class Info_NCE(nn.Module):
    def __init__(self, temperature=0.5):
        super(Info_NCE, self).__init__()
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x, y):
        batchsize = x.size(0)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        output = torch.cat((x, y),0)
        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature

        zero_matrix = torch.zeros((batchsize,batchsize),dtype=torch.bool,device=x.device)
        eye_matrix = torch.eye(batchsize, dtype=torch.bool, device=x.device)

        pos_index = torch.cat((torch.cat((zero_matrix,eye_matrix)),torch.cat((eye_matrix,zero_matrix))),1 )
        neg_index = ~torch.cat((torch.cat((eye_matrix,eye_matrix)),torch.cat((eye_matrix,eye_matrix))),1 )
        
        pos_logits = logits[pos_index].view(2*batchsize, -1)
        neg_index = logits[neg_index].view(2*batchsize, -1)

        final_logits = torch.cat((pos_logits, neg_index), dim=1)
        labels = torch.zeros(final_logits.shape[0], device=x.device, dtype=torch.long)
        loss = self.cross_entropy(final_logits, labels)

        return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res





