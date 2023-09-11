import argparse
import numpy as np
import math
import sys
import pdb
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import USPS
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets import CIFAR10, CIFAR100
from lenet import LeNet5Half
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer

torch.cuda.manual_seed_all(3047)

import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=['MNIST', 'cifar10', 'cifar100', 'AR', 'FashionMNIST', 'usps'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_Gt', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_Gs', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_Ds', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--oh', type=float, default=0.05, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.01, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

accr = 0
accr_best = 0
accr_mean = 0
accr_list = []


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        g1 = self.conv_blocks1(img)
        img = nn.functional.interpolate(g1, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


generator_t = Generator().cuda()
generator_s = Generator().cuda()
teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
teacher = nn.DataParallel(teacher)
generator_t = nn.DataParallel(generator_t)
generator_s = nn.DataParallel(generator_s)


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


#MNIST泛化
# if opt.dataset == 'MNIST':
#     net = LeNet5Half().cuda()
#     net = nn.DataParallel(net)
#     test_transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#                  ])
#     data_test = torchvision.datasets.ImageFolder(
#         root='C://Users//research//Desktop//mnist//test', transform=test_transform)
#     data_test_loader = DataLoader(data_test, batch_size=64, num_workers=0, shuffle=False)
#     optimizer_Gt1 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
#     optimizer_Gt2 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
#     optimizer_Gs = torch.optim.Adam(generator_s.parameters(), lr=opt.lr_Gs)
#     optimizer_Ds = torch.optim.Adam(net.parameters(), lr=opt.lr_Ds)

if opt.dataset == 'MNIST':
    net = LeNet5Half().cuda()#学生判别器
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=0, shuffle=False)
    # Optimizers
    optimizer_Gt1 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
    optimizer_Gt2 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
    optimizer_Gs = torch.optim.Adam(generator_s.parameters(), lr=opt.lr_Gs)
    optimizer_Ds = torch.optim.Adam(net.parameters(), lr=opt.lr_Ds)

if opt.dataset != 'MNIST':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10':
        net = resnet.ResNet18().cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR10(opt.data,
                            train=False,
                            transform=transform_test)
    if opt.dataset == 'cifar100':
        net = resnet.ResNet18(num_classes=100).cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR100(opt.data,
                             train=False,
                             transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)
    optimizer_Gt1 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
    optimizer_Gt2 = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
    optimizer_Gs = torch.optim.Adam(generator_s.parameters(), lr=opt.lr_Gs)
    optimizer_Ds = torch.optim.SGD(net.parameters(), lr=opt.lr_Ds, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch < 800:
        lr = learning_rate
    elif epoch < 1600:
        lr = 0.1 * learning_rate
    else:
        lr = 0.01 * learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
at_criterion = AttentionTransfer()
klloss = nn.MSELoss()

# ----------
#  Trainin
# ----------

batches_done = 0

for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_Ds, epoch, opt.lr_Gs)

    for i in range(120):

        # 优化 Gt
        z1 = Variable(torch.randn(opt.batch_size, opt.latent_dim), requires_grad=True).cuda()#生成一批与真实数据集batch_size一样128，潜在空间维度一样的随机样本1000
        #print(len(z1))
        optimizer_Gt1.zero_grad()
        #print(z1)
        img_t1 = generator_t(z1)#Xt
        outputs_T, features_T, _, _, _ = teacher(img_t1, out_feature=True)
        pred_T = outputs_T.data.max(1)[1]

        # features_T1 = features_T * scaled_x
        loss_activation_T1 = -features_T.abs().mean()#特征选择 在此乘优化Gt后，Z的梯度的归一化矩阵

        loss_one_hot_T = criterion(outputs_T, pred_T)

        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
        loss_information_entropy_T = (softmax_o_T * torch.log10(softmax_o_T)).sum()

        loss_T1 = loss_one_hot_T * opt.oh + loss_information_entropy_T * opt.ie + loss_activation_T1 * opt.a
        z1.retain_grad()
        #with torch.autograd.detect_anomaly():
        loss_T1.backward(retain_graph=True)


        #print(z1.grad)
        x = z1.grad

        # 归一化[0,1]沿着行的方向计算最小值和最大值
        min_vals, _ = torch.min(x, dim=1, keepdim=True)
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        # 最小-最大缩放，将x的范围缩放到[0, 1]
        scaled_x = (x - min_vals) / (max_vals - min_vals)
        #print(scaled_x)

        # print(scaled_x.shape)#torch.Size([128, 1000]) 改完'--latent_dim'default变成torch.Size([128, 120])
        # print(features_T.shape)#torch.Size([128, 120])
        features_T2=features_T*scaled_x
        loss_activation_T2 = -features_T2.abs().mean()#特征选择 在此乘优化Gt后，Z的梯度的归一化矩阵
        # if loss_activation_T2==None:
        #     loss_activation_T2=loss_activation_T1
        loss_T2 = loss_one_hot_T * opt.oh + loss_information_entropy_T * opt.ie + loss_activation_T2 * opt.a
        loss_T2.backward()

        optimizer_Gt1.step()  # 利用Loss进行backward反向传播计算各个参数的梯度之后,采用step()进行一步更新,更新权值参数
        optimizer_Gt2.step()




        #print(z1.is_leaf) #z1不是叶子节点，它(非叶节点)的梯度值在反向传播过程中使用完后就会被清除,不会被保留


        # 优化 Ns
        net.train()
        z2 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_Ds.zero_grad()
        img_s1 = generator_s(z2)
        loss_kd = inter_class_relation(net(img_s1.detach()), teacher(img_s1.detach())) + intra_class_relation(
            net(img_s1.detach()), teacher(img_s1.detach()))
        loss_kd.backward()
        optimizer_Ds.step()

        # 优化 Gs
        z3 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_Gs.zero_grad()
        img_t2 = generator_t(z3)
        img_s2 = generator_s(z3)
        loss_kl = klloss(img_s2, img_t2.detach())
        loss_relation = dist_criterion(net(img_s2), teacher(img_s2)) + angle_criterion(net(img_s2), teacher(img_s2))
        loss_zhiliang = - torch.log(loss_relation + 1)
        loss_S = loss_zhiliang + loss_kl
        loss_S.backward()
        optimizer_Gs.step()

        if i == 1:
            print("[Epoch %d/%d]  [loss_T: %f] [loss_S: %f] [loss_KD: %f]  " % (
            epoch, opt.n_epochs, loss_T2.item(), loss_S.item(), loss_kd.item()))

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    accr_list.append(accr)
    if accr > accr_best:
        torch.save(net, 'C://Users//research//Desktop//our.pth')
        # torch.save(net, opt.output_dir + 'student')
        # torch.save(generator_t, opt.output_dir + 'generator_t')
        # torch.save(generator_s, opt.output_dir + 'generator_s')
        accr_best = accr

print("Best Acc=%.6f" % accr_best)
print(accr_list)

