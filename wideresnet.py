# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import sys
# import numpy as np
#
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
#
# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#         init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)
#
# class wide_basic(nn.Module):
#     def __init__(self, in_planes, planes, dropout_rate, stride=1):
#         super(wide_basic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
#             )
#
#     def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(self.bn1(x))))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += self.shortcut(x)
#
#         return out
#
# class Wide_ResNet40_2(nn.Module):
#     def __init__(self, depth=40, widen_factor=2, dropout_rate=0.0, num_classes=10):
#         super(Wide_ResNet40_2, self).__init__()
#         self.in_planes = 16
#
#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = (depth-4)/6
#         k = widen_factor
#
#         print('| Wide-Resnet %dx%d' %(depth, k))
#         nStages = [16, 16*k, 32*k, 64*k]
#
#         self.conv1 = conv3x3(3,nStages[0])
#         self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
#         self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
#         self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
#         self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
#         self.linear = nn.Linear(nStages[3], num_classes)
#
#     def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
#         strides = [stride] + [1]*(int(num_blocks)-1)
#         layers = []
#
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, dropout_rate, stride))
#             self.in_planes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         feature = out.view(out.size(0), -1)
#         out = self.linear(out)
#
#         return out, feature
#
#
# class Wide_ResNet40_1(nn.Module):
#     def __init__(self, depth=40, widen_factor=1, dropout_rate=0.0, num_classes=10):
#         super(Wide_ResNet40_1, self).__init__()
#         self.in_planes = 16
#
#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = (depth-4)/6
#         k = widen_factor
#
#         print('| Wide-Resnet %dx%d' %(depth, k))
#         nStages = [16, 16*k, 32*k, 64*k]
#
#         self.conv1 = conv3x3(3,nStages[0])
#         self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
#         self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
#         self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
#         self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
#         self.linear = nn.Linear(nStages[3], num_classes)
#
#     def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
#         strides = [stride] + [1]*(int(num_blocks)-1)
#         layers = []
#
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, dropout_rate, stride))
#             self.in_planes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self,x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         feature = out.view(out.size(0), -1)
#         out = self.linear(out)
#
#         return out,feature


# import math #1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                padding=0, bias=False) or None
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)
#
# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)
#
# class Wide_ResNet40_2(nn.Module):
#     def __init__(self, depth=40, num_classes=10, widen_factor=2, dropRate=0.0):
#         super(Wide_ResNet40_2, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x,out_feature=False):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         feature = out.view(-1, self.nChannels)
#         out = self.fc(feature)
#         if out_feature == False:
#             return out
#         else:
#             return out, feature
# class Wide_ResNet16_1(nn.Module):
#     def __init__(self, depth=16, num_classes=1, widen_factor=2, dropRate=0.0):
#         super(Wide_ResNet16_1, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x,out_feature=False):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         feature = out.view(-1, self.nChannels)
#         out = self.fc(feature)
#         if out_feature == False:
#             return out
#         else:
#             return out, feature

import torch
import torch.nn as nn


class WideBasic(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):

        residual = self.residual(x)
        shortcut = self.shortcut(x)

        return residual + shortcut

class WideResNet(nn.Module):
    def __init__(self, num_classes, block, depth=40, widen_factor=2):
        super().__init__()

        self.depth = depth
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        self.init_conv = nn.Conv2d(3, self.in_channels, 3, 1, padding=1)
        self.conv2 = self._make_layer(block, 16 * k, l, 1)
        self.conv3 = self._make_layer(block, 32 * k, l, 2)
        self.conv4 = self._make_layer(block, 64 * k, l, 2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x1 = x.view(x.size(0), -1)
        x = self.linear(x1)

        return x,x1



    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)


# Table 9: Best WRN performance over various datasets, single run results.
def Wide_ResNet40_1(depth=40, widen_factor=1):
    net = WideResNet(10, WideBasic, depth=depth, widen_factor=widen_factor)
    return net
def Wide_ResNet40_2(depth=40, widen_factor=2):
    net = WideResNet(10, WideBasic, depth=depth, widen_factor=widen_factor)
    return net

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, activation, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.activation = activation
#         self.conv1 = nn.Conv2d(
#             in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(out_planes)
#
#         self.conv2 = nn.Conv2d(
#             out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.droprate = dropRate
#         self.equalInOut = in_planes == out_planes
#         self.convShortcut = (
#                 (not self.equalInOut)
#                 and nn.Conv2d(
#             in_planes,
#             out_planes,
#             kernel_size=1,
#             stride=stride,
#             padding=0,
#             bias=False,
#         )
#                 or None
#         )
#
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.activation(self.bn1(x))
#         else:
#             out = self.activation(self.bn1(x))
#         out = self.activation(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)
#
#
# class NetworkBlock(nn.Module):
#     def __init__(self, activation, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(
#             activation, block, in_planes, out_planes, nb_layers, stride, dropRate
#         )
#
#     def _make_layer(self, activation, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(
#                 block(activation,
#                       i == 0 and in_planes or out_planes,
#                       out_planes,
#                       i == 0 and stride or 1,
#                       dropRate
#                       )
#             )
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class WideResNet(nn.Module):
#     def __init__(self,depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
#         super(WideResNet, self).__init__()
#         nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
#         assert (depth - 4) % 6 == 0
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(
#             3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
#         )
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock( n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         # self.activation = activation
#         self.classifier = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2.0 / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def forward(self, x, out_feature=False):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.activation(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         feature = out.view(-1, self.nChannels)
#         out = self.classifier(out)
#         if out_feature == False:
#             return out
#         else:
#             return out, feature
#
#
# # def wrn_28_10(activation, **kwargs):
# #     return WideResNet(activation, depth=28, widen_factor=10, **kwargs)
# #
# #
# # def wrn_28_4(activation, **kwargs):
# #     return WideResNet(activation, depth=28, widen_factor=4, **kwargs)
# #
# #
# # def wrn_28_5(activation, **kwargs):
# #     return WideResNet(activation, depth=28, widen_factor=5, **kwargs)
# #
# #
# # def wrn_28_1(activation, **kwargs):
# #     return WideResNet(activation, depth=28, widen_factor=1, **kwargs)
# #
# #
# # def wrn_34_10(activation, **kwargs):
# #     return WideResNet(activation, depth=34, widen_factor=10, **kwargs)
#
#
# def wrn_40_2():
#     return WideResNet(depth=40, num_classes=10,widen_factor=2,dropRate=0.0)
#
# def wrn_40_1():
#     return WideResNet( depth=40, widen_factor=1)
#
# def wrn_16_2():
#     return WideResNet(depth=16, widen_factor=1)
#
# # def wrn_34_10(activation, **kwargs):
# #     return WideResNet(activation, depth=34, widen_factor=10, **kwargs)
# #
# #
# # # ~4x slower than wrn-28-10
# # def wrn_34_20(activation, **kwargs):
# #     return WideResNet(activation, depth=34, widen_factor=20, **kwargs)
# #
# #
# # # ~6x slower than wrn-28-10
# # def wrn_70_16(activation, **kwargs):
# #     return WideResNet(activation, depth=70, widen_factor=16, **kwargs)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Block(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, stride, droprate=0.0):
#         super(Block, self).__init__()
#         self.stride = stride
#         self.droprate = droprate
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                  stride=stride, padding=0, bias=False)
#
#     def forward(self, x):
#         shortcut = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.droprate > 0:
#             x = F.dropout(x, p=self.droprate)
#         shortcut = self.conv1x1(shortcut)
#
#         x += shortcut
#         x = self.relu(x)
#
#         return x
#
#
# class WideResNet(torch.nn.Module):
#     def __init__(self, num_layers, widen_factor, block, num_classes=10, droprate=0.0):
#         super(WideResNet, self).__init__()
#         self.num_layers = num_layers
#         self.droprate = droprate
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.layer1 = self.get_layers(block, 16, 16 * widen_factor, 1)
#         self.layer2 = self.get_layers(block, 16 * widen_factor, 32 * widen_factor, 2)
#         self.layer3 = self.get_layers(block, 32 * widen_factor, 64 * widen_factor, 2)
#
#         self.avg_pool = nn.AvgPool2d(8, stride=1)
#         self.fc = nn.Linear(64 * widen_factor, num_classes)
#
#     def get_layers(self, block, in_channels, out_channels, stride):
#         layers = []
#
#         for i in range(self.num_layers):
#             if i == 0:
#                 layers.append(block(in_channels, out_channels, stride, self.droprate))
#                 continue
#             layers.append(block(out_channels, out_channels, 1, self.droprate))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
# def wrn_40_2(num_layers=40, widen_factor=2, num_classes=10, droprate=0.0):
#     return WideResNet(num_layers, widen_factor, Block, num_classes, droprate)