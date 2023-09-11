# Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch.nn as nn



class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feature=False):
        b1 = self.conv1(img)
        output = self.relu1(b1)
        output = self.maxpool1(output)
        b2 = self.conv2(output)
        output = self.relu2(b2)
        output = self.maxpool2(output)
        b3 = self.conv3(output)
        output = self.relu3(b3)
        feature = output.view(-1, 120)#重新塑造为一个列数为 120，行数自动推断的二维张量
        output = self.fc1(feature)
        b4 = self.relu4(output)
        output = self.fc2(b4)
        if out_feature == False:
            return output
        else:
            return output, feature,b1,b2,b3


class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feature=False):
        b1 = self.conv1(img)
        output = self.relu1(b1)
        output = self.maxpool1(output)
        b2 = self.conv2(output)
        output = self.relu2(b2)
        output = self.maxpool2(output)
        b3 = self.conv3(output)
        output = self.relu3(b3)
        feature = output.view(-1, 60)
        output = self.fc1(feature)
        b4 = self.relu4(output)
        output = self.fc2(b4)
        if out_feature == False:
            return output
        else:
            return output, feature,b1,b2,b3

