# import torch
# import torch.nn as nn
#
# class VGG11(nn.Module):
#     def __init__(self):
#         super(VGG11, self).__init__()
#
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.conv_block3 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.conv_block4 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.conv_block5 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.linear1 = nn.Sequential(
#             nn.Linear(in_features=3136, out_features=512),
#             nn.ReLU(),
#             nn.Dropout(p=0.5)
#         )
#
#         self.linear2 = nn.Sequential(
#             nn.Linear(in_features=512, out_features=512),
#             nn.ReLU(),
#             nn.Dropout(p=0.5)
#         )
#
#         self.linear3 = nn.Linear(in_features=512, out_features=120)
#
#
#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.conv_block4(x)
#         x = self.conv_block5(x)
#         x = x.view(x.shape[0], -1)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         return x


# import torch
# import torch.nn as nn
#
#
# class vgg(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64,  # [B,64,224,224]
#                       kernel_size=(3, 3),stride=1, padding=1),
#             nn.ReLU(),
#             #nn.BatchNorm1d(),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,64,112,112]
#
#             nn.Conv2d(in_channels=64, out_channels=128,  # [B,128,112,112]
#                       kernel_size=(3, 3), stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,128,56,56]
#
#             # [B,256,56,56]
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1,padding=1),
#             # [B,256,56,56]
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,256,28,28]
#
#             # [B,512,56,56]
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),stride=1, padding=1),
#             # [B,512,56,56]
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [B,512,14,14]
#
#             # [B,512,28,28]
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),stride=1, padding=1),
#             # [B,512,56,56]
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # [B,512,7,7]
#         )
#
#         self.fc_layers=nn.Sequential(
#             nn.Linear(512,4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,1000),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x):
#         conv_out = self.conv_layers(x)
#         feature=conv_out.view(conv_out.size(0), -1)
#         fc_out=self.fc_layers(conv_out.view(conv_out.size(0),-1))
#         return fc_out,feature

# import torch.nn as nn
# import torch.nn.functional as F
#
# # 定义VGG11模型
# class VGG11(nn.Module):
#     def __init__(self):
#         super(VGG11, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(512 * 2 * 2, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.conv7(x))
#         x = F.relu(self.conv8(x))
#         x = F.max_pool2d(x, 2)

#         x = x.view(-1, 512 * 2 * 2)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return x

import torch.nn as nn

vgg11_arch = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def make_conv_layer(arch):
    in_channels = 3
    module_list = []

    for layer_info in arch:
        if layer_info != 'M':
            out_channels = layer_info
            module_list += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()]
            in_channels = out_channels
        else:
            module_list += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]

    return nn.Sequential(*module_list)

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = make_conv_layer(vgg11_arch)

        self.fc_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x,out_feature=False):
        conv_out = self.conv_layer(x)
        feature = conv_out.view(conv_out.size(0), -1)
        conv_out = conv_out.flatten(1)
        fc_out = self.fc_layer(conv_out)
        if out_feature == False:
            return fc_out
        else:
            return fc_out,feature
