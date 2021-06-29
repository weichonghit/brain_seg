

# IMPORTS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np

kernel1 = 1
pad1    = 0

kernel3 = 3
pad3    = 1

kernel5 = 5
pad5    = 2

kernel7 = 7
pad7    = 3

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U, x):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return x*z.expand_as(x)


# Building Blocks
class Cross_Atten_Conv(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.atten_conv1 = cSE(params['num_filters'])
        self.atten_conv2 = cSE(params['num_filters'])
        self.atten_conv3 = cSE(params['num_filters'])

    def forward(self, x1, x2, x3):

        identity1 = x1
        identity2 = x2
        identity3 = x3
        
        x1 = self.atten_conv1(identity3, x1)
        x2 = self.atten_conv2(identity1, x2)
        x3 = self.atten_conv3(identity2, x3)

        return x1,x2,x3



class Res_DenseBlock_cat_branch1(nn.Module):
    def __init__(self, params):
        super(Res_DenseBlock_cat_branch1, self).__init__()

        # Define the learnable layers
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=2*params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel1, stride=1, padding=pad1),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv_bn1(x)
        x = self.relu(x)
        identity1 = x

        x = self.conv_bn2(x)
        x += identity1
        x = self.relu(x)
        # identity2 = x

        x = self.conv_bn3(x)
        # x += identity2
        x = self.relu(x)

        return x

class Res_DenseBlock_cat_branch2(nn.Module):
    def __init__(self, params):
        super(Res_DenseBlock_cat_branch2, self).__init__()

        # Define the learnable layers
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=2*params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel5, stride=1, padding=pad5),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel1, stride=1, padding=pad1),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv_bn1(x)
        x = self.relu(x)
        identity1 = x

        x = self.conv_bn2(x)
        x += identity1
        x = self.relu(x)
        # identity2 = x

        x = self.conv_bn3(x)
        # x += identity2
        x = self.relu(x)

        return x

class Res_DenseBlock_cat_branch3(nn.Module):
    def __init__(self, params):
        super(Res_DenseBlock_cat_branch3, self).__init__()

        # Define the learnable layers
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=2*params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel7, stride=1, padding=pad7),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel1, stride=1, padding=pad1),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv_bn1(x)
        x = self.relu(x)
        identity1 = x

        x = self.conv_bn2(x)
        x += identity1
        x = self.relu(x)
        # identity2 = x

        x = self.conv_bn3(x)
        # x += identity2
        x = self.relu(x)

        return x

class Res_DenseBlock(nn.Module):
    def __init__(self, params):
        super(Res_DenseBlock, self).__init__()

        # Define the learnable layers
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel1, stride=1, padding=pad1),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        x = self.conv_bn1(x)
        x += identity
        x = self.relu(x)

        identity2 = x

        x = self.conv_bn2(x)
        x += identity2
        x = self.relu(x)

        # identity3= x

        x = self.conv_bn3(x)
        # x += identity3
        x = self.relu(x)

        return x

class Res_DenseBlockInput(nn.Module):
    def __init__(self, params):
        super(Res_DenseBlockInput, self).__init__()
        self.bn = nn.BatchNorm2d(params['num_channels'])
        # Define the learnable layers
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel3, stride=1, padding=pad3),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channels=params['num_filters'], out_channels=params['num_filters'],
                               kernel_size=kernel1, stride=1, padding=pad1),
            nn.BatchNorm2d(num_features=params['num_filters'])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.bn(x)
        x = self.conv_bn1(x)
        identity1 = x

        x = self.conv_bn2(x)
        x += identity1
        x = self.relu(x)
        # identity2 = x

        x = self.conv_bn3(x)
        # x += identity2
        x = self.relu(x)

        return x


class EncoderBlock(Res_DenseBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # For Unpooling later on with the indices

    def forward(self, x):
        out_block = super(EncoderBlock, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block


class EncoderBlockInput(Res_DenseBlockInput):
    def __init__(self, params):
        super(EncoderBlockInput, self).__init__(params)  # The init of CompetitiveDenseBlock takes in params
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # For Unpooling later on with the indices

    def forward(self, x):
        out_block = super(EncoderBlockInput, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block

class DecoderBlock_branch3(Res_DenseBlock_cat_branch3):
    def __init__(self, params):
        super(DecoderBlock_branch3, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, out_block):
        upsamp = self.upsamp(x)  # indices 返回输出最大值的序号
        concat = torch.cat((out_block, upsamp), dim=1)  # Competitive Concatenation
        out_block = super(DecoderBlock_branch3, self).forward(concat)

        return out_block

class DecoderBlock_branch2(Res_DenseBlock_cat_branch2):
    def __init__(self, params):
        super(DecoderBlock_branch2, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, out_block):
        upsamp = self.upsamp(x)  # indices 返回输出最大值的序号
        concat = torch.cat((out_block, upsamp), dim=1)  # Competitive Concatenation
        out_block = super(DecoderBlock_branch2, self).forward(concat)

        return out_block

class DecoderBlock_branch1(Res_DenseBlock_cat_branch1):
    def __init__(self, params):
        super(DecoderBlock_branch1, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, out_block):
        upsamp = self.upsamp(x)  # indices 返回输出最大值的序号
        concat = torch.cat((out_block, upsamp), dim=1)  # Competitive Concatenation
        out_block = super(DecoderBlock_branch1, self).forward(concat)

        return out_block

class ClassifierBlock(nn.Module):
    """
    Classification Block
    """
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_classes'], kernel_size=1, stride=1)  # To generate logits
    
    def forward(self, x):
        logits = self.conv(x)
        # plt.imshow(logits[1,1,:,:].detach().cpu().numpy())
        # plt.show()
        return logits  # 输出num_class类
