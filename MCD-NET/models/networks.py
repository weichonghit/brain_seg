

# IMPORTS
import torch
import torch.nn as nn
import models.sub_module as sm
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np


class MCD_Net_c(nn.Module):
    def __init__(self, params):
        super(MCD_Net_c, self).__init__()

        
        # Parameters for the Descending Arm
        self.encode1 = sm.EncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.EncoderBlock(params)
        self.encode3 = sm.EncoderBlock(params)
        self.encode4 = sm.EncoderBlock(params)
        

        # 第一组解码网络
        self.bottleneck = sm.Res_DenseBlock(params)
        self.decode4 = sm.DecoderBlock_branch1(params)
        self.decode3 = sm.DecoderBlock_branch1(params)
        self.decode2 = sm.DecoderBlock_branch1(params)
        self.decode1 = sm.DecoderBlock_branch1(params)
        # 第二组解码网络
        self.bottleneck_part2  = sm.Res_DenseBlock(params)
        self.decode4_part2   = sm.DecoderBlock_branch2(params)
        self.decode3_part2   = sm.DecoderBlock_branch2(params)
        self.decode2_part2   = sm.DecoderBlock_branch2(params)
        self.decode1_part2   = sm.DecoderBlock_branch2(params)
        # 第三组解码网络
        self.bottleneck_part3  = sm.Res_DenseBlock(params)
        self.decode4_part3   = sm.DecoderBlock_branch3(params)
        self.decode3_part3   = sm.DecoderBlock_branch3(params)
        self.decode2_part3   = sm.DecoderBlock_branch3(params)
        self.decode1_part3   = sm.DecoderBlock_branch3(params)
        # 分类器
        self.classifier        = sm.ClassifierBlock(params)
        self.classifier_part2  = sm.ClassifierBlock(params)
        self.classifier_part3  = sm.ClassifierBlock(params)

        self.cross_conv1    = sm.Cross_Atten_Conv(params)
        self.cross_conv2    = sm.Cross_Atten_Conv(params)
        self.cross_conv3    = sm.Cross_Atten_Conv(params)
        self.cross_conv4    = sm.Cross_Atten_Conv(params)
        self.cross_conv5    = sm.Cross_Atten_Conv(params)
        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        
        encoder_output1, skip_encoder_1 = self.encode1.forward(input)
        encoder_output2, skip_encoder_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4 = self.encode4.forward(encoder_output3)

        # 解码模块
        encoder_output4_part2, encoder_output4_part3 = encoder_output4, encoder_output4 # seperate to 3 branches
        bottleneck                             = self.bottleneck(encoder_output4)
        bottleneck_part2                       = self.bottleneck_part2(encoder_output4_part2)
        bottleneck_part3                       = self.bottleneck_part3(encoder_output4_part3)
        bottleneck, bottleneck_part2, bottleneck_part3 = self.cross_conv5(bottleneck, bottleneck_part2, bottleneck_part3)

        decoder_output4                        = self.decode4.forward(bottleneck, skip_encoder_4)
        decoder_output4_part2                  = self.decode4_part2.forward(bottleneck_part2, skip_encoder_4)
        decoder_output4_part3                  = self.decode4_part3.forward(bottleneck_part3, skip_encoder_4)
        decoder_output4, decoder_output4_part2, decoder_output4_part3 = self.cross_conv4(decoder_output4, decoder_output4_part2, decoder_output4_part3)

        decoder_output3                        = self.decode3.forward(decoder_output4, skip_encoder_3)
        decoder_output3_part2                  = self.decode3_part2.forward(decoder_output4_part2, skip_encoder_3)
        decoder_output3_part3                  = self.decode3_part3.forward(decoder_output4_part3, skip_encoder_3)
        decoder_output3, decoder_output3_part2, decoder_output3_part3 = self.cross_conv3(decoder_output3, decoder_output3_part2, decoder_output3_part3)

        decoder_output2                        = self.decode2.forward(decoder_output3, skip_encoder_2)
        decoder_output2_part2                  = self.decode2_part2.forward(decoder_output3_part2, skip_encoder_2)
        decoder_output2_part3                  = self.decode2_part3.forward(decoder_output3_part3, skip_encoder_2)
        decoder_output2, decoder_output2_part2, decoder_output2_part3 = self.cross_conv2(decoder_output2, decoder_output2_part2, decoder_output2_part3)

        decoder_output1                        = self.decode1.forward(decoder_output2, skip_encoder_1)
        decoder_output1_part2                  = self.decode1_part2.forward(decoder_output2_part2, skip_encoder_1)
        decoder_output1_part3                  = self.decode1_part3.forward(decoder_output2_part3, skip_encoder_1)
        decoder_output1, decoder_output1_part2, decoder_output1_part3 = self.cross_conv1(decoder_output1, decoder_output1_part2, decoder_output1_part3)


        # 分类器
        logits_part1 = self.classifier.forward(decoder_output1)
        logits_part2 = self.classifier_part2.forward(decoder_output1_part2)
        logits_part3 = self.classifier_part3.forward(decoder_output1_part3)
    
        return logits_part1, logits_part2, logits_part3
