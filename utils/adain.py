"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Some modifications made
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):

        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):

    def __init__(self, latent_size, channels):

        '''
        latent_size : style(embedding) vector length
        channels : #(channels) of x
        '''

        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2).to(torch.device(0)) # argument로 toss받고싶다

    def forward(self, x, latent): # latent for style

        '''
        x : content vector
        latent : style vector
        '''

        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1 # Denorm
        return x

class AdaIN_layer(nn.Module):
    def __init__(self, latent_size, channel):
        super(AdaIN_layer, self).__init__()

        self.instance_norm = InstanceNorm()
        self.styler = ApplyStyle(latent_size, channel) # latent_size : length of style(=id) vector / channels for feature maps

    def forward(self, latent, id): # for id injection

        latent_norm = self.instance_norm(latent) # latent for content
        latent_styled = self.styler(latent_norm, id) # styled by id

        return latent_styled


        
class AdaIN_ResBlock(nn.Module):

    ## content를 AdaIN-3x3Conv 조합을 두 번 돌린 후(size는 보존되도록 하고, 채널만 절반으로 줄도록), content를 1x1Conv로 채널 절반으로 만든 것과 skip connection해서 output
    ## 이렇게 하면 block을 통과해도 channel, size 전부 보존됨 -> 단일 AdaIN layer보단 spatial-information-reach할 것으로 예상

    def __init__(self, style_size, content_size):
        super(AdaIN_ResBlock, self).__init__()

        self.width = -1
        self.height = -1

        self.adain1 = AdaIN_layer(style_size, content_size)
        self.conv1 = nn.Conv2d(content_size, content_size//2, 3).to(torch.device(0)) # 3x3 convolution preserving #(channels)
        self.adain2 = AdaIN_layer(style_size, content_size//2)
        self.conv2 = nn.Conv2d(content_size//2, content_size//2, 3).to(torch.device(0)) # 3x3 convolution preserving #(channels)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(content_size, content_size//2, 1).to(torch.device(0)) # 1x1 convolution preserving #(channels)

    def forward(self, content, style): # 
        self.width = content.size()[2]
        self.height = content.size()[3]

  
        adain1 = self.relu(self.adain1(content, style))
        output1 = self.conv1(adain1)
        output1 = F.interpolate(output1, size=(self.width, self.height), mode='bilinear', align_corners=False) # 사이즈가 작아서 padding보단 그래도 interpolation이 나을듯?


        adain2 = self.relu(self.adain2(output1, style))
        output2 = self.conv2(adain2)
        output2 = F.interpolate(output2, size=(self.width, self.height), mode='bilinear', align_corners=False)



        skip = self.conv3(content)
        res = torch.cat([output2, skip], dim = 1)

        return res

        

        
    