"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Some modifications made
"""



import torch
import torch.nn as nn


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
        print("reach here?")
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        print("reach here??")
        shape = [-1, 2, x.size(1), 1, 1]
        print("reach here???")
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        print("reach here????")
        
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1 # Denorm
        return x

class AdaIN_block(nn.Module):
    def __init__(self, latent_size, channel):
        super(AdaIN_block, self).__init__()

        self.instance_norm = InstanceNorm()
        self.styler = ApplyStyle(latent_size, channel) # latent_size : length of style(=id) vector / channels for feature maps

    def forward(self, latent, id): # for id injection

        latent_norm = self.instance_norm(latent) # latent for content
        print("latnet_norm size", latent_norm.size())
        print("latnet_norm type", type(latent_norm))
        print("id size", id.size())
        print("id type", type(id))
        latent_styled = self.styler(latent_norm, id) # styled by id

        return latent_styled

        
        

        

        
    