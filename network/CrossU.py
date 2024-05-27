
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import FlowFaceCrossAttentionModel
from network.FFGenerator import UNet, deconv4x4, noskip_deconv4x4, outconv, CAdeconv4x4
from utils.adain import *

class CrossUnetAttentionGenerator(nn.Module):
    # def __init__(self, seq_len, n_head, q_dim, k_dim, kv_dim, backbone='unet'):
    def __init__(self, seq_len:int=16, n_head:int=2, q_dim:int=1024, k_dim:int=1024, kv_dim:int=1024, backbone='unet', num_adain=1):
        super(CrossUnetAttentionGenerator, self).__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.n_head = n_head
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.kv_dim = kv_dim
        self.num_adain = num_adain
        self.CUMAE_src = UNet(backbone='unet')
        self.CUMAE_tgt = UNet(backbone='unet')
        # CUMAE_cross = CrossUMLAttrEncoder(backbone='unet')
        
        # (self, seq_len: int, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        self.FFCA0 = FlowFaceCrossAttentionModel(seq_len=self.seq_len, n_head=self.n_head, q_dim=self.q_dim, k_dim=self.k_dim, kv_dim=self.kv_dim)  ##FFCA0 = bottleneck
        self.FFCA1 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*4, n_head=self.n_head, q_dim=self.q_dim, k_dim=self.k_dim, kv_dim=self.kv_dim)
        self.FFCA2 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*16, n_head=self.n_head, q_dim=self.q_dim//2, k_dim=self.k_dim//2, kv_dim=self.kv_dim//2)
        self.FFCA3 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*64, n_head=self.n_head, q_dim=self.q_dim//4, k_dim=self.k_dim//4, kv_dim=self.kv_dim//4)
        self.FFCA4 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*256, n_head=self.n_head, q_dim=self.q_dim//8, k_dim=self.k_dim//8, kv_dim=self.kv_dim//8)
        self.FFCA5 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*1024, n_head=self.n_head, q_dim=self.q_dim//16, k_dim=self.k_dim//16, kv_dim=self.kv_dim//16)
        self.FFCA6 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*4096, n_head=self.n_head, q_dim=self.q_dim//32, k_dim=self.k_dim//32, kv_dim=self.kv_dim//32)
        self.FFCA7 = FlowFaceCrossAttentionModel(seq_len=self.seq_len*4096, n_head=self.n_head, q_dim=3, k_dim=3, kv_dim=3)  ##computationally expensive
        
        self.AdaIN_layer1 = AdaIN_ResBlock(769, 1024)
        self.AdaIN_layer2 = AdaIN_ResBlock(769, 512)
        self.AdaIN_layer3 = AdaIN_ResBlock(769, 256)
        self.AdaIN_layer4 = AdaIN_ResBlock(769, 128)
        self.AdaIN_layer5 = AdaIN_ResBlock(769, 64)
        self.AdaIN_layer6 = AdaIN_ResBlock(769, 32)
        
        
        # ##noskip_deconv4x4을 하면 crossattention을 안해도 되니 컴퓨팅을 아낄수있다
        # self.deconv1 = noskip_deconv4x4(1024, 1024)  ## 4 ->  8 (w,h)
        # self.deconv2 = deconv4x4(1024, 512)  ## 8 > 16
        # self.deconv3 = deconv4x4(512, 256) ## 16 > 32
        # self.deconv4 = deconv4x4(256, 128) ## 32 > 64
        # self.deconv5 = noskip_deconv4x4(128, 64) ## 64 > 128
        # self.deconv6 = noskip_deconv4x4(64, 32) ## 128 > 256
        # self.deconv7 = outconv(32, 3) ## 128 > 256
        
        self.deconv1 = noskip_deconv4x4(self.q_dim, self.q_dim)  ## 1024x4x4 ->  1024x8x8 (w,h)
        self.deconv2 = CAdeconv4x4(self.q_dim, self.q_dim//2)  ## 1024x8x8 > 512x16x16
        self.deconv3 = CAdeconv4x4(self.q_dim//2, self.q_dim//4) ## 512x16x16 > 256x32x32
        self.deconv4 = CAdeconv4x4(self.q_dim//4, self.q_dim//8) ## 256x32x32 > 128x64x64
        self.deconv5 = noskip_deconv4x4(self.q_dim//8, self.q_dim//16) ## 128x64x64 > 64x128x128
        self.deconv6 = noskip_deconv4x4(self.q_dim//16, self.q_dim//32) ## 64x128x128 > 32x256x256
        self.deconv7 = outconv(self.q_dim//32, 3) ## 32x256x256 > 3x256x256
        


    def forward(self, target, source, id_emb):
        '''
        src나 tgt나 다 똑같다.
        src_bottlneck_attr = 1024x4x4
        src_z_attr1 = 1024x8x8
        src_z_attr2 = 512x16x16
        src_z_attr3 = 256x32x32
        src_z_attr4 = 128x64x64
        src_z_attr5 = 64x128x128
        src_z_attr6 = 32x256x256
        src_z_attr7 = 3x256x256
        
        output:
        'output7, src_z_attr7, tgt_z_attr7'
        
        output7: The final output of CrossAttention Generator
        src_z_attr7: The final output of source face Unet
        tgt_z_attr7: The final output of target face Unet
        '''
        src_bottlneck_attr, src_z_attr1, src_z_attr2, src_z_attr3, src_z_attr4, src_z_attr5, src_z_attr6, src_z_attr7 = self.CUMAE_src(source)
        # print('src_unet passed')
        # print('src_bottlneck_attr size: ', src_bottlneck_attr.shape)
        tgt_bottlneck_attr, tgt_z_attr1, tgt_z_attr2, tgt_z_attr3, tgt_z_attr4, tgt_z_attr5, tgt_z_attr6, tgt_z_attr7 = self.CUMAE_tgt(target) ##z_tgt_attr1 img_size 8  z_tgt_attr6 img_size 256
        # print('tgt_unet passed')
        
        batch_size= tgt_bottlneck_attr.shape[0]
        width0= tgt_bottlneck_attr.shape[2]
        width1= tgt_z_attr1.shape[2]
        width2= tgt_z_attr2.shape[2]
        width3= tgt_z_attr3.shape[2]
        width4= tgt_z_attr4.shape[2]
        width5= tgt_z_attr5.shape[2]
        width6= tgt_z_attr6.shape[2]
        width7= tgt_z_attr7.shape[2]
        
        
        
        ##z_cross_attr0 = 1024x4x4  this is same as bottleneck block
        z_cross_attr0 = self.FFCA0(tgt_bottlneck_attr, src_bottlneck_attr) # [B, 16, 1024]
        # print("z_cross_attr0 shape", z_cross_attr0.size())
        # print('cross_att passed')
        # return z_cross_attr0

        #z_cross_attr1 = [B, 64, 1024]
        z_cross_attr1 = self.FFCA1(tgt_z_attr1, src_z_attr1)  ##tgt_z_attr1 = [B, 1024, 8, 8]
        ##512x16x16 
        z_cross_attr2 = self.FFCA2(tgt_z_attr2, src_z_attr2)
        ##256x32x32
        z_cross_attr3 = self.FFCA3(tgt_z_attr3, src_z_attr3)
        ##128x64x64
        z_cross_attr4 = self.FFCA4(tgt_z_attr4, src_z_attr4)
        
        # print('z_cross_attr4 passed')
        # return z_cross_attr4
        #64x128x128
        # z_cross_attr5 = self.FFCA5(tgt_z_attr5, src_z_attr5)
        #32x256x256
        # z_cross_attr6 = self.FFCA6(tgt_z_attr6, src_z_attr6)
        #3x256x256
        # z_cross_attr7 = self.FFCA7(tgt_z_attr7, src_z_attr7)

        
        ##1024x4x4 -> 1024x8x8 (output1)
        z_cross_attr0 = z_cross_attr0.reshape(batch_size, -1, width0, width0)  ## z_cross_attr0 becomes [B, 1024, 8, 8]
        


        output1 = self.deconv1(z_cross_attr0) ## 4 > 8  ##스킵커넥션없이 conv만 해서 키운것. output1 = [B, 1024, 8, 8]
        # print('output1', output1.shape)
        # print('id_emb', id_emb.shape)
        # print('output1.size()[1]', output1.size()[1])
        # print('id_emb.size()[1]', id_emb.size()[1])
        # print("output1 shape before transferring", output1.size())
        # # Style transfer against output1
        # print("id_emb.size()[1]", id_emb.size()[1])
        # print("output1.size()[1]", output1.size()[1])
        
        # adain_1 = AdaIN_layer(id_emb.size()[1], output1.size()[1])
        # output1 = adain_1(output1, id_emb) if 1<= self.num_adain else output1 # conditional injection
        output1 = self.AdaIN_layer1(output1, id_emb) if 1<= self.num_adain else output1 # conditional injection




        ##1024x8x8 -> 512x16x16 (output2)
        # print('output1:', output1.shape)
        # return output1
        
        ## z_cross_attr1 = [B, seq_len, dim] = (batch_size, 64, 1024)
        ## output1 =  [B, dim, height, width] = (baych, 1024, 8, 8)
        ## here, we should make z_cross_attr1 shape same as output 1 
        z_cross_attr1 = z_cross_attr1.reshape(batch_size, -1, width1, width1)
        # print('reshaped z_cross_attr1:', z_cross_attr1.shape)
        output2 = self.deconv2(output1, z_cross_attr1) ## 8 > 16
        # adain_2 = AdaIN_layer(id_emb.size()[1], output2.size()[1])
        # output2 = adain_2(output2, id_emb) if 2<= self.num_adain else output2 # conditional injection
        output2 = self.AdaIN_layer2(output2, id_emb) if 1<= self.num_adain else output2 # conditional injection
        # print('output2', output1.shape)
        # print('id_emb', id_emb.shape)
        # print('output2.size()[1]', output2.size()[1])
        # print('id_emb.size()[1]', id_emb.size()[1])
        
        
        # output2 = self.AdaIN_layer2(output2, id_emb) if 1<= self.num_adain else output2 # conditional injection
        # print('output2:', output2.shape)
        
        ##512x16x16 -> 256x32x32 (output3)
        z_cross_attr2 = z_cross_attr2.reshape(batch_size, -1, width2, width2)
        output3 = self.deconv3(output2, z_cross_attr2) ## 16 > 32
        # adain_3 = AdaIN_layer(id_emb.size()[1], output3.size()[1])
        # output3 = adain_3(output3, id_emb) if 3<= self.num_adain else output3 # conditional injection
        output3 = self.AdaIN_layer3(output3, id_emb) if 1<= self.num_adain else output3 # conditional injection

        # print('output3:', output3.shape)
        
        ##256x32x32 -> 128x64x64 (output4)
        z_cross_attr3 = z_cross_attr3.reshape(batch_size, -1, width3, width3)
        output4 = self.deconv4(output3, z_cross_attr3) ## 32 > 64
        # adain_4 = AdaIN_layer(id_emb.size()[1], output4.size()[1])
        # output4 = adain_4(output4, id_emb) if 4<= self.num_adain else output4 # conditional injection
        output4 = self.AdaIN_layer4(output4, id_emb) if 1<= self.num_adain else output4 # conditional injection

        # print('output4:', output4.shape)
        
        # ##128x64x64 -> 64x128x128 (output5)
        # output5 = self.deconv5(output4, z_cross_attr4) ## 64 > 128
        # ##64x128x128 -> 32x256x256 (output6)
        # output6 = self.deconv6(output5, z_cross_attr5) ## 128 > 256
        
        ##128x64x64 -> 64x128x128 (output5)
        # z_cross_attr4 = z_cross_attr4.reshape(batch_size, -1, width4, width4)
        output5 = self.deconv5(output4) ## 64 > 128
        # adain_5 = AdaIN_layer(id_emb.size()[1], output5.size()[1])
        # output5 = adain_5(output5, id_emb) if 5<= self.num_adain else output5 # conditional injection
        output5 = self.AdaIN_layer5(output5, id_emb) if 1<= self.num_adain else output5

        # print('output5:', output5.shape)
        
        ##64x128x128 -> 32x256x256 (output6)
        # z_cross_attr5 = z_cross_attr5.reshape(batch_size, -1, width5, width5)
        output6 = self.deconv6(output5) ## 128 > 256
        # adain_6 = AdaIN_layer(id_emb.size()[1], output6.size()[1])
        # output6 = adain_6(output6, id_emb) if 6<= self.num_adain else output6 # conditional injection
        output6 = self.AdaIN_layer6(output6, id_emb) if 1<= self.num_adain else output6
        # print('output6:', output6.shape)
        
        
        ##32x256x256 -> 3x256x256 (output7)
        # z_cross_attr6 = z_cross_attr6.reshape(batch_size, -1, width6, width6)
        output7 = self.deconv7(output6) ## 128 > 256
        # print('output7:', output7.shape)
        
        ##output6 = Final output of image tensor
        ##z_src_attr7 = Final image shape output of src unet
        ##z_tgt_attr7 = Final image shape output of tgt unet
        
        return torch.tanh(output7), torch.tanh(src_z_attr7), torch.tanh(tgt_z_attr7)


    # def ca_forward(self, target, source, mixed_id_embedding): ##cross attention forward
    def ca_forward(self, target, source, id_emb): ##cross attention forward
            '''
            src나 tgt나 다 똑같다.
            src_bottlneck_attr = 1024x4x4
            src_z_attr1 = 1024x8x8
            src_z_attr2 = 512x16x16
            src_z_attr3 = 256x32x32
            src_z_attr4 = 128x64x64
            src_z_attr5 = 64x128x128
            src_z_attr6 = 32x256x256
            src_z_attr7 = 3x256x256
            
            output:
            'output7, src_z_attr7, tgt_z_attr7'
            
            output7: The final output of CrossAttention Generator
            src_z_attr7: The final output of source face Unet
            tgt_z_attr7: The final output of target face Unet
            '''
            src_bottlneck_attr, src_z_attr1, src_z_attr2, src_z_attr3, src_z_attr4, src_z_attr5, src_z_attr6, src_z_attr7 = self.CUMAE_src(source)
            tgt_bottlneck_attr, tgt_z_attr1, tgt_z_attr2, tgt_z_attr3, tgt_z_attr4, tgt_z_attr5, tgt_z_attr6, tgt_z_attr7 = self.CUMAE_tgt(target) ##z_tgt_attr1 img_size 8  z_tgt_attr6 img_size 256

            width0= tgt_bottlneck_attr.shape[2]
            batch_size= tgt_bottlneck_attr.shape[0]
            width1= tgt_z_attr1.shape[2]
            width2= tgt_z_attr2.shape[2]
            width3= tgt_z_attr3.shape[2]
            width4= tgt_z_attr4.shape[2]
            width5= tgt_z_attr5.shape[2]
            width6= tgt_z_attr6.shape[2]
            width7= tgt_z_attr7.shape[2]
            
            
            
            ##z_cross_attr0 = 1024x4x4  this is same as bottleneck block
            z_cross_attr0 = self.FFCA0(tgt_bottlneck_attr, src_bottlneck_attr)


            #z_cross_attr1 = [B, 64, 1024]
            z_cross_attr1 = self.FFCA1(tgt_z_attr1, src_z_attr1)  ##tgt_z_attr1 = [B, 1024, 8, 8]
            ##512x16x16 
            z_cross_attr2 = self.FFCA2(tgt_z_attr2, src_z_attr2)
            ##256x32x32
            z_cross_attr3 = self.FFCA3(tgt_z_attr3, src_z_attr3)
            ##128x64x64
            z_cross_attr4 = self.FFCA4(tgt_z_attr4, src_z_attr4)
            # print('z_cross_attr4 passed')
            # return z_cross_attr4
            #64x128x128
            # z_cross_attr5 = self.FFCA5(tgt_z_attr5, src_z_attr5)
            #32x256x256
            # z_cross_attr6 = self.FFCA6(tgt_z_attr6, src_z_attr6)
            #3x256x256
            # z_cross_attr7 = self.FFCA7(tgt_z_attr7, src_z_attr7)

            
            ##1024x4x4 -> 1024x8x8 (output1)
            z_cross_attr0 = z_cross_attr0.reshape(batch_size, -1, width0, width0)  ## z_cross_attr0 becomes [B, 1024, 8, 8]
            output1 = self.deconv1(z_cross_attr0) ## 4 > 8  ##스킵커넥션없이 conv만 해서 키운것. output1 = [B, 1024, 8, 8]
            output1 = self.AdaIN_layer1(output1, id_emb) if 1<= self.num_adain else output1
            ##1024x8x8 -> 512x16x16 (output2)
            
            ## z_cross_attr1 = [B, seq_len, dim] = (batch_size, 64, 1024)
            ## output1 =  [B, dim, height, width] = (baych, 1024, 8, 8)
            ## here, we should make z_cross_attr1 shape same as output 1 
            z_cross_attr1 = z_cross_attr1.reshape(batch_size, -1, width1, width1)
            # print('reshaped z_cross_attr1:', z_cross_attr1.shape)
            output2 = self.deconv2(output1, z_cross_attr1) ## 8 > 16
            output2 = self.AdaIN_layer2(output2, id_emb) if 1<= self.num_adain else output2
            # print('output2:', output2.shape)
            
            ##512x16x16 -> 256x32x32 (output3)
            z_cross_attr2 = z_cross_attr2.reshape(batch_size, -1, width2, width2)
            output3 = self.deconv3(output2, z_cross_attr2) ## 16 > 32
            output3 = self.AdaIN_layer3(output3, id_emb) if 1<= self.num_adain else output3
            # print('output3:', output3.shape)
            
            ##256x32x32 -> 128x64x64 (output4)
            z_cross_attr3 = z_cross_attr3.reshape(batch_size, -1, width3, width3)
            output4 = self.deconv4(output3, z_cross_attr3) ## 32 > 64
            output4 = self.AdaIN_layer4(output4, id_emb) if 1<= self.num_adain else output4
            # print('output4:', output4.shape)
            
            # ##128x64x64 -> 64x128x128 (output5)
            # output5 = self.deconv5(output4, z_cross_attr4) ## 64 > 128
            # ##64x128x128 -> 32x256x256 (output6)
            # output6 = self.deconv6(output5, z_cross_attr5) ## 128 > 256
            
            ##128x64x64 -> 64x128x128 (output5)
            # z_cross_attr4 = z_cross_attr4.reshape(batch_size, -1, width4, width4)
            output5 = self.deconv5(output4) ## 64 > 128
            output5 = self.AdaIN_layer5(output5, id_emb) if 1<= self.num_adain else output5
            # print('output5:', output5.shape)
            
            ##64x128x128 -> 32x256x256 (output6)
            # z_cross_attr5 = z_cross_attr5.reshape(batch_size, -1, width5, width5)
            output6 = self.deconv6(output5) ## 128 > 256
            output6 = self.AdaIN_layer6(output6, id_emb) if 1<= self.num_adain else output6
            # print('output6:', output6.shape)
            
            
            ##32x256x256 -> 3x256x256 (output7)
            # z_cross_attr6 = z_cross_attr6.reshape(batch_size, -1, width6, width6)
            output7 = self.deconv7(output6) ## 128 > 256
            # print('output7:', output7.shape)
            
            ##output6 = Final output of image tensor
            ##z_src_attr7 = Final image shape output of src unet
            ##z_tgt_attr7 = Final image shape output of tgt unet
            
            return z_cross_attr0, output1, output2, output3, output4, output5, output6, output7
            
            
    def src_unet_forward(self, source):
                '''
                src나 tgt나 다 똑같다.
                src_bottlneck_attr = 1024x4x4
                src_z_attr1 = 1024x8x8
                src_z_attr2 = 512x16x16
                src_z_attr3 = 256x32x32
                src_z_attr4 = 128x64x64
                src_z_attr5 = 64x128x128
                src_z_attr6 = 32x256x256
                src_z_attr7 = 3x256x256
                
                output:
                'output7, src_z_attr7, tgt_z_attr7'
                
                output7: The final output of CrossAttention Generator
                src_z_attr7: The final output of source face Unet
                tgt_z_attr7: The final output of target face Unet
                '''
                src_bottlneck_attr, src_z_attr1, src_z_attr2, src_z_attr3, src_z_attr4, src_z_attr5, src_z_attr6, src_z_attr7 = self.CUMAE_src(source)
                return src_bottlneck_attr, src_z_attr1, src_z_attr2, src_z_attr3, src_z_attr4, src_z_attr5, src_z_attr6, src_z_attr7



    def tgt_unet_forward(self, target):
                '''
                src나 tgt나 다 똑같다.
                src_bottlneck_attr = 1024x4x4
                src_z_attr1 = 1024x8x8
                src_z_attr2 = 512x16x16
                src_z_attr3 = 256x32x32
                src_z_attr4 = 128x64x64
                src_z_attr5 = 64x128x128
                src_z_attr6 = 32x256x256
                src_z_attr7 = 3x256x256
                
                output:
                'output7, src_z_attr7, tgt_z_attr7'
                
                output7: The final output of CrossAttention Generator
                src_z_attr7: The final output of source face Unet
                tgt_z_attr7: The final output of target face Unet
                '''
                tgt_bottlneck_attr, tgt_z_attr1, tgt_z_attr2, tgt_z_attr3, tgt_z_attr4, tgt_z_attr5, tgt_z_attr6, tgt_z_attr7 = self.CUMAE_tgt(target)
                return tgt_bottlneck_attr, tgt_z_attr1, tgt_z_attr2, tgt_z_attr3, tgt_z_attr4, tgt_z_attr5, tgt_z_attr6, tgt_z_attr7