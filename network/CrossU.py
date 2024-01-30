
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import FlowFaceCrossAttentionModel
from network.FFGenerator import UNet, deconv4x4, noskip_deconv4x4, outconv

class CrossUnetAttentionGenerator(nn.Module):
    # def __init__(self, seq_len, n_head, q_dim, k_dim, kv_dim, backbone='unet'):
    def __init__(self, seq_len:int=16, n_head:int=2, q_dim:int=1024, k_dim:int=1024, kv_dim:int=1024, backbone='unet'):
        super(CrossUnetAttentionGenerator, self).__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.n_head = n_head
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.kv_dim = kv_dim

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
        
        # ##noskip_deconv4x4을 하면 crossattention을 안해도 되니 컴퓨팅을 아낄수있다
        # self.deconv1 = noskip_deconv4x4(1024, 1024)  ## 4 ->  8 (w,h)
        # self.deconv2 = deconv4x4(1024, 512)  ## 8 > 16
        # self.deconv3 = deconv4x4(512, 256) ## 16 > 32
        # self.deconv4 = deconv4x4(256, 128) ## 32 > 64
        # self.deconv5 = noskip_deconv4x4(128, 64) ## 64 > 128
        # self.deconv6 = noskip_deconv4x4(64, 32) ## 128 > 256
        # self.deconv7 = outconv(32, 3) ## 128 > 256
        
        self.deconv1 = noskip_deconv4x4(self.q_dim, self.q_dim)  ## 4 ->  8 (w,h)
        self.deconv2 = deconv4x4(self.q_dim, self.q_dim//2)  ## 8 > 16
        self.deconv3 = deconv4x4(self.q_dim//2, self.q_dim//4) ## 16 > 32
        self.deconv4 = deconv4x4(self.q_dim//4, self.q_dim//8) ## 32 > 64
        self.deconv5 = noskip_deconv4x4(self.q_dim//8, self.q_dim//16) ## 64 > 128
        self.deconv6 = noskip_deconv4x4(self.q_dim//16, self.q_dim//32) ## 128 > 256
        self.deconv7 = outconv(self.q_dim//32, 3) ## 128 > 256
        


    def forward(self, source, target):
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
        
        
        ##z_cross_attr0 = 1024x4x4  this is same as bottleneck block
        z_cross_attr0 = self.FFCA0(tgt_bottlneck_attr, src_bottlneck_attr)
        
        return z_cross_attr0
        ##1024x8x8
        # z_cross_attr1 = self.FFCA1(tgt_z_attr1, src_z_attr1)
        # ##512x16x16 
        # z_cross_attr2 = self.FFCA2(tgt_z_attr2, src_z_attr2)
        # ##256x32x32
        # z_cross_attr3 = self.FFCA3(tgt_z_attr3, src_z_attr3)
        # ##128x64x64
        # z_cross_attr4 = self.FFCA4(tgt_z_attr4, src_z_attr4)
        # #64x128x128
        # # z_cross_attr5 = self.FFCA5(tgt_z_attr5, src_z_attr5)
        # #32x256x256
        # # z_cross_attr6 = self.FFCA6(tgt_z_attr6, src_z_attr6)
        # #3x256x256
        # # z_cross_attr7 = self.FFCA7(tgt_z_attr7, src_z_attr7)
        
        
        # ##1024x4x4 -> 1024x8x8 (output1)
        # output1 = self.deconv1(z_cross_attr0) ## 4 > 8  ##스킵커넥션없이 conv만 해서 키운것 
        # ##1024x8x8 -> 512x16x16 (output2)
        # output2 = self.deconv2(output1, z_cross_attr1) ## 8 > 16
        # ##512x16x16 -> 256x32x32 (output3)
        # output3 = self.deconv3(output2, z_cross_attr2) ## 16 > 32
        # ##256x32x32 -> 128x64x64 (output4)
        # output4 = self.deconv4(output3, z_cross_attr3) ## 32 > 64
        
        # # ##128x64x64 -> 64x128x128 (output5)
        # # output5 = self.deconv5(output4, z_cross_attr4) ## 64 > 128
        # # ##64x128x128 -> 32x256x256 (output6)
        # # output6 = self.deconv6(output5, z_cross_attr5) ## 128 > 256
        
        # ##128x64x64 -> 64x128x128 (output5)
        # output5 = self.deconv5(output4) ## 64 > 128
        # ##64x128x128 -> 32x256x256 (output6)
        # output6 = self.deconv6(output5) ## 128 > 256
        # ##32x256x256 -> 3x256x256 (output7)
        # output7 = self.deconv7(output6) ## 128 > 256
        
        # ##output6 = Final output of image tensor
        # ##z_src_attr7 = Final image shape output of src unet
        # ##z_tgt_attr7 = Final image shape output of tgt unet
        
        # return output7, src_z_attr7, tgt_z_attr7 
        