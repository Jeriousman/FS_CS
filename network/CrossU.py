
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import FlowFaceCrossAttentionModel
from network.FFGenerator import CrossUMLAttrEncoder, deconv4x4, noskip_deconv4x4

class CrossUnetAttentionGenerator(nn.Module):
    # def __init__(self, seq_len, n_head, q_dim, k_dim, kv_dim, backbone='unet'):
    def __init__(self, backbone='unet'):
        super(CrossUnetAttentionGenerator, self).__init__()
        self.backbone = backbone
        # self.seq_len = seq_len
        # self.n_head = n_head
        # self.q_dim = q_dim
        # self.k_dim = k_dim
        # self.kv_dim = kv_dim
        
        # (self, seq_len: int, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        self.FFCA1 = FlowFaceCrossAttentionModel(seq_len=16, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        self.FFCA2 = FlowFaceCrossAttentionModel(seq_len=64, n_head=2, q_dim=512, k_dim=512, kv_dim=512)
        self.FFCA3 = FlowFaceCrossAttentionModel(seq_len=256, n_head=2, q_dim=256, k_dim=256, kv_dim=256)
        self.FFCA4 = FlowFaceCrossAttentionModel(seq_len=1024, n_head=2, q_dim=128, k_dim=128, kv_dim=128)
        self.FFCA5 = FlowFaceCrossAttentionModel(seq_len=4096, n_head=2, q_dim=64, k_dim=64, kv_dim=64)
        # self.FFCA6 = FlowFaceCrossAttentionModel(seq_len=16384, n_head=2, q_dim=32, k_dim=32, kv_dim=32)
        # self.FFCA7 = FlowFaceCrossAttentionModel(seq_len=65536, n_head=2, q_dim=3, k_dim=3, kv_dim=3)  ##computationally expensive
        
        
        self.CUMAE_src = CrossUMLAttrEncoder(backbone='unet')
        self.CUMAE_tgt = CrossUMLAttrEncoder(backbone='unet')
        # CUMAE_cross = CrossUMLAttrEncoder(backbone='unet')


        self.deconv1 = noskip_deconv4x4(1024, 1024)  ## 4 ->  8 (w,h)
        self.deconv2 = deconv4x4(1024, 512)  ## 8 > 16
        self.deconv3 = deconv4x4(1536, 256) ## 16 > 32
        self.deconv4 = deconv4x4(768, 128) ## 32 > 64
        self.deconv5 = deconv4x4(384, 64) ## 64 > 128
        self.deconv6 = noskip_deconv4x4(192, 3) ## 128 > 256



    def forward(self,source, target):
        '''
        src나 tgt나 다 똑같다.
        z_src_attr1 = 1024x4x4
        z_src_attr2 = 1024x8x8
        z_src_attr3 = 512x16x16
        z_src_attr4 = 256x32x32
        z_src_attr5 = 128x64x64
        z_src_attr6 = 64x128x128
        z_src_attr7 = 3x256x256
        
        '''
        z_src_attr1, z_src_attr2, z_src_attr3, z_src_attr4, z_src_attr5, z_src_attr6, z_src_attr7 = self.CUMAE_src(source)
        z_tgt_attr1, z_tgt_attr2, z_tgt_attr3, z_tgt_attr4, z_tgt_attr5, z_tgt_attr6, z_tgt_attr7 = self.CUMAE_tgt(target) ##z_tgt_attr1 img_size 8  z_tgt_attr6 img_size 256
        
        
        ##1024x4x4
        z_cross_attr1 = self.FFCA1(z_tgt_attr1, z_src_attr1)
        ##1024x8x8
        z_cross_attr2 = self.FFCA2(z_tgt_attr2, z_src_attr2)
        ##512x16x16 
        z_cross_attr3 = self.FFCA3(z_tgt_attr3, z_src_attr3)
        ##256x32x32
        z_cross_attr4 = self.FFCA4(z_tgt_attr4, z_src_attr4)
        ##128x64x64
        z_cross_attr5 = self.FFCA5(z_tgt_attr5, z_src_attr5)
        ##64x128x128
        # z_cross_attr6 = self.FFCA6(z_tgt_attr6, z_src_attr6)
        ##3x256x256
        # z_cross_attr7 = self.FFCA7(z_tgt_attr7, z_src_attr7)
        
        ##1024x4x4 -> 1024x8x8 (output1)
        output1 = self.deconv1(z_cross_attr1) ## 4 > 8  ##feat4 = skip connection the same size with same level counterpart
        ##1024x8x8 -> 512x8x8 + 1024x8x8 -> 1536x16x16 (output2)
        output2 = self.deconv2(output1, z_cross_attr2) ## 8 > 16
        ##(1536x16x16 -> 256x16x16) + 512x16x16  -> 768x32x32 (output3)
        output3 = self.deconv3(output2, z_cross_attr3) ## 16 > 32
        ##(768x32x32 -> 128x32x32) + 256x32x32 -> 384x64x64 (output4)
        output4 = self.deconv4(output3, z_cross_attr4) ## 32 > 64
        ##(384x64x64 -> 64x64x64) + 128x64x64 -> 192x128x128 (output5)
        output5 = self.deconv5(output4, z_cross_attr5) ## 64 > 128
        ##192x128x128 -> 3x256x256
        output6 = self.deconv6(output5) ## 128 > 256
        
        ##output6 = Final output of image tensor
        ##z_src_attr7 = Final image shape output of src unet
        ##z_tgt_attr7 = Final image shape output of tgt unet
        
        return output6, z_src_attr7, z_tgt_attr7 
        