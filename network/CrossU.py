
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models. import FlowFaceCrossAttentionModel
from network.CrossU import CrossUMLAttrEncoder

class CrossUnetAttentionModel(nn.Module):
    def __init__(self, source, target, seq_len, n_head, q_dim, k_dim, kv_dim):
        super(CrossUnetAttentionModel, self).__init__()
        # (self, seq_len: int, n_head: int, k_dim: int, q_dim: int, kv_dim: int):
        FFCA1 = FlowFaceCrossAttentionModel(seq_len=64, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        FFCA2 = FlowFaceCrossAttentionModel(seq_len=256, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        FFCA3 = FlowFaceCrossAttentionModel(seq_len=, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        FFCA4 = FlowFaceCrossAttentionModel(seq_len=64, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        FFCA5 = FlowFaceCrossAttentionModel(seq_len=64, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        FFCA6 = FlowFaceCrossAttentionModel(seq_len=64, n_head=2, q_dim=1024, k_dim=1024, kv_dim=1024)
        
        CUMAE_src = CrossUMLAttrEncoder(...)
        CUMAE_tgt = CrossUMLAttrEncoder(...)
        
    def forward(self,source, target):
        z_src_attr1, z_src_attr2, z_src_attr3, z_src_attr4, z_src_attr5, z_src_attr6 = CUMAE_src(source)
        z_tgt_attr1, z_tgt_attr2, z_tgt_attr3, z_tgt_attr4, z_tgt_attr5, z_tgt_attr6 = CUMAE_tgt(target) ##z_tgt_attr1 img_size 8  z_tgt_attr6 img_size 256
        
        
        FFCA()