import torch
import torch.nn as nn
from extractor.arcface_model.iresnet import iresnet100
import sys
from utils.deep3d import *
import torch.nn.functional as F
sys.path.append('./extractor/')
#sys.path.append('./models/')
#sys.path.append('./models/deep3D')
#sys.path.append('./models/deep3D/models')


class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self, f_3d_checkpoint_path, f_id_checkpoint_path):
        super(ShapeAwareIdentityExtractor, self).__init__()
        
        self.device = torch.device(0)



        self.f_3d = torch.load(f_3d_checkpoint_path)
        self.f_3d = self.f_3d.to(self.device)
        self.f_3d.eval()

        #print("hurray")
        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location='cpu'))
        ##self.f_id = self.f_id.to(self.device) ##
        self.f_id.eval()


    @torch.no_grad()
    def forward(self, i_source, i_target):


        ## v_sid
        lm3d_std = load_lm3d("./deep3D/BFM") 
        source_img, lm_src = read_data(i_source, lm3d_std)

        source_img = source_img.to(self.device) 
        c_s = self.f_3d(source_img)

        target_img, lm_src = read_data(i_target, lm3d_std)
        target_img = target_img.to(self.device)

        c_t = self.f_3d(target_img)
        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
        v_id_source = F.normalize(self.f_id(F.interpolate((i_source - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)
        v_sid = torch.cat((c_fuse, v_id_source), dim=1)

        ## source arc face only embedding (id loss와 infonce loss에서 사용)
        src_emb = v_id_source

        ## target arc face only embedding (infonce loss에서 사용)
        v_id_target = F.normalize(self.f_id(F.interpolate((i_target - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)
        tgt_emb = v_id_target



        #print(f"v_sid shape : {v_sid.size()}, src_emb shape : {src_emb.size()}, tgt_emb shape : {tgt_emb.size()}")
        return v_sid, src_emb, tgt_emb
    
    
    @torch.no_grad()
    def id_forward(self, i_swapped): # swapped의 arc face only embedding(id loss와 Infonce loss에서 사용)을 위한 method

        i_swapped = i_swapped.cpu()
        swapped_emb = F.normalize(self.f_id(F.interpolate((i_swapped - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)


        return swapped_emb