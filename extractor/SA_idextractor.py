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

        print("hurray")
        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location='cpu'))
        self.f_id = self.f_id.to(self.device) ##
        self.f_id.eval()


    @torch.no_grad()
    def forward(self, i_source, i_target):

        lm3d_std = load_lm3d("./deep3D/BFM") 

        # preprocess for deep3d against i_source

        ## i_source and i_target are batches ##

        source_img, lm_src = read_data(i_source, lm3d_std)

        source_img = source_img.to(self.device) 
        print("source_img shape : ", source_img.size())

        c_s = self.f_3d(source_img)

        # preprocess for deep3d against i_target
        target_img, lm_src = read_data(i_target, lm3d_std)
        target_img = target_img.to(self.device)

        c_t = self.f_3d(target_img)

        
        # fusion!
        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        # preprocess for arcface against i_source
        #v_id = F.normalize(self.f_id(F.interpolate((i_source - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)

        #v_sid = torch.cat((c_fuse, v_id), dim=1)
        #return v_sid
        return c_fuse