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
from deep3D.models.bfm import ParametricFaceModel

class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self, f_3d_checkpoint_path, f_id_checkpoint_path, mixed_precision, mode = 'hififace'):
        super(ShapeAwareIdentityExtractor, self).__init__()
        
        # self.device = torch.device(0)
        self.mode = mode # binary : hififace / arcface
        self.mixed_precision = mixed_precision
        self.f_3d = torch.load(f_3d_checkpoint_path)
        # self.f_3d = self.f_3d.to(self.device)
        self.f_3d.eval()
        if self.mixed_precision:  
            self.f_id = iresnet100(pretrained=False, fp16=True)
        else:
            self.f_id = iresnet100(pretrained=False, fp16=False)
                
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location='cpu'))
        self.f_id.eval()

        self.face_model = ParametricFaceModel()
        

    @torch.no_grad()
    def forward(self, i_source, i_target):

        ## v_sid
        lm3d_std = load_lm3d("./deep3D/BFM") 
        #print("i_source shape : ", i_source.size())
        #source_img, lm_src = read_data(i_source, lm3d_std) 1) 256 -> 224 crop 2) align
        #source_img = source_img.to(self.device) 
        #print("source img shape : ", source_img.size())
        source_img = F.interpolate(i_source, size=224, mode='bilinear') # Instead of alinging & cropping, simply resize assuming input images are well cropped
                                                                        # 1) Align 과정에서 쓰는 landmark를 98개 쓰는 것으로 대체 -> 당장은 어려워보임
                                                                        # 2) foreground만 input으로 넣기 -> deep3d값 잘 나오는진 호준님거랑 싱크 맞추고 테스트해야할듯
        
        # for testing
        tensor_to_pil = transforms.ToPILImage()
        im = tensor_to_pil(source_img[0]).convert('RGB')
        im.save("/workspace/images/resized.jpg")


        c_s = self.f_3d(source_img)#.to(self.device))
        #print(c_s)

        #target_img, lm_src = read_data(i_target, lm3d_std)
        #target_img = target_img.to(self.device)
        target_img = F.interpolate(i_target, size=224, mode='bilinear')
        c_t = self.f_3d(target_img)#.to(self.device))

        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        v_id_source = F.normalize(self.f_id(F.interpolate((i_source - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)#.to(self.device)
        v_id_target = F.normalize(self.f_id(F.interpolate((i_target - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)#.to(self.device)
        
        v_sid = torch.cat((c_fuse, v_id_source), dim=1)




        if self.mode == 'hififace': 
            '''
            return :
                v_sid : source의 arcface + (source의 deep3d vector 중 identity + target의 " 중 the rest)
                src_emb : source의 arcface + source의 deep3d
                tgt_emb : target의 arcface + target의 deep3d
            '''

            src_emb = torch.cat((c_s, v_id_source), dim=1)
            tgt_emb = torch.cat((c_t, v_id_target), dim=1)

            return v_sid, src_emb, tgt_emb
        

        elif self.mode == 'arcface':
            '''
            return :
                v_sid : source의 arcface + (source의 deep3d vector 중 identity + target의 " 중 the rest)
                src_emb : source의 arcface
                tgt_emb : target의 arcface
            '''

            src_emb = v_id_source
            tgt_emb = v_id_target

            return v_sid, src_emb, tgt_emb

        else:
            raise Exception("Illegal argument. Use either \'hififace\' or \'arcface\'")


    @torch.no_grad()
    def id_forward(self, i_swapped): # swapped의 arc face only embedding(id loss와 Infonce loss에서 사용)을 위한 method

        lm3d_std = load_lm3d("./deep3D/BFM") 


        if self.mode == 'hififace': 
#            print("id_forward for hififace !!!!!!!!!!")

            i_swapped = i_swapped#.cpu()
            swapped_emb = F.normalize(self.f_id(F.interpolate((i_swapped - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)#.to(self.device)


            ### i_swapped must be an image ###
            #swapped_img, lm_swapped = read_data(i_swapped, lm3d_std)
            #swapped_img = swapped_img.to(self.device) 
            swapped_img = F.interpolate(i_swapped, size=224, mode='bilinear')
            c_swapped = self.f_3d(swapped_img)#.to(self.device))
            v_swapped = torch.cat((c_swapped, swapped_emb), dim=1)
            return v_swapped

        elif self.mode == 'arcface': 

            i_swapped = i_swapped#.cpu()
            swapped_emb = F.normalize(self.f_id(F.interpolate((i_swapped - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)#.to(self.device)
            return swapped_emb

        else:
            raise Exception("Illegal argument. Use either \'hififace\' or \'arcface\'")
        
    
    @torch.no_grad()
    def shapeloss_forward(self, i_source, i_target, i_swapped): # shapeloss score를 계산하기 위한 method. 실제 shape_loss function에서도 이 method를 이용한다.
                                            # i_source, i_target, i_swapped이 형태?가 같아야 한다. tensor상태이던 image 상태이던.

        #if self.mode == 'arcface': # arcface mode라는 것은 아직 epoch이 2보다 작다는 뜻 -> shape loss 사용 안함
        #    return torch.zeros(2, 68, 2).to(self.device), torch.zeros(2, 68, 2).to(self.device)

        lm3d_std = load_lm3d("./deep3D/BFM") 

        #source_img, lm_src = read_data(i_source, lm3d_std)
        #source_img = source_img.to(self.device) 
        source_img = F.interpolate(i_source, size=224, mode='bilinear')
        c_s = self.f_3d(source_img)#.to(self.device))

        #target_img, lm_tgt = read_data(i_target, lm3d_std)
        #target_img = target_img.to(self.device)
        target_img = F.interpolate(i_target, size=224, mode='bilinear')
        c_t = self.f_3d(target_img)#.to(self.device))

        #swapped_img, lm_swapped = read_data(i_swapped, lm3d_std)
        #swapped_img = swapped_img.to(self.device)
        #c_r = self.f_3d(F.interpolate(swapped_img, size=224, mode='bilinear')) # 이렇게 그냥 input으로 들어오는 hojun님으로부터 1차로 crop된 image의 foreground를 resize만 해서 바로 넣어봤을 때 어떨지 테스트해봐야함.
        
        #c_r = self.f_3d(swapped_img) #test를 위해서 잠시 꺼놓기

        #c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        '''
        (B, 257)
        80 # id layer
        64 # exp layer
        80 # tex layer
        3  # angle layer
        27 # gamma layer
        2  # tx, ty
        1  # tz
        '''

        with torch.no_grad():
            c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)#.to(self.device)
            _, _, _, q_fuse = self.face_model.compute_for_render(c_fuse)
        #_, _, _, q_r = self.face_model.compute_for_render(c_r) #test를 위해서 잠시 꺼놓기

        #return q_fuse, q_r
        return q_fuse, q_fuse

    # @torch.no_grad()
    # def forward(self, i_source, i_target):


    #     ## v_sid
    #     lm3d_std = load_lm3d("./deep3D/BFM") 
    #     source_img, lm_src = read_data(i_source, lm3d_std)

    #     source_img = source_img.to(self.device) 
    #     c_s = self.f_3d(source_img)

    #     target_img, lm_src = read_data(i_target, lm3d_std)
    #     target_img = target_img.to(self.device)

    #     c_t = self.f_3d(target_img)
    #     c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
    #     v_id_source = F.normalize(self.f_id(F.interpolate((i_source - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)
    #     v_sid = torch.cat((c_fuse, v_id_source), dim=1)

    #     ## source arc face only embedding (id loss와 infonce loss에서 사용)
    #     src_emb = v_id_source

    #     ## target arc face only embedding (infonce loss에서 사용)
    #     v_id_target = F.normalize(self.f_id(F.interpolate((i_target - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)
    #     tgt_emb = v_id_target



    #     #print(f"v_sid shape : {v_sid.size()}, src_emb shape : {src_emb.size()}, tgt_emb shape : {tgt_emb.size()}")
    #     return v_sid, src_emb, tgt_emb
    
    
    # @torch.no_grad()
    # def id_forward(self, i_swapped): # swapped의 arc face only embedding(id loss와 Infonce loss에서 사용)을 위한 method



    #     i_swapped = i_swapped.cpu()
    #     swapped_emb = F.normalize(self.f_id(F.interpolate((i_swapped - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2).to(self.device)


    #     return swapped_emb