import sys
import argparse
import time
import cv2
import wandb
from PIL import Image
import metric
import os

#For Native Torch multi GPUs
import datetime
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group


from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduler

## custom imports
# sys.path.append('./apex/')
# from apex import amp

from network.CrossU import CrossUnetAttentionGenerator, UNet
from extractor.SA_idextractor import ShapeAwareIdentityExtractor
# from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.training.Dataset import FaceEmbedCombined, FaceEmbed, FaceEmbedSubdir, FaceEmbedFFHQ, FaceEmbedCelebA, FaceEmbedCustom#FaceEmbedAllFlat
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.losses import hinge_loss, compute_discriminator_loss, compute_generator_losses
from utils.training.detector import detect_landmarks, paint_eyes
from utils.training.landmark_detector import detect_all_landmarks
from AdaptiveWingLoss.core import models
from arcface_model.iresnet import iresnet100
from models.model import FlowFaceCrossAttentionModel, FlowFaceCrossAttentionLayer







def train_one_epoch(G: 'generator model', 
                    D: 'discriminator model', 
                    id_extractor: 'id_extractor model',
                    opt_G: "generator opt", 
                    opt_D: "discriminator opt",
                    scheduler_G: "scheduler G opt",
                    scheduler_D: "scheduler D opt",
                    netArc: 'ArcFace model',
                    model_ft: 'Landmark Detector',
                    args: 'Args Namespace',
                    train_dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int,
                    starting_iteration: 'iteration currently at', 
                    loss_adv_accumulated:int,
                    config:dict
                    ):
    

    
    # ##loading pretrained models for extracting IDs
    # f_3d_path = "/datasets/pretrained/pretrained_model.pth"
    # f_id_path = "/datasets/pretrained/backbone.pth"
    
    # id_extractor = ShapeAwareIdentityExtractor(f_3d_path, f_id_path, args.id_mode).to(args.device)
    # id_extractor = DistributedDataParallel(id_extractor, device_ids=[config['local_rank']])
    # #print(id_extractor)

    ## mixed-precision을 사용하기 위해 amp 라이브러리를 사용한다. fp32 -> fp16 으로 계산하기 위함이다. 그러나 현재 fp16을 사용하면 gradient가 깨지는 현상이 발생하고 있기 때문에, 사용하지말고 사용하기 위해 코드 수정을 해야할 것으로 보인다.
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=False)  
        

    for iteration, data in enumerate(train_dataloader):
        start_time = time.time()
        iteration  = starting_iteration + iteration
        
        ##amp mixed-precision을 사용하는 경우:
        if args.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): ##input data는 일단 그대로 flaot32이지만 input이 계산될때 + output은 float16이된다
                '''
                ##id_ext_src_input과 id_ext_tgt_input은 shape aware identity extractor (저스틴 파트)의 인풋으로 사용된다.
                Xt_f는 target face 인풋의 foreground, Xt_b는 target face 인풋의 background 데이터이다.
                Xs_f는 source face 인풋의 foreground, Xs_b는 source face 인풋의 background 데이터이다.
                same_person은 Ghost의 코드에서 온 것인데, 정확한 의미는 모르겠으나 일단 남겨두었다. (큰 의미 없는 것으로 보임)
                '''
                id_ext_src_input, id_ext_tgt_input, Xt_f, Xt_b, Xs_f, Xs_b, same_person = data 
                id_ext_src_input = id_ext_src_input.to(args.device)
                id_ext_tgt_input = id_ext_tgt_input.to(args.device)
                Xs_f = Xs_f.to(args.device)
                Xt_f = Xt_f.to(args.device)
                same_person = same_person.to(args.device)
                realtime_batch_size = Xt_f.shape[0] 

                with torch.autocast(device_type="cuda", enabled=False):
                # with torch.autocast(device_type="cuda", dtype=torch.float16):  ## 이거  때문에 개망하기때문에 나중에 없애고 인덴트 들여써라  
                    mixed_id_embedding, src_id_emb, tgt_id_emb = id_extractor.module.forward(id_ext_src_input.float(), id_ext_tgt_input.float()) ## id_embedding = [B, 769]
                    # print('mixed_id_embedding', mixed_id_embedding.dtype) ##mixed_id_embedding is float32 bcuz of enabled=False and id_ext_src_input.float()..

                diff_person = torch.ones_like(same_person)
                if args.diff_eq_same:
                    same_person = diff_person

                # generator training
                opt_G.zero_grad() ##generator optimizer의 축적된 gradients를 비워준다
                swapped_face, recon_f_src, recon_f_tgt = G.module.forward(Xt_f, Xs_f, mixed_id_embedding) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다.
                Xt_f_attrs = G.module.CUMAE_tgt(Xt_f) # UNet으로 Xt의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?
    
                with torch.autocast(device_type="cuda", enabled=False):
                    ##swapped_id_emb = ArcFace value. this embedding is for infoNCE loss mostly
                    swapped_id_emb = id_extractor.module.id_forward(swapped_face.float())

                if args.shape_loss:  ##저스틴이 개발한 shape aware identity extractor의 shape loss는 여기서 계산한다.
                    with torch.autocast(device_type="cuda", enabled=False):
                        q_fuse, q_r = id_extractor.module.shapeloss_forward(id_ext_src_input.float(), id_ext_tgt_input.float(), swapped_face.float())  # Y가 network의 output tensor에 denorm까지 되었다고 가정 & q_r은 지금 당장 잡아낼 수가 없으므로(swap 결과가 초반엔 별로여서) 당장은 q_fuse를 똑같이 씀
                else:
                    pass
                    q_fuse, q_r = 0, 0

                # Y, recon_f_src, recon_f_tgt = G(Xt, Xs, id_embedding) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다. MAE의 경우 Xt_embed, Xs_embed를 넣으면 될 것 같다 (same latent space)
                # Xt_attrs = G.CUMAE_tgt(Xt) # UNet으로 Xt의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?
                # Xs_attrs = G.CUMAE_src(Xs) # UNet으로 Xs의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?

                Di = D.module(swapped_face)  ##이렇게 나온 swapped face 결과물을 Discriminator에 넣어서 가짜로 구별을 해내는지 확인해 보는 것이다. 0과 가까우면 가짜라고하는것이다.
                
                if args.eye_detector_loss: 
                    Xt_f_eyes, Xt_f_heatmap_left, Xt_f_heatmap_right = detect_landmarks(Xt_f, model_ft)  ##detect_landmarks 부분에 다른 eye loss 뿐만이 아니라 다른 part도 계산하고 싶으면 여기다 코드를 추가해서 넣으면 될거같다. 현재는 눈만 감지하는중.
                    swapped_face_eyes, swapped_face_heatmap_left, swapped_face_heatmap_right = detect_landmarks(swapped_face, model_ft)
                    eye_heatmaps = [Xt_f_heatmap_left, Xt_f_heatmap_right, swapped_face_heatmap_left, swapped_face_heatmap_right]    
                else:
                    eye_heatmaps = None
                
                # landmark extractor
                if args.landmark_detector_loss: ## 눈만 감지해서 loss 계산하는 것 뿐만이 아닌, 다양한 랜드마크를 모두 참고하여 loss를 계산하기 위함. 처음 트레이닝할 때에는 얼굴이 제대로 생성되지 않을 것이기 때문에, 몇 epoch 이후에 이 loss를 활성화 시켜야 한다.
                    Xt_f_pred_heatmap, Xt_f_landmarks = detect_all_landmarks(Xt_f, model_ft)
                    swapped_face_pred_heatmap, swapped_face_landmarks = detect_all_landmarks(swapped_face, model_ft)
                    all_landmark_heatmaps = [Xt_f_pred_heatmap, swapped_face_pred_heatmap]
                    all_landmarks = [Xt_f_landmarks, swapped_face_landmarks]
                else:
                    all_landmark_heatmaps = None
                    all_landmarks = None
                    
                lossG, loss_adv_accumulated, L_adv, L_id, L_attr, L_rec, L_l2_eyes, L_cycle, L_cycle_identity, L_contrastive, L_source_unet, L_target_unet, L_landmarks, L_shape = compute_generator_losses(G, swapped_face, Xt_f, Xs_f, Xt_f_attrs, Di,
                                                                                    eye_heatmaps, loss_adv_accumulated, 
                                                                                    diff_person, same_person, mixed_id_embedding, src_id_emb, tgt_id_emb, swapped_id_emb, recon_f_src, recon_f_tgt, q_fuse, q_r, all_landmark_heatmaps, args)
        
                lossD = compute_discriminator_loss(D, swapped_face, Xs_f, Xt_f, recon_f_src, recon_f_tgt, diff_person, args.device)
                # discriminator training
                opt_D.zero_grad()
                
        else: ##mixed_precision을 쓰지 않는 경우. 현재는 이것으로 트레이닝을 해야한다. amp 실행 시 gradient가 망가짐.
           
            id_ext_src_input, id_ext_tgt_input, Xt_f, Xt_b, Xs_f, Xs_b, same_person = data
            id_ext_src_input = id_ext_src_input.to(args.device)
            id_ext_tgt_input = id_ext_tgt_input.to(args.device)
            
            Xs_f = Xs_f.to(args.device)
            Xt_f = Xt_f.to(args.device)

            same_person = same_person.to(args.device)
            realtime_batch_size = Xt_f.shape[0] 

            mixed_id_embedding, src_id_emb, tgt_id_emb = id_extractor.module.forward(id_ext_src_input, id_ext_tgt_input) ## id_embedding = [B, 769]

            diff_person = torch.ones_like(same_person)
            if args.diff_eq_same:
                same_person = diff_person

            # generator training
            opt_G.zero_grad() ##축적된 gradients를 비워준다

            swapped_face, recon_f_src, recon_f_tgt = G.module.forward(Xt_f, Xs_f, mixed_id_embedding) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다.
            Xt_f_attrs = G.module.CUMAE_tgt(Xt_f) # UNet으로 Xt의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?

            ##swapped_emb = ArcFace value. this is for infoNCE loss mostly
            swapped_id_emb = id_extractor.module.id_forward(swapped_face)

            if args.shape_loss:
                q_fuse, q_r = id_extractor.module.shapeloss_forward(id_ext_src_input, id_ext_tgt_input, swapped_face)  # Y가 network의 output tensor에 denorm까지 되었다고 가정 & q_r은 지금 당장 잡아낼 수가 없으므로(swap 결과가 초반엔 별로여서) 당장은 q_fuse를 똑같이 씀
            else:
                q_fuse, q_r = 0, 0

            # Y, recon_f_src, recon_f_tgt = G(Xt, Xs, id_embedding) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다. MAE의 경우 Xt_embed, Xs_embed를 넣으면 될 것 같다 (same latent space)
            # Xt_attrs = G.CUMAE_tgt(Xt) # UNet으로 Xt의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?
            # Xs_attrs = G.CUMAE_src(Xs) # UNet으로 Xs의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?
            Di = D.module(swapped_face)  ##이렇게 나온  swapped face 결과물을 Discriminator에 넣어서 가짜로 구별을 해내는지 확인해 보는 것이다. 0과 가까우면 가짜라고하는것이다.
            
            if args.eye_detector_loss:
                Xt_f_eyes, Xt_f_heatmap_left, Xt_f_heatmap_right = detect_landmarks(Xt_f, model_ft)  ##detect_landmarks 부문에 다른 eye loss 뿐만이 아니라 다른 part도 계산하고 싶으면 여기다 코드를 추가해서 넣으면 될거같다
                swapped_face_eyes, swapped_face_heatmap_left, swapped_face_heatmap_right = detect_landmarks(swapped_face, model_ft)
                eye_heatmaps = [Xt_f_heatmap_left, Xt_f_heatmap_right, swapped_face_heatmap_left, swapped_face_heatmap_right]    
            else:
                eye_heatmaps = None
            
            # landmark extractor
            if args.landmark_detector_loss:
                Xt_f_pred_heatmap, Xt_f_landmarks = detect_all_landmarks(Xt_f, model_ft)
                swapped_face_pred_heatmap, swapped_face_landmarks = detect_all_landmarks(swapped_face, model_ft)
                all_landmark_heatmaps = [Xt_f_pred_heatmap, swapped_face_pred_heatmap]
                all_landmarks = [Xt_f_landmarks, swapped_face_landmarks]
            else:
                all_landmark_heatmaps = None
                all_landmarks = None

            lossG, loss_adv_accumulated, L_adv, L_id, L_attr, L_rec, L_l2_eyes, L_cycle, L_cycle_identity, L_contrastive, L_source_unet, L_target_unet, L_landmarks, L_shape = compute_generator_losses(G, swapped_face, Xt_f, Xs_f, Xt_f_attrs, Di,
                                                                                    eye_heatmaps, loss_adv_accumulated, 
                                                                                    diff_person, same_person, mixed_id_embedding, src_id_emb, tgt_id_emb, swapped_id_emb, recon_f_src, recon_f_tgt, q_fuse, q_r, all_landmark_heatmaps, args)
            # discriminator training
            opt_D.zero_grad()
            lossD = compute_discriminator_loss(D, swapped_face, Xs_f, Xt_f, recon_f_src, recon_f_tgt, diff_person, args.device)

        # if (iteration + 1) % 100 != 0 and not last_step: # Accumulate gradients for 100 steps
        #     with G.no_sync() and D.no_sync(): # Disable gradient synchronization
        #             loss = loss_fn(model(data), labels) # Forward step
        #             loss.backward() # Backward step + gradient ACCUMULATION
        
        if args.mixed_precision:
        ##for amp implementation (@hojun Seo)        
            scaler.scale(lossG).backward()
            scaler.step(opt_G)
            if args.scheduler:
                scheduler_G.step()
                
        else:
            lossG.backward()
            opt_G.step()
            if args.scheduler:
                scheduler_G.step()
                
        if args.mixed_precision:
        ##for amp implementation (@hojun Seo)
            scaler.scale(lossD).backward()
            if (not args.discr_force) or (loss_adv_accumulated < 4.):
                scaler.step(opt_D)
            if args.scheduler:
                ##https://aimaster.tistory.com/83
                scheduler_D.step()
            
        else:
            lossD.backward() 
            if (not args.discr_force) or (loss_adv_accumulated < 4.):
                opt_D.step()
            if args.scheduler:
                ##https://aimaster.tistory.com/83
                scheduler_D.step()
            
        if args.mixed_precision:
            scaler.update() ##even tho we have 2 loss backwards, update should only be done once
        else:
            pass
            
        '''
        Here onwards, we must (maybe) convert amp mixed precision tensors in autocast region manually if we want to use them in float32 format
        '''
        
        batch_time = time.time() - start_time
        
        if iteration % args.show_step == 0:
            images = [Xs_f, Xt_f, swapped_face]
            if args.eye_detector_loss:
                Xt_f_eyes_img = paint_eyes(Xt_f, Xt_f_eyes)
                # print(f'eyes: ', {Xt_f_eyes.shape})
                # break
                Yt_f_eyes_img = paint_eyes(swapped_face, swapped_face_eyes)
                images.extend([Xt_f_eyes_img, Yt_f_eyes_img])
            image = make_image_list(images)
            if (args.use_wandb) and config['global_rank'] == 0:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/generated_image.jpg', image[:,:,::-1])

            if (args.use_wandb) and (args.use_reconsimg)  and config['global_rank'] == 0:
                images = [Xs_f, recon_f_src, Xt_f, recon_f_tgt]
                image = make_image_list(images)
                wandb.log({"OrgSrc_ReconSrc_OrgTgt_ReconTgt":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})   
        
        if iteration % 10 == 0:
            print(f'GPU {config["local_rank"]} epoch: {epoch}   current iteration: {iteration} / max iteration size: {len(train_dataloader)}')
            print(f'GPU {config["local_rank"]} lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'GPU {config["local_rank"]} L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()} \n')
            if args.eye_detector_loss:
                print(f'GPU {config["local_rank"]} L_l2_eyes: {L_l2_eyes.item()} \n')
            if args.landmark_detector_loss:
                print(f'GPU {config["local_rank"]} L_landmarks: {L_landmarks.item()} \n')
            if args.cycle_loss:
                print(f'GPU {config["local_rank"]} L_cycle: {L_cycle.item()} \n')
            # if args.cycle_identity_loss:
                print(f'GPU {config["local_rank"]} L_cycle_identity: {L_cycle_identity.item()} \n')
            if args.contrastive_loss:
                print(f'GPU {config["local_rank"]} L_contrastive: {L_contrastive.item()} \n')
            if args.unet_loss:
                print(f'GPU {config["local_rank"]} L_source_unet: {L_source_unet.item()} \n')    
                print(f'GPU {config["local_rank"]} L_target_unet: {L_target_unet.item()} \n')
            if args.shape_loss:
                print(f'GPU {config["local_rank"]} L_shape: {L_shape.item()} \n')
                
            print(f'GPU {config["local_rank"]} loss_adv_accumulated: {loss_adv_accumulated} \n')
            if args.scheduler:
                print(f'GPU {config["local_rank"]} scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()} \n')

        if args.use_wandb and config['global_rank'] == 0:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            if args.landmark_detector_loss:
                wandb.log({"loss_landmarks": L_landmarks.item()}, commit=False)
            if args.cycle_loss:
                wandb.log({"loss_cycle": L_cycle.item()}, commit=False)
            # if args.cycle_identity_loss:
                wandb.log({"loss_cycle_identity": L_cycle_identity.item()}, commit=False)
            if args.contrastive_loss:
                wandb.log({"loss_contrastive": L_contrastive.item()}, commit=False)
            if args.unet_loss:
                wandb.log({"loss_source_unet": L_source_unet.item()}, commit=False) 
                wandb.log({"loss_target_unet": L_target_unet.item()}, commit=False)
            if args.shape_loss:
                wandb.log({"loss_shape": L_shape.item()}, commit=False)

            # 설정 필요하면 args에 true false 추가
            # wandb.log({"loss_source_unet": L_source_unet.item()}, commit=False)
            # wandb.log({"loss_target_unet": L_target_unet.item()}, commit=False)
            # wandb.log({"loss_shape": L_shape.item()}, commit=False)
                
            wandb.log({
                    "loss_id": L_id.item(),
                    "lossD": lossD.item(),
                    "lossG": lossG.item(),
                    "loss_adv": L_adv.item(),
                    "loss_attr": L_attr.item(),
                    "loss_rec": L_rec.item(),
                    #    "loss_cycle": L_cycle.item(),
                    #    "loss_cycle_identity": L_cycle_identity.item(),
                    #    "loss_contrastive": L_contrastive.item(),
                    # "loss_source_unet": L_source_unet.item(),
                    # "loss_target_unet": L_target_unet.item(),                       
                    #    "loss_landmarks": L_landmarks.item()
                    })
        
        #if iteration % 10000 == 0:
            
        #if epoch % args.max_epoch == 0: # From this
        if iteration == 0 & epoch % args.save_epoch == 0: # To this

            
            if config['global_rank'] == 0: ##Multi-GPUs를 사용하고 있기 때문에 GPU 0에서만 모델을 트레이닝한 후 checkpoint를 저장하기 위한 설정.
                    
                # torch.save(G.module.state_dict(), f'./saved_models_{args.run_name}/G_latest.pth')
                # torch.save(D.module.state_dict(), f'./saved_models_{args.run_name}/D_latest.pth')
                # torch.save(G.module.state_dict(), f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
                # torch.save(D.module.state_dict(), f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
                
                torch.save({
                    'epoch': epoch, ## 몇번째 epoch까지 트레이닝 하고 있었는지,
                    'iteration': iteration, ## 몇번째 iteration까지 트레이닝 하고 있었는지
                    'batch_size': args.batch_size, ##트레이닝하면서 사용한 batch size 기록
                    'model_state_dict': G.module.state_dict(), ##모델 parameter tensors
                    'optimizer_state_dict': opt_G.state_dict(),##옵티마이저 parameter tensors
                    'wandb_project': args.wandb_project,  ##WandB를 사용한다면 기록 
                    'wandb_entity': args.wandb_entity  ##WandB를 사용한다면 기록
                }, f'./saved_models_{args.run_name}/G_latest.pth')  ##저장할 모델 명
                print('Generator model checkpoint saved')

                torch.save({
                    'epoch': epoch, 
                    'iteration': iteration, 
                    'batch_size': args.batch_size,
                    'model_state_dict': D.module.state_dict(),
                    'optimizer_state_dict': opt_D.state_dict(),
                    'wandb_project': args.wandb_project,
                    'wandb_entity': args.wandb_entity
                }, f'./saved_models_{args.run_name}/D_latest.pth')   
                print('Discriminator model checkpoint saved')

                torch.save({
                    'epoch': epoch,
                    'iteration': iteration,
                    'batch_size': args.batch_size,
                    'model_state_dict': G.module.state_dict(),
                    'optimizer_state_dict': opt_G.state_dict(),
                    'wandb_project': args.wandb_project,
                    'wandb_entity': args.wandb_entity
                }, f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

                torch.save({
                    'epoch': epoch,
                    'iteration': iteration,
                    'batch_size': args.batch_size,
                    'model_state_dict': D.module.state_dict(),
                    'optimizer_state_dict': opt_D.state_dict(),
                    'wandb_project': args.wandb_project,
                    'wandb_entity': args.wandb_entity
                }, f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')                

              
        #if (iteration % 100 == 0) and (args.use_wandb) and config['global_rank'] == 0:
        # if (iteration % 1000 == 0) and (args.use_wandb) and config['global_rank'] == 0: 
        #     G.eval()
        #     res1 = get_faceswap('examples/images/training/source1.png', 'examples/images/training/target1.png', G, id_extractor, device)
        #     res2 = get_faceswap('examples/images/training/source2.png', 'examples/images/training/target2.png', G, id_extractor, device)  
        #     res3 = get_faceswap('examples/images/training/source3.png', 'examples/images/training/target3.png', G, id_extractor, device)
        #     res4 = get_faceswap('examples/images/training/source4.png', 'examples/images/training/target4.png', G, id_extractor, device)
        #     res5 = get_faceswap('examples/images/training/source5.png', 'examples/images/training/target5.png', G, id_extractor, device)  
        #     res6 = get_faceswap('examples/images/training/source6.png', 'examples/images/training/target6.png', G, id_extractor, device)
        #     output1 = np.concatenate((res1, res2, res3), axis=0)
        #     output2 = np.concatenate((res4, res5, res6), axis=0)
        #     output = np.concatenate((output1, output2), axis=1)
        #     wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
        #     G.train()


def train(args, config):
    
    ##Multi GPU setting
    assert torch.cuda.is_available(), "Training on CPU is not supported as Multi-GPU strategy is set"
    device = args.device
    print(f"[GPU {config['local_rank']}] is using device: {args.device}")
    print(f"[GPU {config['local_rank']}] is loading dataset")

    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch    
    
    ## initializing id extractor model
    f_3d_path = "/datasets/pretrained/pretrained_model.pth"
    f_id_path = "/datasets/pretrained/backbone.pth"
    id_extractor = ShapeAwareIdentityExtractor(f_3d_path, f_id_path, args.mixed_precision, args.id_mode).to(args.device)
    id_extractor = DistributedDataParallel(id_extractor, device_ids=[config['local_rank']])
    id_extractor.eval()

    # initializing main models
    G = CrossUnetAttentionGenerator(backbone='unet', num_adain = args.num_adain).to(args.device)
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    G = DistributedDataParallel(G, device_ids=[config['local_rank']])
    
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(args.device)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)
    D = DistributedDataParallel(D, device_ids=[config['local_rank']])
    
    # initializing model for identity extraction
    if args.mixed_precision == True:  
        netArc = iresnet100(fp16=True)
    else:
        netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('/datasets/pretrained/backbone.pth'))
    netArc = netArc.to(args.device)
    # netArc=netArc.cuda()
    netArc = DistributedDataParallel(netArc, device_ids=[config['local_rank']])
    netArc.eval()
    
    if args.eye_detector_loss:
        model_ft = models.FAN(4, "False", "False", 98)
        # checkpoint = torch.load('./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth')
        checkpoint = torch.load('/datasets/pretrained/WFLW_4HG.pth')
        
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
        model_ft = model_ft.to(args.device)
        model_ft = DistributedDataParallel(model_ft, device_ids=[config['local_rank']])
        model_ft.eval()
    else:
        model_ft=None
    
    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        scheduler_D = None
        
    starting_epoch = 0
    if args.pretrained:
        try:
            # G.module.load_state_dict(torch.load(args.G_path, map_location=torch.device(config['local_rank'])), strict=False)
            # D.module.load_state_dict(torch.load(args.D_path, map_location=torch.device(config['local_rank'])), strict=False)
            # G.module.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            # D.module.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            
            G_state = torch.load(f'./saved_models_{args.run_name}/G_latest.pth')
            D_state = torch.load(f'./saved_models_{args.run_name}/D_latest.pth')
            
            G.load_state_dict(G_state['model_state_dict'])
            starting_epoch = G_state['epoch'] + 1
            starting_iteration = G_state['iteration'] + 1
            opt_G.load_state_dict(G_state['optimizer_state_dict'])
            print(f'GPU {config["local_rank"]} - Preloading model ./saved_models_{args.run_name}/G_latest.pt')
            
            D.load_state_dict(D_state['model_state_dict'])
            starting_epoch = D_state['epoch'] + 1
            starting_iteration = D_state['iteration'] + 1
            opt_D.load_state_dict(D_state['optimizer_state_dict'])
            print(f'GPU {config["local_rank"]} - Preloading model ./saved_models_{args.run_name}/D_latest.pt')
                        
            print(f'[GPU {config["local_rank"]}]: Loaded pretrained weights for G and D')
        except FileNotFoundError as e:
            print(f'[GPU {config["local_rank"]}]: Not found pretrained weights. Continue without any pretrained weights.')
    else:
        starting_iteration = 0
    
    ## 커스텀 데이터셋 불러오기
    dataset = FaceEmbedCombined(ffhq_data_path = args.ffhq_data_path, same_prob=0.8, same_identity=args.same_identity)
    ##데이터셋을 train/validation세트로 분할
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_ratio)
    validation_size = int(dataset_size - train_size)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    ## Multi-GPU 셋팅 안에서 각 GPU 안에 train/validation 데이터 사이즈 체크하기
    print(f'[GPU {config["local_rank"]}]: Training Data Size : {len(train_dataset)}')
    print(f'[GPU {config["local_rank"]}]: Validation Data Size : {len(validation_dataset)}')
    
    ##나누어진 데이터를 각각의 데이터로더에 넣는다. Multi GPU를 사용하기 때문에, shuffle을 False로 해주어야하고, 대신 sampler에서 DistributedSampler를 설정해 준 후 거기에 shuffle을 True로 넣는다.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=DistributedSampler(train_dataset, shuffle=True))
    ##validation data는 모든 gpu에서 계산하는 것이 아니기 때문에 DistributedSampler를 설정하지 않아도 된다.
    valid_dataloader = DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=True, drop_last=True)

    # Будем считать аккумулированный adv loss, чтобы обучать дискриминатор только когда он ниже порога, если discr_force=True
    loss_adv_accumulated = 20.
    
    for epoch in range(starting_epoch, max_epoch):
        torch.cuda.empty_cache()
        G.train()
        D.train()
        ##training 시작
        train_one_epoch(G,
                        D,
                        id_extractor,
                        opt_G,
                        opt_D,
                        scheduler_G,
                        scheduler_D,
                        netArc,
                        model_ft,
                        args,
                        train_dataloader,
                        device,
                        epoch,
                        starting_iteration,
                        loss_adv_accumulated,
                        config)
        
    ##Multi-GPU 환경에서 global rank가 0인 단 하나의 지피유에서만 validation data를 사용하여 4가지 metrics를 계산한다.
    if config['global_rank'] == 0:        
        
        ##This below is validation part
        # running_vloss = 0.0
        running_pose_metric = 0.0
        running_id_metric = 0.0
        running_fid_metric = 0.0
        running_expression_metric = 0.0

        G.eval()
        D.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, valid_minibatch in enumerate(valid_dataloader):
                val_id_ext_src_input, val_id_ext_tgt_input, val_Xt_f, val_Xt_b, val_Xs_f, val_Xs_b, val_same_person = valid_minibatch
                val_id_ext_src_input = val_id_ext_src_input.to(args.device)
                val_id_ext_tgt_input = val_id_ext_tgt_input.to(args.device)
                val_Xs_f = val_Xs_f.to(args.device)
                val_Xt_f = val_Xt_f.to(args.device)
                val_same_person = val_same_person.to(args.device)
                val_realtime_batch_size = val_Xt_f.shape[0] 

                ##id_embedding = arcface + shapeaware embedding, [src_emb, tgt_emb] = arcface embedding
                val_id_embedding, val_src_id_emb, val_tgt_id_emb = id_extractor.module.forward(val_id_ext_src_input, val_id_ext_tgt_input)
                val_id_embedding, val_src_id_emb, val_tgt_id_emb = val_id_embedding.to(args.device), val_src_id_emb.to(args.device), val_tgt_id_emb.to(args.device)

                val_diff_person = torch.ones_like(val_same_person)

                if args.diff_eq_same:
                    val_same_person = val_diff_person
            
                    
                val_swapped_face, val_recon_f_src, val_recon_f_tgt = G.module.forward(val_Xt_f, val_Xs_f, val_id_embedding)
                
                pose_value, fid_value, id_value, expression_value = 0, 0, 0, 0
                metrics_processors = []
                if args.metrics_pose:
                    metrics_processors.append("POSE")
                if args.metrics_fid:
                    metrics_processors.append("FID")
                if args.metrics_id:
                    metrics_processors.append("ID")
                if args.metrics_expression:
                    metrics_processors.append("EXPRESSION")

                result = metric.run(val_Xs_f, val_Xt_f, val_swapped_face, metrics_processors)
                print("metrics result : ", result)
                if args.metrics_pose:
                    pose_value = result["metrics.POSE"]
                if args.metrics_fid:
                    fid_value = result["metrics.FID"]
                if args.metrics_id:
                    id_value = result["metrics.ID"]
                if args.metrics_expression:
                    expression_value = result["metrics.EXPRESSION"]
                
                running_pose_metric += pose_value
                running_id_metric += id_value
                running_fid_metric += fid_value
                running_expression_metric += expression_value

                if args.use_wandb and config['global_rank'] == 0:
                    wandb.log({"running_pose_metric": running_pose_metric, "running_id_metric": running_id_metric, "running_fid_metric": running_fid_metric}, commit=False)
                    if args.landmark_detector_loss:
                        wandb.log({"running_expression_metric": running_expression_metric}, commit=False)
                        
                        
            if config['global_rank'] == 0: 
                
                avg_pose_metric = running_pose_metric / (i + 1)
                avg_id_metric = running_id_metric / (i + 1)
                avg_fid_metric = running_fid_metric / (i + 1)
                avg_expression_metric = running_expression_metric / (i + 1)
                
                
                wandb.log({"avg_pose_metric": avg_pose_metric, "avg_id_metric": avg_id_metric, "avg_fid_metric": avg_fid_metric}, commit=False)
                if args.landmark_detector_loss:
                    wandb.log({"avg_expression_metric": avg_expression_metric}, commit=False)
                
                ## adding functions for WandB to log the metrics
                ## put up the generated validation images in WandB
                ## saving functions for best G and D
                # torch.save(model.state_dict(), model_path)
            

def main(args):

    ##Multi-GPU를 Pytorch에서 구성하기 위한 파트
    config = dict()
    config['local_rank'] = int(os.environ['LOCAL_RANK'])
    config['global_rank'] = int(os.environ['RANK'])

    assert config['local_rank'] != -1, "LOCAL_RANK environment variable not set"
    assert config['global_rank'] != -1, "RANK environment variable not set"
    
    # Print configuration (only once per server)
    if config['local_rank'] == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:>20}: {value}")  
    ##WandB는 모든 GPU를 참고할 필요없고 global rank 0 GPU 하나에서만 확인하면 되기 때문에 아래와 같이 체크를 함.
    if args.use_wandb==True and config['global_rank'] == 0:
        wandb.init(project=args.wandb_project, 
                   entity=args.wandb_entity, 
                   settings=wandb.Settings(start_method='fork'),
                #    id=args.wandb_id,
                   resume='allow')
        configs = wandb.config
        configs.ffhq_data_path = args.ffhq_data_path
        configs.train_ratio = args.train_ratio
        configs.G_path = args.G_path
        configs.D_path = args.D_path

        configs.weight_adv = args.weight_adv
        configs.weight_attr = args.weight_attr
        configs.weight_id = args.weight_id
        configs.weight_rec = args.weight_rec
        configs.weight_eyes = args.weight_eyes
        configs.weight_cycle = args.weight_cycle
        configs.weight_cycle_identity = args.weight_cycle_identity
        configs.weight_contrastive = args.weight_contrastive
        configs.weight_source_unet = args.weight_source_unet
        configs.weight_target_unet = args.weight_target_unet
        configs.weight_landmarks = args.weight_landmarks
        configs.backbone = args.backbone
        configs.num_blocks = args.num_blocks
        configs.num_adain = args.num_adain
        configs.id_mode = args.id_mode
        configs.seq_len = args.seq_len
        configs.n_head = args.n_head
        configs.total_embed_dim = args.total_embed_dim
        configs.q_dim = args.q_dim
        configs.k_dim = args.k_dim
        configs.kv_dim = args.kv_dim
        configs.same_person = args.same_person
        configs.same_identity = args.same_identity
        configs.diff_eq_same = args.diff_eq_same
        configs.discr_force = args.discr_force
        configs.scheduler = args.scheduler
        configs.scheduler_step = args.scheduler_step
        configs.scheduler_gamma = args.scheduler_gamma
        configs.eye_detector_loss = args.eye_detector_loss
        configs.landmark_detector_loss = args.landmark_detector_loss
        configs.cycle_loss = args.cycle_loss
        configs.contrastive_loss = args.contrastive_loss
        configs.shape_loss = args.shape_loss
        configs.wandb_id = args.wandb_id
        configs.run_name = args.run_name
        configs.wandb_project = args.wandb_project
        
        configs.batch_size = args.batch_size
        configs.val_batch_size = args.val_batch_size
        
        configs.lr_G = args.lr_G
        configs.lr_D = args.lr_D
        configs.max_epoch = args.max_epoch
        configs.show_step = args.show_step
        configs.save_epoch = args.save_epoch
        configs.mixed_precision = args.mixed_precision
        configs.device = args.device

        configs.pretrained = args.pretrained

        
    elif not os.path.exists('./images'):
        os.mkdir('./images')

    ##Multi-GPU를 Pytorch에서 구성하기 위한 파트     
    # Setup distributed training
    print(f'[GPU {config["local_rank"]}]: Setting up distributed training..')
    init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    print(f'[GPU {config["local_rank"]}]: initiating distributed process with nccl')
    torch.cuda.set_device(config['local_rank'])  
    print(f'[GPU {config["local_rank"]}]: setting device with cuda in local rank')
    print(f'[GPU {config["local_rank"]}]: Starting training')\
    # train(args, device=device)
    train(args, config)
    
    # Clean up distributed training
    destroy_process_group()
    print(f'[GPU {config["local_rank"]}]: destroyed distributed process after training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset params
    ##the 4 arguments are newly added by Hojun
    # parser.add_argument('--vgg_data_path', default='/datasets/VGG', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    # parser.add_argument('--ffhq_data_path', default='/datasets/FFHQ', type=str,help='path to ffhq dataset in string format')
    parser.add_argument('--ffhq_data_path', default='/datasets/FFHQ_parsed_img', type=str,help='path to ffhq dataset in string format')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='how much data to be used as training set. The rest will be used as validation set. e.g.) if 0.9, validation data will be 0.1 (10%)')
    
    # parser.add_argument('--celeba_data_path', default='/datasets/CelebHQ/CelebA-HQ-img', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    # parser.add_argument('--dob_data_path', default='/datasets/DOB', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    
    parser.add_argument('--G_path', default='./saved_models/G.pth', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--D_path', default='./saved_models/D.pth', help='Path to pretrained weights for D. Only used if pretrained=True')

    # weights for loss
    parser.add_argument('--weight_adv', default=2, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=15, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=10, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=1.5, type=float, help='Eyes Loss weight')
    parser.add_argument('--weight_cycle', default=1., type=float, help='Cycle Loss weight for generator')
    parser.add_argument('--weight_cycle_identity', default=1., type=float, help='Cycle Identity Loss weight for generator')
    parser.add_argument('--weight_contrastive', default=1.5, type=float, help='Contrastive Loss weight for idendity embedding of generator')
    parser.add_argument('--weight_source_unet', default=0.5, type=float, help='Source Image Unet Reconstruction Loss weight for generator')
    parser.add_argument('--weight_target_unet', default=0.5, type=float, help='Target Image Unet Reconstruction Loss weight for generator')
    parser.add_argument('--weight_landmarks', default=2., type=float, help='Landmark Loss weight for generator')
    parser.add_argument('--weight_shape', default=3., type=float, help='Shape Loss weight for generator')
    

    # training params you may want to change
    ##parameters for model configs
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder. The other modes are not applicable')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    parser.add_argument('--num_adain', default=6, type=int, help='Numbers of AdaIN_ResBlocks') # 1부터 6까지. AdaIN_Resblock을 시작점으로부터 N개 사용한다는 의미
    parser.add_argument('--id_mode', default='arcface', type=str, help='Mode change is possible between 1) arcface 2) hififace') # 1부터 6까지. AdaIN_Resblock을 시작점으로부터 N개 사용한다는 의미
    
    parser.add_argument('--seq_len', default=196, type=int, help='sequence length = height*width, number of patches of ViT. It would normally be H*W = 196 or 256')
    parser.add_argument('--n_head', default=2, type=int, help='number of multi attention head')
    parser.add_argument('--total_embed_dim', default=512, type=int, help="Full query dim (and query's value dimension) before dividing by num head ")
    parser.add_argument('--q_dim', default=1024, type=int, help="Full query dim (and query's value dimension) before dividing by num head ")
    parser.add_argument('--k_dim', default=1024, type=int, help="Full key dim (and/or key's value dimension) before dividing by num head ")
    parser.add_argument('--kv_dim', default=1024, type=int, help='value dim of key before dividing by num head. Key value dimension doesnt neccessarily have to be same as key dim')
    
    parser.add_argument('--same_person', default=0.2, type=float, help='Probability of using same person identity during training')
    parser.add_argument('--same_identity', default=True, type=bool, help='Using simswap approach, when source_id = target_id. Only possible with vgg=True')
    parser.add_argument('--diff_eq_same', default=False, type=bool, help='Don\'t use info about where is defferent identities')
    parser.add_argument('--pretrained', default=False, type=bool, help='If using the pretrained weights for training or not')
    parser.add_argument('--discr_force', default=False, type=bool, help='If True Discriminator would not train when adversarial loss is high')
    parser.add_argument('--scheduler', default=False, type=bool, help='If True decreasing LR is used for learning of generator and discriminator')
    parser.add_argument('--scheduler_step', default=5000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    parser.add_argument('--eye_detector_loss', default=False, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    parser.add_argument('--landmark_detector_loss', default=False, type=bool, help='If True landmark loss is applied to generator')
    parser.add_argument('--cycle_loss', default=False, type=bool, help='If True, cycle & cycle identity losses are applied to generator')
    parser.add_argument('--contrastive_loss', default=True, type=bool, help='If True, contrastive loss is applied to generator')
    parser.add_argument('--unet_loss', default=True, type=bool, help='If True, unet losses for source and target are applied to generator')
    parser.add_argument('--shape_loss', default=False, type=bool, help='If True, contrastive loss is applied to generator')
    
    
    # info about this run
    parser.add_argument('--use_wandb', default=True, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--wandb_id', default='123456', type=bool, help='unique IDs for wandb run')
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str, help='name of project. for example, faceswap_basemodel')
    parser.add_argument('--wandb_entity', default='your-login', type=str, help='name of team in wandb. ours is dob_faceswapteam')
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--val_batch_size', default=8, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=20, type=int)
    parser.add_argument('--save_epoch', default=2, type=int)
    parser.add_argument('--mixed_precision', default=False, type=bool)
    parser.add_argument('--device', default='cuda', type=str, help='setting device between cuda and cpu')
    parser.add_argument('--use_reconsimg', default=True, type=bool, help='create reconsimg for wandb')

    # metrics info
    parser.add_argument('--metrics_expression', default=False, type=bool)
    parser.add_argument('--metrics_fid', default=True, type=bool)
    parser.add_argument('--metrics_id', default=True, type=bool)
    parser.add_argument('--metrics_pose', default=True, type=bool)
    args = parser.parse_args()

    # Создаем папки, чтобы было куда сохранять последние веса моделей, а также веса с каждой эпохи
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
        os.mkdir(f'./current_models_{args.run_name}')
    
    main(args)