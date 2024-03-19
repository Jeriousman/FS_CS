import sys
import argparse
import time
import cv2
import wandb
from PIL import Image
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

# # custom imports
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
import torch






def train_one_epoch(G: 'generator model', 
                    D: 'discriminator model', 
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
                    loss_adv_accumulated:int,
                    config:dict
                    ):
    

    
    ##loading pretrained models for extracting IDs
    f_3d_path = "/datasets/pretrained/pretrained_model.pth"
    f_id_path = "/datasets/pretrained/backbone.pth"
    
    id_extractor = ShapeAwareIdentityExtractor(f_3d_path, f_id_path, args.id_mode)
    id_extractor = DistributedDataParallel(id_extractor, device_ids=[config['local_rank']])
    #print(id_extractor)
    
    # Xs.shape
    for iteration, data in enumerate(train_dataloader):
        start_time = time.time()
        
        id_ext_src_input, id_ext_tgt_input, Xt_f, Xt_b, Xs_f, Xs_b, same_person = data


        Xs_f = Xs_f.to(args.device)
        # Xs.shape
        Xt_f = Xt_f.to(args.device)
        # Xt.shape
        same_person = same_person.to(args.device)
        realtime_batch_size = Xt_f.shape[0] 
        # print("realtime_batchsize : ", data)
        # break

        ##id_embedding = arcface + shapeaware embedding, [src_emb, tgt_emb] = arcface embedding
        id_embedding, src_id_emb, tgt_id_emb = id_extractor(id_ext_src_input, id_ext_tgt_input)
        id_embedding, src_id_emb, tgt_id_emb = id_embedding.to(args.device), src_id_emb.to(args.device), tgt_id_emb.to(args.device)



        diff_person = torch.ones_like(same_person)

        if args.diff_eq_same:
            same_person = diff_person


        # generator training
        opt_G.zero_grad() ##축적된 gradients를 비워준다



        swapped_face, recon_f_src, recon_f_tgt = G.module(Xt_f, Xs_f, id_embedding) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다. MAE의 경우 Xt_embed, Xs_embed를 넣으면 될 것 같다 (same latent space)
        Xt_f_attrs = G.module.CUMAE_tgt(Xt_f) # UNet으로 Xt의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?
        Xs_f_attrs = G.module.CUMAE_src(Xs_f) # UNet으로 Xs의 bottleneck 이후 feature maps -> 238번 line을 통해 forward가 돌아갈 때 한 번에 계산해놓을 수 있을듯?


        ##swapped_emb = ArcFace value. this is for infoNCE loss mostly
        swapped_id_emb = id_extractor.module.id_forward(swapped_face)
        swapped_id_emb = swapped_id_emb.to(args.device)

        #q_fuse, q_r = id_extractor.shapeloss_forward(id_ext_src_input, id_ext_tgt_input, swapped_face)  # Y가 network의 output tensor에 denorm까지 되었다고 가정 & q_r은 지금 당장 잡아낼 수가 없으므로(swap 결과가 초반엔 별로여서) 당장은 q_fuse를 똑같이 씀



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


        
        lossG, loss_adv_accumulated, L_adv, L_id, L_attr, L_rec, L_l2_eyes, L_cycle, L_cycle_identity, L_contrastive, L_source_unet, L_target_unet, L_landmarks = compute_generator_losses(G, swapped_face, Xt_f, Xs_f, Xt_f_attrs, Di,
                                                                             eye_heatmaps, loss_adv_accumulated, 
                                                                             diff_person, same_person, src_id_emb, tgt_id_emb, swapped_id_emb, recon_f_src, recon_f_tgt, all_landmark_heatmaps, config)

        # with amp.scale_loss(lossG, opt_G) as scaled_loss:
        #     scaled_loss.backward()
        # lossG.backward(retain_graph=True)
        # torch.autograd.set_detect_anomaly(True)
        # lossG.backward(retain_graph=True)
        lossG.backward()

        opt_G.step()
        if args.scheduler:
            scheduler_G.step()
        
        # discriminator training
        opt_D.zero_grad()
        # lossD = compute_discriminator_loss(D, Y, Xs, diff_person)

        lossD = compute_discriminator_loss(D, swapped_face, Xs_f, Xt_f, recon_f_src, recon_f_tgt, diff_person, args.device)
        
        # with amp.scale_loss(lossD, opt_D) as scaled_loss:
        #     scaled_loss.backward()
        lossD.backward() 

        if (not args.discr_force) or (loss_adv_accumulated < 4.):
            opt_D.step()
        if args.scheduler:
            scheduler_D.step()
        
        
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs_f, Xt_f, swapped_face]
            if args.eye_detector_loss:
                Xt_f_eyes_img = paint_eyes(Xt_f, Xt_f_eyes)
                Yt_f_eyes_img = paint_eyes(swapped_face, swapped_face_eyes)
                images.extend([Xt_f_eyes_img, Yt_f_eyes_img])
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/generated_image.jpg', image[:,:,::-1])
        
        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(train_dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if args.eye_detector_loss:
                print(f'L_l2_eyes: {L_l2_eyes.item()}')
            if args.landmark_detector_loss:
                print(f'L_landmarks: {L_landmarks.item()}')
            if args.cycle_loss:
                pass
            if args.contrastive_loss:
                pass
            if args.shape_loss:
                pass
                
            print(f'loss_adv_accumulated: {loss_adv_accumulated}')
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}')

        if args.use_wandb:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            if args.landmark_detector_loss:
                wandb.log({"loss_landmarks": L_landmarks.item()}, commit=False)
            if args.cycle_loss:
                wandb.log({"loss_cycle": L_cycle.item()}, commit=False)
            if args.cycle_loss:
                wandb.log({"loss_cycle_identity": L_cycle_identity.item()}, commit=False)
            if args.contrastive_loss:
                wandb.log({"loss_contrastive": L_contrastive.item()}, commit=False)
            if args.shape_loss:
                pass

            # 설정 필요하면 args에 true false 추가
            wandb.log({"loss_source_unet": L_source_unet.item()}, commit=False)
            wandb.log({"loss_target_unet": L_target_unet.item()}, commit=False)
            
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
                       "loss_source_unet": L_source_unet.item(),
                       "loss_target_unet": L_target_unet.item(),                       
                    #    "loss_landmarks": L_landmarks.item()
                       })
        
        if iteration % 10000 == 0:
            
            if config['global_rank'] == 0:
                    
                torch.save(G.module.state_dict(), f'./saved_models_{args.run_name}/G_latest.pth')
                torch.save(D.module.state_dict(), f'./saved_models_{args.run_name}/D_latest.pth')

                torch.save(G.module.state_dict(), f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
                torch.save(D.module.state_dict(), f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        if (iteration % 100 == 0) and (args.use_wandb) and config['global_rank'] == 0:

            G.eval()

            res1 = get_faceswap('examples/images/training/source1.png', 'examples/images/training/target1.png', G, id_extractor, device)
            res2 = get_faceswap('examples/images/training/source2.png', 'examples/images/training/target2.png', G, id_extractor, device)  
            res3 = get_faceswap('examples/images/training/source3.png', 'examples/images/training/target3.png', G, id_extractor, device)
            res4 = get_faceswap('examples/images/training/source4.png', 'examples/images/training/target4.png', G, id_extractor, device)
            res5 = get_faceswap('examples/images/training/source5.png', 'examples/images/training/target5.png', G, id_extractor, device)  
            res6 = get_faceswap('examples/images/training/source6.png', 'examples/images/training/target6.png', G, id_extractor, device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})

            G.train()

# def train(args, config):
def train(config):
    
    ##Multi GPU setting
    assert torch.cuda.is_available(), "Training on CPU is not supported as Multi-GPU strategy is set"
    device = args.device
    print(f"GPU {config['local_rank']} is using device: {args.device}")
    print(f"GPU {config['local_rank']} is loading dataset")

    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch    
    
    
    # # training params
    # batch_size = config['batch_size
    # max_epoch = config['max_epoch
    
    # initializing main models
    # G = AEI_Net(config['backbone, num_blocks=config['num_blocks, c_id=512).to(device)
    G = CrossUnetAttentionGenerator(backbone='unet', num_adain = args.num_adain).to(args.device)
    G = DistributedDataParallel(G, device_ids=[config['local_rank']])
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(args.device)    
    D = DistributedDataParallel(D, device_ids=[config['local_rank']])
    

    
    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.module.load_state_dict(torch.load('/datasets/pretrained/backbone.pth'))
    netArc=netArc.cuda()
    netArc = DistributedDataParallel(netArc, device_ids=[config['local_rank']])
    netArc.eval()

    
    
    if args.eye_detector_loss:
        model_ft = models.FAN(4, "False", "False", 98)
        # checkpoint = torch.load('./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth')
        checkpoint = torch.load('/datasets/pretrained/WFLW_4HG.pth')
        
        if 'state_dict' not in checkpoint:
            model_ft.module.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.module.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.module.load_state_dict(model_weights)
        model_ft = model_ft.to(args.device)
        model_ft = DistributedDataParallel(model_ft, device_ids=[config['local_rank']])
        model_ft.eval()
    else:
        model_ft=None
    
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)

    # G, opt_G = amp.initialize(G, opt_G, opt_level=config['optim_level)
    # D, opt_D = amp.initialize(D, opt_D, opt_level=config['optim_level)
    
    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        scheduler_D = None
        
    if args.pretrained:
        try:
            G.module.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            D.module.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G and D")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    # if config['vgg:
    
    dataset = FaceEmbedCombined(ffhq_data_path = args.ffhq_data_path, same_prob=0.8, same_identity=args.same_identity)
    # dataset = FaceEmbedCombined(ffhq_data_path=config['ffhq_data_path, same_prob=0.8, same_identity=config['same_identity)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    # dataset = FaceEmbedCustom('/workspace/examples/images/training')
    

    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_ratio)
    validation_size = int(dataset_size - train_size)
    
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")
    
    # dataloader = DataLoader(dataset, batch_size=config['batch_size, shuffle=True, drop_last=True)

    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=DistributedSampler(train_dataset, shuffle=True))
    # valid_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size, shuffle=False, drop_last=True)
    valid_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=DistributedSampler(validation_dataset, shuffle=True))
    # print(next(iter(dataloader)))
    # print(next(iter(dataloader))[0])
    ##In case of multi GPU, turn off shuffle
    # dataloader = DataLoader(dataset, batch_size=config['batch_size, sampler=DistributedSampler(dataset, shuffle=True))


    

    # Будем считать аккумулированный adv loss, чтобы обучать дискриминатор только когда он ниже порога, если discr_force=True
    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        # if epoch >= 1:
        #     config['id_mode = 'Hififace'
            
        G.train()
        D.train()
        torch.cuda.empty_cache()
        train_one_epoch(G,
                        D,
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
                        loss_adv_accumulated,
                        config)
        

        
        
        ##This below is validation part
        running_vloss = 0.0
        running_pose_metric = 0.0
        running_id_metric = 0.0
        running_fid_metric = 0.0
        running_expression_metric = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
        G.eval()
        D.eval()
        
    ##loading pretrained models for extracting IDs
        f_3d_path = "/datasets/pretrained/pretrained_model.pth"
        f_id_path = "/datasets/pretrained/backbone.pth"
        id_extractor = ShapeAwareIdentityExtractor(f_3d_path, f_id_path, args.id_mode)
        id_extractor = DistributedDataParallel(id_extractor, device_ids=[config['local_rank']])
        id_extractor.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, valid_minibatch in enumerate(valid_dataloader):
                val_id_ext_src_input, val_id_ext_tgt_input, val_Xt_f, val_Xt_b, val_Xs_f, val_Xs_b, val_same_person = valid_minibatch

                val_Xs_f = val_Xs_f.to(args.device)
                # Xs.shape
                val_Xt_f = val_Xt_f.to(args.device)
                # Xt.shape
                val_same_person = val_same_person.to(args.device)
                val_realtime_batch_size = val_Xt_f.shape[0] 
                # break

                ##id_embedding = arcface + shapeaware embedding, [src_emb, tgt_emb] = arcface embedding
                val_id_embedding, val_src_id_emb, val_tgt_id_emb = id_extractor(val_id_ext_src_input, val_id_ext_tgt_input)
                val_id_embedding, val_src_id_emb, val_tgt_id_emb = val_id_embedding.to(args.device), val_src_id_emb.to(args.device), val_tgt_id_emb.to(args.device)

                val_diff_person = torch.ones_like(val_same_person)

                if args.diff_eq_same:
                    val_same_person = val_diff_person
                
                
                val_swapped_face, val_recon_f_src, val_recon_f_tgt = G.module(val_Xt_f, val_Xs_f, val_id_embedding)
                
                ## calculate the 4 metrics
                
                ## pose_value = POSE(val_swapped_face, val_Xt)
                ## id_value = ID(val_swapped_face)
                ## fid_value = FID(val_swapped_face)
                ## expression_value = EXPRESSION(val_swapped_face)
                
                
                
                ## probably losses dont need to be calculated
                # voutputs = model(vinputs)
                # vloss = loss_fn(val_swapped_face)
                if config['global_rank'] == 0:
                
                    # running_vloss += vloss
                    running_pose_metric += pose_value
                    running_id_metric += id_value
                    running_fid_metric += fid_value
                    running_expression_metric += expression_value
                        
                    
                    wandb.log({"running_pose_metric": running_pose_metric, "running_id_metric": running_id_metric, "running_fid_metric": running_fid_metric}, commit=False)
                    if args.landmark_detector_loss:
                        wandb.log({"running_expression_metric": running_expression_metric}, commit=False)
                        
                    ## logging and checking validation images generated swapped faces    
                    # if (i % 50 == 0):
                        # wandb.log({"Val Generated Images" : wandb.Image()})
                        # wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
                        
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

    config = dict()
    # config.update(vars(args))
    config['local_rank'] = int(os.environ['LOCAL_RANK'])
    config['global_rank'] = int(os.environ['RANK'])

    assert config['local_rank'] != -1, "LOCAL_RANK environment variable not set"
    assert config['global_rank'] != -1, "RANK environment variable not set"
    
    # Print configuration (only once per server)
    if config['local_rank'] == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:>20}: {value}")  
            
    # Setup distributed training
    init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(config['local_rank'])  
    
    
    print("Starting training")\
    # train(args, device=device)
    train(config)
    
    # Clean up distributed training
    destroy_process_group()
    # config = dict()
    # config['local_rank'] = int(os.environ(['LOCAL_RANK']))
    # config['global_rank'] = int(os.environ(['RANK']))

    # assert config['local_rank'] != -1, "LOCAL_RANK environment variable not set"
    # assert config['global_rank'] != -1, "RANK environment variable not set"


    # # Print configuration (only once per server)
    # if config['local_rank'] == 0:
    #     print("Configuration:")
    #     for key, value in config.items():
    #         print(f"{key:>20}: {value}")
            
    # # Setup distributed training
    # init_process_group(backend='nccl')
    # torch.cuda.set_device(config.local_rank)
    
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # device = 'cpu'
    # if not torch.cuda.is_available():
    #     print('cuda is not available. using cpu. check if it\'s ok')
    
    # print("Starting training")
    # train(args, gpu_config)
    
    # # Clean up distributed training
    # destroy_process_group()


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if not torch.cuda.is_available():
    #     print('cuda is not available. using cpu. check if it\'s ok')
    




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
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=20, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=2., type=float, help='Eyes Loss weight')
    parser.add_argument('--weight_cycle', default=1., type=float, help='Cycle Loss weight for generator')
    parser.add_argument('--weight_cycle_identity', default=1., type=float, help='Cycle Identity Loss weight for generator')
    parser.add_argument('--weight_contrastive', default=1., type=float, help='Contrastive Loss weight for idendity embedding of generator')
    parser.add_argument('--weight_source_unet', default=2., type=float, help='Source Image Unet Reconstruction Loss weight for generator')
    parser.add_argument('--weight_target_unet', default=2., type=float, help='Target Image Unet Reconstruction Loss weight for generator')
    parser.add_argument('--weight_landmarks', default=3., type=float, help='Landmark Loss weight for generator')
    

    
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
    # parser.add_argument('--seq_len', default=196, type=int, help='number of patches of ViT. It would normally be H*W = 196 or 256')
    
    
    parser.add_argument('--same_person', default=0.2, type=float, help='Probability of using same person identity during training')
    parser.add_argument('--same_identity', default=True, type=bool, help='Using simswap approach, when source_id = target_id. Only possible with vgg=True')
    parser.add_argument('--diff_eq_same', default=False, type=bool, help='Don\'t use info about where is defferent identities')
    parser.add_argument('--pretrained', default=True, type=bool, help='If using the pretrained weights for training or not')
    parser.add_argument('--discr_force', default=False, type=bool, help='If True Discriminator would not train when adversarial loss is high')
    parser.add_argument('--scheduler', default=False, type=bool, help='If True decreasing LR is used for learning of generator and discriminator')
    parser.add_argument('--scheduler_step', default=5000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    parser.add_argument('--eye_detector_loss', default=True, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    parser.add_argument('--landmark_detector_loss', default=True, type=bool, help='If True landmark loss is applied to generator')
    parser.add_argument('--cycle_loss', default=True, type=bool, help='If True, cycle & cycle identity losses are applied to generator')
    parser.add_argument('--contrastive_loss', default=True, type=bool, help='If True, contrastive loss is applied to generator')
    parser.add_argument('--shape_loss', default=False, type=bool, help='If True, contrastive loss is applied to generator')
    
    # info about this run
    parser.add_argument('--use_wandb', default=False, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--wandb_id', default='123456', type=bool, help='unique IDs for wandb run')
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    parser.add_argument('--wandb_entity', default='your-login', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    # parser.add_argument('--optim_level', default='O2', type=str)
    parser.add_argument('--optim_level', default='None', type=str)
    parser.add_argument('--device', default='cuda', type=str, help='setting device between cuda and cpu')

    args = parser.parse_args()
    


    # if bool(config['vgg_data_path)==False and config['same_identity==True:
    #     raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if args.use_wandb==True:
        wandb.init(project=args.wandb_project, 
                   entity=args.wandb_entity, 
                   settings=wandb.Settings(start_method='fork'),
                   id=args.wandb_id,
                   resume='allow')
        config = wandb.config
        # config.vgg_data_path = config['vgg_data_path
        config.ffhq_data_path = args.ffhq_data_path
        # config.celeba_data_path = config['celeba_data_path
        # config.dob_data_path = config['dob_data_path
        config.weight_adv = args.weight_adv
        config.weight_attr = args.weight_attr
        config.weight_id = args.weight_id
        config.weight_rec = args.weight_rec
        config.weight_eyes = args.weight_eyes
        config.same_person = args.same_person
        config.same_identity = args.same_identity
        config.diff_eq_same = args.diff_eq_same
        config.discr_force = args.discr_force
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.eye_detector_loss = args.eye_detector_loss
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.D_path = args.D_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.lr_D = args.lr_D
    elif not os.path.exists('./images'):
        os.mkdir('./images')
    
    # Создаем папки, чтобы было куда сохранять последние веса моделей, а также веса с каждой эпохи
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
        os.mkdir(f'./current_models_{args.run_name}')
    
    main(args)