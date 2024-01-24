print("started imports")

import sys
import argparse
import time
import cv2
import wandb
from PIL import Image
import os

##For Native Torch multi GPUs
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group


from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduler

# # custom imports
# sys.path.append('./apex/')
# from apex import amp
from network.CrossU import CrossUnetAttentionGenerator, UNet
# from network.AEI_Net import *
# from network.MultiscaleDiscriminator import *
from utils.training.Dataset import FaceEmbedCombined, FaceEmbed, FaceEmbedSubdir, FaceEmbedFFHQ, FaceEmbedCelebA, FaceEmbedCustom#FaceEmbedAllFlat
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.losses import hinge_loss, compute_discriminator_loss, compute_generator_losses
from utils.training.detector import detect_landmarks, paint_eyes
from AdaptiveWingLoss.core import models
from arcface_model.iresnet import iresnet100
from models.models import FlowFaceCrossAttentionModel, FlowFaceCrossAttentionLayer
import torch
# from mae import models_mae
print("finished imports")


# def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
#     # build model
#     model = getattr(models_mae, arch)()
#     # load model
#     checkpoint = torch.load(chkpt_dir, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     print(msg)
#     return model



def train_one_epoch(G: 'generator model', 
                    D: 'discriminator model', 
                    opt_G: "generator opt", 
                    opt_D: "discriminator opt",
                    scheduler_G: "scheduler G opt",
                    scheduler_D: "scheduler D opt",
                    netArc: 'ArcFace model',
                    model_ft: 'Landmark Detector',
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int,
                    loss_adv_accumulated:int
                    # config:dict
                    ):
    
    # # Only initialize W&B on the global rank 0 node
    # if config['global_rank'] == 0:
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project=args.wandb_project,
    #         # allow resuming existing run with the same name (in case the rank 0 node crashed)
    #         id=args.wandb_entity,
    #         resume="allow",
    #         # track hyperparameters and run metadata
    #         config=args
    #     )


    # zz = FlowFaceCrossAttentionLayer(args.n_head, args.k_dim, args.q_dim, args.q_dim)
    # MAE.to(device)
    # MAE.eval()
    FFCA = FlowFaceCrossAttentionModel(seq_len=args.seq_len, n_head=args.n_head, k_dim=args.k_dim, q_dim=args.q_dim, kv_dim=args.kv_dim)
    # FFCA.to(device)
    # FFCA.train()
    
    unet = UNet(backbone='unet').to(device)
    z = unet(Xs)
    z[0].shape
    z[1].shape
    z[2].shape
    z[3].shape
    z[4].shape
    z[5].shape
    z[6].shape
    z[7].shape
    
    # z[7].shape
    # z[6].shape
    # Xs.shape
    # Xs

        # conv1 = conv4x4(3, 32,same_size=True).to(device)  ##
        # z1 = conv1(Xs)
        # z1.shape
        
        # conv2 = conv4x4(32, 64).to(device) 
        # z2 = conv2(z1)
        # z2.shape
        
        # conv3 = conv4x4(64, 128).to(device) 
        # z3 = conv3(z2)
        # z3.shape
        
        # conv4 = conv4x4(128, 256).to(device)
        # z4 = conv4(z3)
        # z4.shape
        
        # conv5 = conv4x4(256, 512).to(device)
        # z5 = conv5(z4)
        # z5.shape
        # conv6 = conv4x4(512, 1024).to(device)
        # z6 = conv6(z5)
        # z6.shape
        # conv7 = conv4x4(1024, 1024).to(device)
        # z7 = conv7(z6)
        # z7.shape
        


        # deconv1 = deconv4x4(1024, 1024).to(device)
        # o1 = deconv1(z7, z6 ,backbone='unet')
        # o1.shape
        # z6.shape
        # z5.shape
        # z4.shape
        # deconv2 = deconv4x4(1024, 512).to(device)  ## 8 ->  16
        # o2 = deconv2(o1, z5 ,backbone='unet')
        # o2.shape
        # deconv3 = deconv4x4(512, 256).to(device)  ## 16 > 32
        # o3 =  deconv3(o2, z4, backbone='unet')
        # o3.shape
        # deconv4 = deconv4x4(256, 128).to(device) ## 32 > 64
        # o4 =  deconv4(o3, z3, backbone='unet')
        # o4.shape
        # deconv5 = deconv4x4(128, 64).to(device) ## 64 > 128
        # o5 =  deconv5(o4, z2, backbone='unet')
        # o5.shape
        
        # deconv6 = deconv4x4(64, 32).to(device) ## 64 > 128
        # o6 =  deconv6(o5, z1, backbone='unet')
        # o6.shape
        
        # deconv7 = outconv(32, 3).to(device) ## 64 > 128
        # o7 =  deconv7(o6)
        # o7.shape
        
        
        
    
    
    # Xs.shape
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        Xs_mae, Xs, Xt, same_person = data


        Xs_mae.to(device)  ##Xs_mae = batch_size, 3, 224, 224
        Xs = Xs.to(device)
        Xs.shape
        Xt = Xt.to(device)
        Xt.shape
        same_person = same_person.to(device)
        realtime_batch_size = Xs.shape[0] 
        break
        
        # z,zz,zzz = G(Xs, Xt)
        # get the identity embeddings of Xs
        with torch.no_grad():
            embed = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))
            


        diff_person = torch.ones_like(same_person)

        if args.diff_eq_same:
            same_person = diff_person
        ##Generator
        ##Generator option 1 = ViT decoder
        
        ##Generator option 2 = UMAP multiscale convolutional decoder
        
        
        # generator training
        opt_G.zero_grad() ##축적된 gradients를 비워준다
        
        Y, Xt_attr = G(Xt, embed) ##제너레이터에 target face와 source face identity를 넣어서 결과물을 만든다. MAE의 경우 Xt_embed, Xs_embed를 넣으면 될 것 같다 (same latent space)
        Di = D(Y)  ##이렇게 나온 Y = swapped face 결과물을 Discriminator에 넣어서 가짜로 구별을 해내는지 확인해 보는 것이다. 0과 가까우면 가짜라고하는것이다.
        ZY = netArc(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False))   ##swapped face의 identity를  ArcFace를 사용해서 구하는 것
        
    
        if args.eye_detector_loss:
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, model_ft)  ##detect_landmarks 부문에 다른 eye loss 뿐만이 아니라 다른 part도 계산하고 싶으면 여기다 코드를 추가해서 넣으면 될거같다
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, model_ft)
            eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
        else:
            eye_heatmaps = None
            
        lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = compute_generator_losses(G, Y, Xt, Xt_attr, Di,
                                                                             embed, ZY, eye_heatmaps, loss_adv_accumulated, 
                                                                             diff_person, same_person, args)
        
        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()
        opt_G.step()
        if args.scheduler:
            scheduler_G.step()
        
        # discriminator training
        opt_D.zero_grad()
        lossD = compute_discriminator_loss(D, Y, Xs, diff_person)
        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()

        if (not args.discr_force) or (loss_adv_accumulated < 4.):
            opt_D.step()
        if args.scheduler:
            scheduler_D.step()
        
        
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
            if args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(Xt, Xt_eyes)
                Yt_eyes_img = paint_eyes(Y, Y_eyes)
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/generated_image.jpg', image[:,:,::-1])
        
        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if args.eye_detector_loss:
                print(f'L_l2_eyes: {L_l2_eyes.item()}')
            print(f'loss_adv_accumulated: {loss_adv_accumulated}')
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}')
        
        if args.use_wandb:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            wandb.log({"loss_id": L_id.item(),
                       "lossD": lossD.item(),
                       "lossG": lossG.item(),
                       "loss_adv": L_adv.item(),
                       "loss_attr": L_attr.item(),
                       "loss_rec": L_rec.item()})
        
        if iteration % 10000 == 0:
            
            if gpu_config['global_rank'] == 0:
                
                torch.save(G.state_dict(), f'./saved_models_{args.run_name}/G_latest.pth')
                torch.save(D.state_dict(), f'./saved_models_{args.run_name}/D_latest.pth')

                torch.save(G.state_dict(), f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
                torch.save(D.state_dict(), f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        if (iteration % 250 == 0) and (args.use_wandb) and gpu_config['global_rank'] == 0:
            ### Посмотрим как выглядит свап на трех конкретных фотках, чтобы проследить динамику
            G.eval()

            res1 = get_faceswap('examples/images/training//source1.png', 'examples/images/training//target1.png', G, netArc, device)
            res2 = get_faceswap('examples/images/training//source2.png', 'examples/images/training//target2.png', G, netArc, device)  
            res3 = get_faceswap('examples/images/training//source3.png', 'examples/images/training//target3.png', G, netArc, device)
            
            res4 = get_faceswap('examples/images/training//source4.png', 'examples/images/training//target4.png', G, netArc, device)
            res5 = get_faceswap('examples/images/training//source5.png', 'examples/images/training//target5.png', G, netArc, device)  
            res6 = get_faceswap('examples/images/training//source6.png', 'examples/images/training//target6.png', G, netArc, device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})

            G.train()

# def train(args, config):
def train(args, device):
    
    # ##Multi GPU setting
    # assert torch.cuda.is_available(), "Training on CPU is not supported as Multi-GPU strategy is set"
    # device = torch.device('cuda')
    # print(f"GPU {gpu_config['local_rank']} is using device: {device}")
    # print(f"GPU {gpu_config['local_rank']} is loading dataset")
    

    
    
    
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    # G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    G = CrossUnetAttentionGenerator(backbone='unet').to(device)
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)    
    
    G.train()
    D.train()
    
    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
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
        model_ft = model_ft.to(device)
        model_ft.eval()
    else:
        model_ft=None
    
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)

    G, opt_G = amp.initialize(G, opt_G, opt_level=args.optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=args.optim_level)
    
    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        scheduler_D = None
        
    if args.pretrained:
        try:
            G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            D.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G and D")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    # if args.vgg:
    
    # dataset = FaceEmbedCombined(args.vgg_dataset_path, args.dob_dataset_path, args.celeba_dataset_path, same_prob=0.8, same_identity=args.same_identity)
    # dataset = FaceEmbedCombined(celeba_data_path=args.celeba_data_path, ffhq_data_path=args.ffhq_data_path, vgg_data_path=args.ffhq_data_path, dob_data_path=args.dob_data_path, same_prob=0.8, same_identity=args.same_identity)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataset = FaceEmbedCustom('/workspace/examples/images/training')
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ##In case of multi GPU, turn off shuffle
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=True))


    

    # Будем считать аккумулированный adv loss, чтобы обучать дискриминатор только когда он ниже порога, если discr_force=True
    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        train_one_epoch(G,
                        D,
                        opt_G,
                        opt_D,
                        scheduler_G,
                        scheduler_D,
                        netArc,
                        model_ft,
                        args,
                        dataloader,
                        device,
                        device,
                        epoch,
                        loss_adv_accumulated)
                        # config)

def main(args):
    
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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')
    
    print("Starting traing")
    train(args, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset params
    ##the 4 arguments are newly added by Hojun
    parser.add_argument('--vgg_data_path', default='/datasets/VGG', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    parser.add_argument('--ffhq_data_path', default='/datasets/FFHQ', type=str,help='Paasdasde')
    parser.add_argument('--celeba_data_path', default='/datasets/CelebHQ/CelebA-HQ-img', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    # parser.add_argument('--dob_data_path', default='/datasets/DOB', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    
    parser.add_argument('--G_path', default='./saved_models/G.pth', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--D_path', default='./saved_models/D.pth', help='Path to pretrained weights for D. Only used if pretrained=True')
    
    
    # weights for loss
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=20, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=0., type=float, help='Eyes Loss weight')
    # training params you may want to change
    
    
    ##parameters for model configs
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')

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
    parser.add_argument('--landmark_detector_loss', default=True, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # info about this run
    parser.add_argument('--use_wandb', default=False, type=bool, help='Use wandb to track your experiments or not')
    # parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    parser.add_argument('--wandb_entity', default='your-login', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    # parser.add_argument('--optim_level', default='O2', type=str)
    parser.add_argument('--optim_level', default='None', type=str)

    args = parser.parse_args()
    
    
    # if bool(args.vgg_data_path)==False and args.same_identity==True:
    #     raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if args.use_wandb==True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, settings=wandb.Settings(start_method='fork'))
        config = wandb.config
        config.vgg_data_path = args.vgg_data_path
        config.ffhq_data_path = args.ffhq_data_path
        config.celeba_data_path = args.celeba_data_path
        config.dob_data_path = args.dob_data_path
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
