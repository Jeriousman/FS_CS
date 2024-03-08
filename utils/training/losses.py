import torch
from network.MultiscaleDiscriminator import PatchDiscriminator
from torch.nn import functional as F
from info_nce import InfoNCE, info_nce

# from info_nce import InfoNCE, info_nce
##https://github.com/RElbers/info-nce-pytorch
l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()


def hinge_loss(X, positive=True): ## https://m.blog.naver.com/wooy0ng/222666100291
    if positive:
        return torch.relu(1-X) ##X = y_hat * true_y if normal hinge loss
    else:
        return torch.relu(X+1)




    
# def compute_generator_losses(G, Y, Xt, Xt_attr, Di, embed, ZY, eye_heatmaps, loss_adv_accumulated, ##Y = swapped face ##Xt_attr = target image multi scale features
#                              diff_person, same_person, args):
    
def compute_generator_losses(G, Y, Xt, Xs, Xt_attr, Di, eye_heatmaps, loss_adv_accumulated, ##Y = swapped face ##Xt_attr = target image multi scale features
                             diff_person, same_person, args, id_emb, tgt_id_emb, swapped_id_emb, recon_source, recon_target):
    # adversarial loss
    L_adv = 0.
    for di in Di:
        L_adv += hinge_loss(di[0], True).mean(dim=[1, 2, 3])
    L_adv = torch.sum(L_adv * diff_person) / (diff_person.sum() + 1e-4)

    # id loss
    L_id =(1 - torch.cosine_similarity(id_emb, swapped_id_emb, dim=1)).mean()  ##id_embed = source id embed. 
    ##즉, embed는 source face의 arcface embedding, ZY는 swapped Face의 embedding?

    # attr loss  ##Y_attr is the target multi scale attr
    if args.optim_level == "O2" or args.optim_level == "O3":
        Y_attr = G.ca_forward(Xt.type(torch.half), Xs.type(torch.half))
        
    else:
        Y_attr = G.ca_forward(Xt, Xs)
    
    L_attr = 0
    for i in range(len(Xt_attr)): 
        L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(args.batch_size, -1), dim=1).mean()
    L_attr /= 2.0 ##2 bcuz ?

    # reconstruction loss ##같은 인물이지만 같은 데이터일 필요는 없다고 생각하기 때문에 같은 사람것을 사용 
    L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
    
    ## Source Unet Loss
    L_source_unet = l2_loss(Xs, recon_source)
    
    ## Target Unet Loss
    L_target_unet = l2_loss(Xt, recon_target)
    
    # l2 eyes loss
    if args.eye_detector_loss:
        Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right = eye_heatmaps
        L_l2_eyes = l2_loss(Xt_heatmap_left, Y_heatmap_left) + l2_loss(Xt_heatmap_right, Y_heatmap_right)
    else:
        L_l2_eyes = 0
        
        
    ## Cycle GAN loss 
    
    if args.cycle_loss:
        # swapped_face, recon_src, recon_tgt = G(Xt, Xs, id_emb)
        # cycleloss_src = l1_loss(swapped_face, recon_src)  ##original cycle gan should be:  l1_loss(Xt, recon_src)
        # cycleloss_tgt = l1_loss(swapped_face, recon_tgt)  ##original cycle gan should be:  l1_loss(Xs, recon_tgt)
        cycleloss_src = l1_loss(Y, recon_source)  ##original cycle gan should be:  l1_loss(Xt, recon_src)
        cycleloss_tgt = l1_loss(Y, recon_target)  ##original cycle gan should be:  l1_loss(Xs, recon_tgt)
        L_cycle = cycleloss_src + cycleloss_tgt
        ## identity loss (for cycle GAN)
        # identityloss_src = l1_loss(Xs, recon_src)
        # identityloss_tgt = l1_loss(Xt, recon_tgt) 
        identityloss_src = l1_loss(Xs, recon_source)
        identityloss_tgt = l1_loss(Xt, recon_target) 
        L_cycle_identity = identityloss_src + identityloss_tgt
        

        
    # final loss of generator
    # lossG = args.weight_adv*L_adv + args.weight_attr*L_attr + args.weight_id*L_id + args.weight_rec*L_rec + args.weight_eyes*L_l2_eyes
    lossG = args.weight_adv*L_adv + args.weight_id*L_id + args.weight_attr*L_attr + args.weight_rec*L_rec + args.weight_eyes*L_l2_eyes + args.weight_cycle*L_cycle + args.weight_cycle_identity*L_cycle_identity  + args.weight_constrasive*L_constrasive  + args.weight_source_unet*L_source_unet + args.weight_target_unet*L_target_unet#+ args.weight_landmarks*L_landmarks
    loss_adv_accumulated = loss_adv_accumulated*0.98 + L_adv.item()*0.02
    
    # return lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes
    return lossG, loss_adv_accumulated, L_adv, L_id, L_attr, L_rec, L_l2_eyes, L_cycle, L_cycle_identity, L_constrasive, L_source_unet, L_target_unet#, L_landmarks


def compute_discriminator_loss(D, Y, recon_Xs, recon_Xt, Xs, Xt, diff_person, device, id_embed):
    # fake part
    fake_D = D(Y.detach())
    loss_fake = 0
    for di in fake_D:
        loss_fake += torch.sum(hinge_loss(di[0], False).mean(dim=[1, 2, 3]) * diff_person) / (diff_person.sum() + 1e-4)

    # ground truth part
    true_D = D(Xs)
    loss_true = 0
    for di in true_D:
        loss_true += torch.sum(hinge_loss(di[0], True).mean(dim=[1, 2, 3]) * diff_person) / (diff_person.sum() + 1e-4)

    lossD = 0.5*(loss_true.mean() + loss_fake.mean())
    
    disc_src = PatchDiscriminator(3).to(device)
    disc_tgt = PatchDiscriminator(3).to(device)

    
    ## cyclegan loss for Unet reconstruction
    disc_src_fake = disc_src(recon_Xs)
    disc_src_real = disc_src(Xs)
    disc_tgt_fake = disc_tgt(recon_Xt)
    disc_tgt_real = disc_tgt(Xt)
    
    # disc_src_fake = disc_src(recon_src)
    # disc_src_real = disc_src(Xs)
    # disc_tgt_fake = disc_tgt(recon_tgt)
    # disc_tgt_real = disc_tgt(Xt)
    

    disc_src_fake_loss = l2_loss(disc_src_fake, torch.zeros_like(disc_src_fake))
    disc_src_real_loss = l2_loss(disc_src_real, torch.ones_like(disc_src_real))
    disc_src_loss = disc_src_fake_loss + disc_src_real_loss
    
    disc_tgt_fake_loss = l2_loss(disc_tgt_fake, torch.zeros_like(disc_tgt_fake))
    disc_tgt_real_loss = l2_loss(disc_tgt_real, torch.ones_like(disc_tgt_real))
    disc_tgt_loss = disc_tgt_fake_loss + disc_tgt_real_loss
    
    lossCycle = (disc_src_loss + disc_tgt_loss)/2
    
    return lossD + lossCycle.item()



def infoNce_id_loss(swapped_ids, source_ids, negative_ids):  ##negative_ids = anything thats not swapped and source face. So in our case, it is target face 
        
    loss = InfoNCE(negative_mode='unpaired') # negative_mode='unpaired' is the default value
#     batch_size, num_negative, embedding_size = 32, 48, 128
#     query = torch.randn(batch_size, embedding_size)
#     positive_key = torch.randn(batch_size, embedding_size)
#     negative_keys = torch.randn(num_negative, embedding_size)
    infonce_loss = loss(swapped_ids, source_ids, negative_ids)
    return infonce_loss
#     result = -math.log(exp(F.cosine_similarity(swapepd, source))/(exp(F.cosine_similarity(swapped, target) + sigma(exp(F.cosine_similarity(swapped, negative))))
    