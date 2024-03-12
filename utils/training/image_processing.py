import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def torch2image(torch_image: torch.tensor) -> np.ndarray:  ##torch 텐서를 이미지 255로 de-normalization을 취하기 위한 reverse 프로세스
    batch = False
    
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    
    device = torch_image.device
    # mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    # std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    
    denorm_image = (std * torch_image) + mean
    
    if batch:
        denorm_image = denorm_image.permute(0, 2, 3, 1)
    else:
        denorm_image = denorm_image.permute(1, 2, 0)
    
    np_image = denorm_image.detach().cpu().numpy()
    np_image = np.clip(np_image*255., 0, 255).astype(np.uint8)
    
    if batch:
        return np.concatenate(np_image, axis=1)
    else:
        return np_image


def make_image_list(images) -> np.ndarray:    
    np_images = []
    
    for torch_image in images:
        np_img = torch2image(torch_image)
        np_images.append(np_img)
    
    return np.concatenate(np_images, axis=0)


def read_torch_image(path: str) -> torch.tensor:
    
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = Image.fromarray(image[:, :, ::-1])
    image = transformer_Arcface(image)
    image = image.view(-1, image.shape[0], image.shape[1], image.shape[2])
    
    return image


def get_faceswap_(source_path: str, target_path: str, 
                 G: 'generator model', netArc: 'arcface model', 
                 device: 'torch device') -> np.array:
    source = read_torch_image(source_path)
    source = source.to(device)

    embeds = netArc(F.interpolate(source, [112, 112], mode='bilinear', align_corners=False))
    # embeds = F.normalize(embeds, p=2, dim=1)

    target = read_torch_image(target_path)
    target = target.cuda()

    with torch.no_grad():
        Yt, _ = G(target, embeds)
        Yt = torch2image(Yt)

    source = torch2image(source)
    target = torch2image(target)

    return np.concatenate((cv2.resize(source, (256, 256)), target, Yt), axis=1)  


# Xt_f, Xs_f, id_embedding

def get_faceswap(source_path: str, target_path: str, 
                 G: 'generator model', netArc: 'arcface model', 
                 device: 'torch device') -> np.array:
    
    
    source = read_torch_image(source_path)
    source = source.to(device)

    embeds = netArc(F.interpolate(source, [112, 112], mode='bilinear', align_corners=False))
    # embeds = F.normalize(embeds, p=2, dim=1)
    
    # ## hojun added
    # id_embedding = 

    target = read_torch_image(target_path)
    target = target.cuda()

    with torch.no_grad():
        swapped_face, _, _ = G(Xt_f, Xs_f, id_embedding)
        swapped_face = torch2image(swapped_face)

    source = torch2image(source)
    target = torch2image(target)
    
    
    

    return np.concatenate((cv2.resize(source, (256, 256)), target, swapped_face), axis=1)  


