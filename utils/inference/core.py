from typing import List, Tuple, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .faceshifter_run import faceshifter_batch
from .image_processing import crop_face, normalize_and_torch, normalize_and_torch_batch
from .video_processing import read_video, crop_frames_and_get_transforms, resize_frames


def transform_target_to_torch(resized_frs: np.ndarray, half=True) -> torch.tensor:
    """
    Transform target, so it could be used by model
    """
    target_batch_rs = torch.from_numpy(resized_frs.copy()).cuda()
    target_batch_rs = target_batch_rs[:, :, :, [2,1,0]]/255.  ##RGB로 바꾸고 정규화
        
    if half:
        target_batch_rs = target_batch_rs.half()  ##self.half() is equivalent to self.to(torch.float16)
        
    target_batch_rs = (target_batch_rs - 0.5)/0.5 # normalize
    target_batch_rs = target_batch_rs.permute(0, 3, 1, 2) ##(B, C, H, W)
    
    return target_batch_rs


def model_inference(full_frames: List[np.ndarray],
                    source: List,
                    target: List, 
                    netArc: Callable,
                    G: Callable,
                    app: Callable,  ##app = Face_detect_crop(name='antelope', root='./insightface_func/models')
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using original frames get faceswaped frames and transofrmations
    """
    # Get Arcface embeddings of target image
    target_norm = normalize_and_torch_batch(np.array(target)) ##이미지를 정규화해주고 torch tensor로 만든다
    
    ##여기에 arcface나 hififace를 넣는다. 즉 hififace 코드 업데이트 요망
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))  ##target identity를 extract하기 위해 arcface사용
    
    # Get the cropped faces from original frames and transformations to get those crops
    ##crop_frames_and_get_transforms에서도 arface대체하는 hififace 코드도 업데이트 해 놓아야 할 것으로 보인다.
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(full_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th) ##app ->##app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    
    
    # Normalize source images and transform to torch and get Arcface embeddings
    source_embeds = []
    sources_tensor = []
    for source_curr in source:
        source_curr = normalize_and_torch(source_curr)
        
        ##Hififace 코드 업데이트 요망 
        sources_tensor.append(source_curr)
        source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
    
    final_frames_list = []
    for idx, (crop_frames, tfm_array, source_embed, source_tensor) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds, sources_tensor)):
        # Resize croped frames and get vector which shows on which frames there were faces
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # transform embeds of Xs and target frames to use by model
        target_batch_rs = transform_target_to_torch(resized_frs, half=half)

        if half:
            source_embed = source_embed.half()
            source_tensor = source_tensor.half()

        # run model
        size = target_batch_rs.shape[0]
        model_output = []

        for i in tqdm(range(0, size, BS)):
            Y_st = faceshifter_batch(source_embed, target_batch_rs[i:i+BS], source_tensor, G)
            model_output.append(Y_st)
        torch.cuda.empty_cache()
        model_output = np.concatenate(model_output)

        # create list of final frames with transformed faces
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present):
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list.append(final_frames)
    
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list   