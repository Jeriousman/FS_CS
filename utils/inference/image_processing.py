import base64
from io import BytesIO
from typing import Callable, List

import numpy as np
import torch
import cv2
from .masks import face_mask_static 
from matplotlib import pyplot as plt
from insightface.utils import face_align


def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    """
    kps = app.get(image_full, crop_size)  ##app.get은 insightFace의 문법인듯  kps = cropped face  
    M, _ = face_align.estimate_norm(kps[0], crop_size, mode ='None')   ##M은 
    align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0) ##M은 warpAffine을 하기 위한 transition matrix로 사용된다 
    return [align_img]


def normalize_and_torch(image: np.ndarray) -> torch.tensor:  ##값을 노멀라이징하고 토치 텐서 형태로 만든다 
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray) -> torch.tensor:
    """
    Normalize batch images and transform to torch
    """
    batch_frames = torch.from_numpy(frames.copy()).cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.

    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def get_final_image(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    handler) -> None:
    """
    Create final video from frames
    """
    final = full_frame.copy()
    params = [None for i in range(len(final_frames))]
    
    for i in range(len(final_frames)):
        frame = cv2.resize(final_frames[i][0], (224, 224)) ##for loop을 통해 순차적으로 Frame을 224로 resize해준다.
        
        landmarks = handler.get_without_detection_without_transform(frame)     
        landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i][0])

        mask, _ = face_mask_static(crop_frames[i][0], landmarks, landmarks_tgt, params[i])
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0]) ##https://076923.github.io/docs/invertAffineTransform affine한것을 반대로 역행렬을 구한다. 타겟 inverted mask를 가져온다.

        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)  ##final_swapped_face_frame에 target_mat_rev를 집어넣어 얼라인하는 부분인 것 같다.
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))  ##swapp한 마스크 부분으로 얼라인하는 부분인것같다.
        mask_t = np.expand_dims(mask_t, 2)  ## axis 2에다가 dimension을 추가하라는 뜻 

        final = mask_t*swap_t + (1-mask_t)*final
    final = np.array(final, dtype='uint8')
    return final


def show_images(images: List[np.ndarray], 
                titles=None, 
                figsize=(20, 5), 
                fontsize=15):
    if titles:
        assert len(titles) == len(images), "Amount of images should be the same as the amount of titles"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image[:, :, ::-1])
        if titles:
            ax.set_title(titles[idx], fontsize=fontsize)
        ax.axis("off")
