from face_parsing_pytorch.logger import setup_logger

from face_parsing_pytorch.model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse
import glob
import torch.nn.functional as F

def vis_parsing_maps(args, im, parsing_anno, stride, save_path):
    
    # Colors for all 20 parts (Just Sample)
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    # target_body_parts
    body_parts = ["background", "face", "left_eyebrow", "right_eyebrow", "left_eye", "right_eye", "glasses", "left_ear", "right_ear", "earring", "nose",\
                   "tooth", "uppder_lip", "lower_lip", "neck", "necklace", "clothes", "hair",  "hat"]
    
    # background area index : 1~5, 7,8,10, 11~13
    face_parts = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
    background_parts = [i for i in range(0,19) if i not in face_parts]
    
    # landmark index (temp)
    #landmark_parts = [2, 3, 7, 8, 10, 11, 12, 13]

    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    
    # Create Empty Numpy
    color_face_img = im.copy().astype(np.uint8)
    color_background_img = im.copy().astype(np.uint8)

    ## Generate maskingimg
    if args.maskingimg:
        face_mask = np.isin(vis_parsing_anno, face_parts)
        background_mask = ~face_mask
        masking_background_ndarray = background_mask.astype(int)
        masking_face_ndarray = face_mask.astype(int)

        img_filename = save_path[:-4] + '_b_m.jpg'
        # print("Masking Image : ", img_filename)
        cv2.imwrite(img_filename, masking_background_ndarray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        img_filename = save_path[:-4] + '_f_m.jpg'
        # print("Masking Image : ", img_filename)
        cv2.imwrite(img_filename, masking_face_ndarray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Generate Original Color Img in this masking img
        if args.originalimg:
            background_index = np.where(masking_background_ndarray == 1)
            face_index = np.where(masking_face_ndarray == 1)

            base_color_img = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
            base_color_img[background_index[0], background_index[1], :] = part_colors[0]
            img_filename = save_path[:-4] + '_b_m_o.jpg'
            cv2.imwrite(img_filename, base_color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            base_color_img = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
            base_color_img[face_index[0], face_index[1], :] = part_colors[1]
            img_filename = save_path[:-4] + '_f_m_o.jpg'
            cv2.imwrite(img_filename, base_color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


    ## Generate Colorimg
    if args.colorimg:
        face_mask = np.isin(vis_parsing_anno, face_parts)
        background_mask = ~face_mask

        color_background_img[face_mask] = [0, 0, 0]
        color_face_img[background_mask] = [0, 0, 0]

        img_filename = save_path[:-4] + '_b.jpg'
        # print("Color Image : ", img_filename)
        cv2.imwrite(img_filename, color_background_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


        img_filename = save_path[:-4] + '_f.jpg'
        # print("Color Image : ", img_filename)
        cv2.imwrite(img_filename, color_face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



            
    # 0.배경 1.얼굴(피부만) 2.왼쪽눈썹 3.오른쪽눈썹 4.왼쪽눈 5.오른쪽눈 6.안경 7.왼쪽귀 8.오른쪽귀 9.귀걸이 10.코
    # 11.치아 12.윗입술 13.아랫입술 14.목 15.목걸이(추측) 16.옷 17.머리카락 18.모자

    ## 1.마스킹 처리된 이미지 (ImgFileName_)
    ## 1~5, 7,8,10, 11~13 (이것만 합친 결과물) 1
    ## 나머지 0으로 처리 

    ## File Name 
    ## 마스킹 처리된 배경 이미지 : ImgFileName_b_m
    ## 마스킹 처리된 얼굴 이미지 : ImgFileName_f_m
    ## 배경 이미지 : ImgFileName_b
    ## 얼굴 이미지 : ImgFileName_f
    ## Landmark 이름 : ImgFileName_lms
    ## 순서대로
                    
    ## 2. 원본 이미지
    ## 마스킹 처리된 이미지의 원본만 남기기(원본 pixel 그대로 남아있기)
    ## 얼굴 값만 있는거 + 배경만 남겨져 있는것
            
    ## 3. landmark image 
    ## landmark의 좌표로 할지? 고민 하고 하기.
    ## landmark별 loss 계산 가능 (2,3,7,8,10,11,12,13)
    ## 얘네만 값을 유지하고, 나머지는 버리는 방식


def parsing(args):
    respth = args.respth
    dspth = args.dspth
    cp = args.modelpth
    
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Directory 구조 (기준 Dataset : FFHQ)
    with torch.no_grad():
        folders_list = glob.glob(f'{dspth}/*')
        
        for folder in folders_list:
            print("Parsing Started : ", folder)
            
            # 미변환할 데이터 혹은 폴더 리스트 (제거 해도 상관없음)
            lists = ['ffhq_list.txt']
            if(folder.split('/')[-1] in lists):
                continue
            for image_path in os.listdir(folder):
                img = Image.open(osp.join(folder, image_path))
                image = img.resize((512, 512), Image.BILINEAR)
                resized_image = img.resize((256, 256), Image.BILINEAR)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = net(img)[0]
                
                #resize Images
                out = F.interpolate(out, size=(256,256), mode='bilinear', align_corners=False)
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                # 상위 계층 1개 기준
                result_dir_name = folder.split('/')[-1]
                result_path = osp.join(respth, result_dir_name)

                # 결과 생성 폴더 생성
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                vis_parsing_maps(args, resized_image, parsing, stride=1, save_path=osp.join(result_path, image_path))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--respth', default='/datasets/FFHQ_parsed_img', help='Path of parsed image')
    # Dataset 이미지 계층 단계에서 상위 폴더는 1개여야만 함. (ex. FFHQ/40000/1.jpg에서 FFHQ까지만 입력)
    parser.add_argument('--dspth', default='/datasets/FFHQ', help='Path to the dataset')
    parser.add_argument('--modelpth', default='/workspace/res/cp/79999_iter.pth', help='Path to the parsing model')
    parser.add_argument('--maskingimg', default=True, type=bool, help="If you want to create face/background masking image, set True")
    parser.add_argument('--colorimg', default=True, type=bool, help="If you want to create face/background color image, set True")
    parser.add_argument('--landmarkimg', default=True, type=bool, help="If you want to create landmark, set True")
    # Masking된 이미지만 채색하여 보여줌 (--maskingimg True일때만 사용)
    parser.add_argument('--originalimg', default=True, type=bool, help="If you want to generate original segmentation face model")

    args = parser.parse_args()

    parsing(args)
    

    