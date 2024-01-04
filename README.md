# FS-Ghost

## Face Swap Model Ghost (Sber-Swap) Code & Docs

### 1. 환경설정 (hojun-mae 기준)

```
https를 사용하여 git clone을 할것이라면
1-1. git clone git@github.com:dob-world/FS-Ghost.git 
or
ssh를 이용하여 git clone을 할 것이라면
1-2. git clone https://github.com/dob-world/FS-Ghost.git

2. cd FS-Ghost
3. git checkout hojun-mae
4. git submodule init
5. git submodule update
6. ./mae/models_mae.py 의 line 40, 54에서 qk_scale=None를 제거
7. from util.pos_embed import get_2d_sincos_pos_embed을 from mae.util.pos_embed import get_2d_sincos_pos_embed
8. root 디렉토리에 .devcontainer에서 두개의 --name 파트를 도커 컨테이너를 짓고 싶은 이름으로 바꾸어 넣는다. --name 두개에 모두 같은 이름으로 넣는다 (만에하나 모르니)
```

### 2. Weight 다운로드

```
sh download_models.sh
```

### 3. Docker 생성

```
docker build -t ghost:latest .

## GPU / 데이터셋 / 디렉토리 정보 등 환경에 맞게 변경 필요
docker run -d -it -v .:/workspace --gpus all --name ghost_container ghost 

docker exec -it ghost_container /bin/bash
```

### 4. 기타 (Pretrained Model 다운로드 - Kakao ML epoch 75회 학습)

구글 드라이브 : [ckpt_epoch75 - Google Drive](https://drive.google.com/drive/folders/1JmS-y-zAH0-DtC-oG0OPVcidVOU99sNA?usp=sharing)

### 5. Inference

- Image Swap
  
  ```
  python inference.py --target_path {PATH_TO_IMAGE} --image_to_image True --G_path {Pretrained_Model_PATH}
  ```
  
- Video Swap
  
  ```
  python inference.py --source_paths {PATH_TO_IMAGE} --target_faces_paths {PATH_TO_IMAGE} --target_video {PATH_TO_VIDEO} --G_path {Pretrained_Model_PATH}
  ```
  

### 6. Train
** AdaptiveWingLoss 별도 다운로드 필요 : https://github.com/NastyaMittseva/AdaptiveWingLoss/tree/6a6e42fcc435dc8fd604f211390099725a4222a6
```
python train.py --run_name {YOUR_RUN_NAME} --pretrained True --G_path {PATH_TO_GPATH} -- D_path {PATH_TO_G_PATH}
```
=======
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9851423)] [[Habr](https://habr.com/ru/company/sberbank/blog/645919/)]

# 👻 GHOST: Generative High-fidelity One Shot Transfer 

Our paper ["GHOST—A New Face Swap Approach for Image and Video Domains"](https://ieeexplore.ieee.org/abstract/document/9851423) has been published on IEEE Xplore.

<p align="left">
  Google Colab Demo
</p>
<p align="left">
  <a href="https://colab.research.google.com/drive/1vXTpsENipTmjTMggwveCkXASwxUk270n">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
</p>

## GHOST Ethics 

Deepfake stands for a face swapping algorithm where the source and target can be an image or a video. Researchers have investigated sophisticated generative adversarial networks (GAN), autoencoders, and other approaches to establish precise and robust algorithms for face swapping. However, the achieved results are far from perfect in terms of human and visual evaluation. In this study, we propose a new one-shot pipeline for image-to-image and image-to-video face swap solutions - GHOST (Generative High-fidelity One Shot Transfer).

Deep fake synthesis methods have been improved a lot in quality in recent years. The research solutions were wrapped in easy-to-use API, software and different plugins for people with a little technical knowledge. As a result, almost anyone is able to make a deepfake image or video by just doing a short list of simple operations. At the same time, a lot of people with malicious intent are able to use this technology in order to produce harmful content. High distribution of such a content over the web leads to caution, disfavor and other negative feedback to deepfake synthesis or face swap research.

As a group of researchers, we are not trying to denigrate celebrities and statesmen or to demean anyone. We are computer vision researchers, we are engineers, we are activists, we are hobbyists, we are human beings. To this end, we feel that it's time to come out with a standard statement of what this technology is and isn't as far as us researchers are concerned.
* GHOST is not for creating inappropriate content.
* GHOST is not for changing faces without consent or with the intent of hiding its use.
* GHOST is not for any illicit, unethical, or questionable purposes.
* GHOST exists to experiment and discover AI techniques, for social or political commentary, for movies, and for any number of ethical and reasonable uses.

We are very troubled by the fact that GHOST can be used for unethical and disreputable things. However, we support the development of tools and techniques that can be used ethically as well as provide education and experience in AI for anyone who wants to learn it hands-on. Now and further, we take a **zero-tolerance approach** and **total disregard** to anyone using this software for any unethical purposes and will actively discourage any such uses.


## Image Swap Results 
![](/examples/images/example1.png)

![](/examples/images/example2.png)

## Video Swap Results
<div>
<img src="/examples/videos/orig.webp" width="360"/>
<img src="/examples/videos/elon.webp" width="360"/>
<img src="/examples/videos/khabenskii.webp" width="360"/>
<img src="/examples/videos/mark.webp" width="360"/>
<div/>

## Installation
  
1. Clone this repository
  ```bash
  git clone https://github.com/sberbank-ai/sber-swap.git
  cd sber-swap
  git submodule init
  git submodule update
  ```
2. Install dependent packages
  ```bash
  pip install -r requirements.txt
  ```
  If it is not possible to install onnxruntime-gpu, try onnxruntime instead  
  
3. Download weights
  ```bash
  sh download_models.sh
  ```
## Usage
  1. Colab Demo <a href="https://colab.research.google.com/drive/1B-2JoRxZZwrY2eK_E7TB5VYcae3EjQ1f"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> or you can use jupyter notebook [SberSwapInference.ipynb](SberSwapInference.ipynb) locally
  2. Face Swap On Video
  
  Swap to one specific person in the video. You must set face from the target video (for example, crop from any frame).
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE} --target_faces_paths {PATH_TO_IMAGE} --target_video {PATH_TO_VIDEO}
  ```
  Swap to many person in the video. You must set multiple faces for source and the corresponding multiple faces from the target video.
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_faces_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_video {PATH_TO_VIDEO}
  ```
  3. Face Swap On Image
  
  You may set the target face, and then source will be swapped on this person, or you may skip this parameter, and then source will be swapped on any person in the image.
  ```bash
  python inference.py --target_path {PATH_TO_IMAGE} --image_to_image True
  ```
  
## Training
  
We also provide the training code for face swap model as follows:
  1. Download [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
  2. Crop and align faces with out detection model.
  ```bash
  python preprocess_vgg.py --path_to_dataset {PATH_TO_DATASET} --save_path {SAVE_PATH}
  ```
  3. Start training. 
  ```bash
  python train.py --run_name {YOUR_RUN_NAME}
  ```
We provide a lot of different options for the training. More info about each option you can find in `train.py` file. If you would like to use wandb logging of the experiments, you should login to wandb first  `--wandb login`.
  
### Tips
  1. For the first epochs we suggest not to use eye detection loss and scheduler if you train from scratch.
  2. In case of finetuning you can variate losses coefficients to make the output look similar to the source identity, or vice versa, to save features and attributes of target face.
  3. You can change the backbone of the attribute encoder and num_blocks of AAD ResBlk using parameters `--backbone` and `--num_blocks`.
  4. During the finetuning stage you can use our pretrain weights for generator and discriminator that are located in `weights` folder. We provide the weights for models with U-Net backbone and 1-3 blocks in AAD ResBlk. The main model architecture contains 2 blocks in AAD ResBlk.
  
## Cite
If you use our model in your research, we would appreciate using the following citation

  ### BibTeX Citation
  ```
  @article{9851423,  
           author={Groshev, Alexander and Maltseva, Anastasia and Chesakov, Daniil and Kuznetsov, Andrey and Dimitrov, Denis},  
           journal={IEEE Access},   
           title={GHOST—A New Face Swap Approach for Image and Video Domains},   
           year={2022},  
           volume={10},  
           number={},  
           pages={83452-83462},  
           doi={10.1109/ACCESS.2022.3196668}
  }
  ```
  
  ### General Citation
  
  A. Groshev, A. Maltseva, D. Chesakov, A. Kuznetsov and D. Dimitrov, "GHOST—A New Face Swap Approach for Image and Video Domains," in IEEE Access, vol. 10, pp. 83452-83462, 2022, doi: 10.1109/ACCESS.2022.3196668.
  
>>>>>>> ae224d3b880759f007ac9fc1178c9c2f35587d40
