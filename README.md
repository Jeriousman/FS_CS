
# CS

Using a novel cross-attention mechanism + U-Net structure equipped with richer ID extractor.
Science paper work is ongoing (first authur)

### 1. 환경설정 (hojun-mae 기준)

```
https를 사용하여 git clone을 할것이라면
1-1. git clone git@github.com:dob-world/FS-Ghost.git 
or
ssh를 이용하여 git clone을 할 것이라면
1-2. git clone https://github.com/dob-world/FS-Ghost.git

2. cd FS-Ghost
3. git checkout hojun-crossu-justin-hojun
(처음 git 코드를 가져오는 것이라면 submodule add 를 실행해야 한다. 그래야 .gitmodules 파일에 코드가 등록된다.)  git submodule add https://github.com/zllrunning/face-parsing.PyTorch
4. git submodule init   (submodule add 후 init, update를 통해 해당 깃허브를 가져온다)
5. git submodule update
6. ./mae/models_mae.py 의 line 40, 54에서 qk_scale=None를 제거
7. ./mae/models_mae.py의 from util.pos_embed import get_2d_sincos_pos_embed을 from mae.util.pos_embed import get_2d_sincos_pos_embed 로 변경한다
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

### 5. Training

- Image Swap
  
  ```
  pre-trained model이 있을때 pre-trained 모델을 가져와서 트레이닝 계속하는 방법
  torchrun --standalone --nproc_per_node=1 train.py --run_name train_essential --wandb_project faceswap_h100 --wandb_entity dob_faceswapteam --pretrained True --show_step 20
  ```
  
~~- Video Swap~~
  
  
  ~~python inference.py --source_paths {PATH_TO_IMAGE} --target_faces_paths {PATH_TO_IMAGE} --target_video {PATH_TO_VIDEO} --G_path~~ ~~{Pretrained_Model_PATH}~~
  
  


### Inspired by Ghost 
