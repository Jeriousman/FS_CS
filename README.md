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
