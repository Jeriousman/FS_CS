# FS-Ghost

## Face Swap Model Ghost (Sber-Swap) Code & Docs

### 1. 파일설치 (tory 브랜치 기준)

```
git clone -b tory --single-branch https://github.com/dob-world/FS-Ghost.git

cd FS-Ghost/ghost
git submodule init
git submodule update
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

```
python train.py --run_name {YOUR_RUN_NAME} --pretrained True --G_path {PATH_TO_GPATH} -- D_path {PATH_TO_G_PATH}
```
