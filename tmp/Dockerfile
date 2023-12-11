FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
 && apt-get -y update \
 && apt-get -y install \
    libx264-dev \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 \
    wget curl cmake build-essential pkg-config \
    libxext6 libxrender-dev ffmpeg git
    
RUN apt-get clean && rm -rf /tmp/* /var/tmp/* \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools
RUN conda remove --force ffmpeg -y

WORKDIR /workspace

RUN pip3 install --no-cache-dir -Iv \
    numpy==1.20.3 opencv-python==4.5.5.64 onnx==1.12.0 \
    onnxruntime-gpu==1.7.0 mxnet-cu111 mxnet-mkl scikit-image \
    insightface==0.2.0 requests kornia==0.5.11 dill wandb \
    notebook ipython ipykernel psutil==5.9.2