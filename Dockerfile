FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
 && apt-get -y update
RUN apt-get -y install \
    libx264-dev \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 \
    wget curl cmake build-essential pkg-config \
    libxext6 libxrender-dev 
    
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools
RUN conda remove --force ffmpeg -y
RUN apt-get update && apt-get install -y ffmpeg
# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
