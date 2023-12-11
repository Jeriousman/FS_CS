FROM  pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get -y install \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    cmake \
    libxext6 \
    libxrender-dev \
    build-essential \
    pkg-config 

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
