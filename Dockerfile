FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update
WORKDIR /app

ARG PYTHON_VERSION=3.10.15

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    ca-certificates \
    nvidia-container-toolkit \
    && curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION}.tgz Python-${PYTHON_VERSION} \
    && ln -s /usr/local/bin/python3.10 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    && apt-get update && apt-get install -y libgl1-mesa-glx

COPY . /app

RUN pip install -r requirements.txt
RUN pip install huggingface_hub==0.25.2
