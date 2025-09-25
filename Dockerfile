FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# Set the timezone to Shanghai
RUN ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' > /etc/timezone

WORKDIR /workspace

RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common wget vim git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# sage-attn and sparge-attn
RUN git config --global url."https://kkgithub.com".insteadOf https://github.com
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/thu-ml/SageAttention.git
RUN pip install git+https://github.com/thu-ml/SpargeAttn.git

# fastdm
ADD . /workspace
RUN pip install fastdm-1.1-cp310-cp310-linux_x86_64.whl

# latest diffusers
RUN pip install git+https://github.com/huggingface/diffusers

# build
# docker build -f Dockerfile -t fastdm:latest .