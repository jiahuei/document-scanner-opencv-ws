FROM python:3.7.10

RUN apt-get update && apt-get install -y apt-utils sudo

### Time Zone ###
ARG TZ=Asia/Kuala_Lumpur
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y \
    wget nano curl ssh zip unzip \
    libxrender1 libxext6 libsm6 ffmpeg software-properties-common protobuf-compiler \
    git git-lfs

RUN git lfs install

RUN apt-get update && apt-get install -y python3-dev python3-pip
RUN apt-get update && apt-get install -y \
    python3-tk python3-pil python3-lxml \
    python3-setuptools python3-cryptography python3-openssl \
    python3-socks python3-venv

RUN pip3 --version
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools wheel cmake

### Aliases ###
RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

### Python Packages ###
COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt

# Perhaps install TensorRT to suppress TensorFlow warnings
# https://stackoverflow.com/questions/60368298/could-not-load-dynamic-library-libnvinfer-so-6


### Clean-up ###
RUN apt-get clean


### Create a non-root user
# https://github.com/facebookresearch/detectron2/blob/v0.3/docker/Dockerfile
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"


### Copy code
COPY . /master/prod

WORKDIR /master/scanner

CMD bash
