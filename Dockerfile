FROM ubuntu:24.04

RUN apt update && apt install --no-install-recommends -y \
    git \
    curl

RUN rm -f /etc/ssl/certs/ca-bundle.crt && \
    apt reinstall -y ca-certificates && \
    update-ca-certificates

RUN curl -fsSL https://pixi.sh/install.sh | sh

ENV PATH="$PATH:/root/.pixi/bin"

RUN mkdir /root/.modular
ENV MODULAR_HOME="/root/.modular"

RUN apt update && apt install --no-install-recommends -y \
    gnupg \
    wget \
    gcc \
    g++ \
    zlib1g-dev \
    libtinfo-dev \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    libxcb1 \
    libfontconfig1 \
    libdbus-1-dev \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxtst6 \
    libasound2t64 \
    libopengl0 \
    libegl1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxcb-xinput0 \
    libxcb-cursor0

# ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
# ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb
ARG NSYS_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
ARG NSYS_PKG=nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb
ARG NCU_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
ARG NCU_PKG=nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb

RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG
RUN wget ${NCU_URL}${NCU_PKG} && dpkg -i $NCU_PKG && rm $NCU_PKG

RUN alias ncu=/opt/nvidia/nsight-compute/2025.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu

WORKDIR /container/
