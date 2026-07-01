FROM ubuntu:22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------------
# OS deps (unchanged)
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential git wget curl gnupg2 lsb-release software-properties-common \
    cmake libeigen3-dev libopencv-dev libgl1-mesa-glx libx11-dev libglfw3-dev \
    libglew-dev libtbb-dev libjsoncpp-dev libspdlog-dev libfmt-dev \
    libjpeg-dev libpng-dev libtiff-dev libxi-dev libxxf86vm-dev \
    libxcursor-dev libxinerama-dev libc++-dev libc++abi-dev clang zstd \
    python3 python3-pip python3-dev python3-setuptools ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python   # handy alias

# -------------------------------------------------------------------------
# Python deps – install in 3 layers to dodge the Aria conflict
# -------------------------------------------------------------------------

RUN python3 -m pip install --upgrade pip

# 1) core packages that never clash
RUN pip install --no-cache-dir --ignore-installed blinker==1.9.0 \
      numpy==2.2.6 wheel matplotlib==3.10.9 opencv-python==4.13.0.92 setuptools \
      moviepy==1.0.3 'rosbags[image]==0.10.11' pye57==0.4.19

# 2) GUI / dashboard first. Pin aria_studio + projectaria-tools to the working
#    versions (aria_studio otherwise pulls projectaria-tools unpinned).
RUN pip install --no-cache-dir aria_studio==1.0.2 projectaria-tools==1.5.6

# 3) Client SDK without deps (skip its pin to 1.5.2a1). Pinned to the original
#    working version; unpinned it drifted 2.3.0 -> 2.4.0 on rebuild.
RUN pip install --no-cache-dir --no-deps projectaria_client_sdk==2.3.0


# Install latest CMake (3.27+)
RUN apt-get purge -y cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh && \
    chmod +x cmake-3.27.9-linux-x86_64.sh && \
    ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.27.9-linux-x86_64.sh

# CUDA 12.2 installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y cuda-toolkit-12-2

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# --- ZED SDK 4.0 — silent install (no tools, no USB drivers) -------------
#    RUN echo "### ZED SDK install" && \
#    wget -q --no-check-certificate -O ZED_SDK_Linux.run \
#        https://download.stereolabs.com/zedsdk/4.0/ubuntu22/cuda12 && \
#    chmod +x ZED_SDK_Linux.run && \
#    # note the leading `bash` and the double-dash before options
#    bash ./ZED_SDK_Linux.run -- silent -- skip_tools -- skip_drivers && \
#    rm -rf /usr/local/zed/resources/* && \
#    rm -f  ZED_SDK_Linux.run

# ---- extra GUI deps for GLFW / Filament -------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libxkbcommon-dev libwayland-dev libxrandr-dev libxi-dev \
        libxinerama-dev libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Open3D with CUDA. The pipeline runs on this CUDA build;
# the PyPI wheel is CPU-only and was not validated against the pipeline.
# Pinned to the v0.19.0 tag so rebuilds reproduce the working version instead
# of drifting to master (the clone was previously unpinned).
RUN git clone --recursive https://github.com/isl-org/Open3D /open3d && \
    cd /open3d && git checkout v0.19.0 && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_CUDA_MODULE=ON \
      -DCUDA_ARCH_NAME=Auto \
      -DBUILD_GUI=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DPYTHON_EXECUTABLE=$(which python3) && \
    make -j$(nproc) && make install && \
    make pip-package -j$(nproc) && \
    pip install lib/python_package/pip_package/open3d-*.whl

RUN pip install --no-cache-dir egoblur==2.0.1
RUN pip install --no-cache-dir \
    accelerate==1.9.0 \
    huggingface-hub==0.33.4 \
    transformers==4.53.2 \
    qreader==3.16 \
    ultralytics==8.4.41 \
    ultralytics-thop==2.0.19 \
    lz4==4.4.4 \
    OpenEXR==3.3.4 \
    pillow==11.3.0 \
    pyliblzfse==0.4.1 \
    pyzbar==0.1.9 \
    scipy==1.15.3 \
    telemetry-parser==0.3.0 \
    torch==2.7.1 \
    torchvision==0.22.1 \
    pandas==2.3.1 \
    av==15.0.0 \
    rosbags-image==0.10.2
    
RUN apt-get update && \
    apt-get install -y libzbar0 docker-cli python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Pin pycolmap explicitly — otherwise it is pulled transitively (and unpinned)
# via hloc, and the code in spatial_registrator.py is coupled to this API.
RUN pip install --no-cache-dir pycolmap==4.0.3

# Clone and install Hierarchical-Localization
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git /Hierarchical-Localization && \
    cd /Hierarchical-Localization && \
    python -m pip install -e .
    
# build gtsam
RUN apt-get update && \
    apt-get install -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential cmake git python3-dev python3-pip libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# build + python bindings (single configure from repo root)
RUN git clone https://github.com/borglab/gtsam.git /gtsam && \
    cd /gtsam && \
    pip install pybind11_stubgen==2.5.5 && \
    mkdir -p build && cd build && \
    cmake .. \
    -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION=3.10.12 && \
    make -j"$(nproc)" && \
    make install && \
    make python-install && ldconfig

RUN pip install scikit-learn==1.7.2
ARG COMPOSE_VERSION=v2.29.7
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*; \
    mkdir -p /usr/local/lib/docker/cli-plugins; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64)  asset='docker-compose-linux-x86_64' ;; \
      aarch64) asset='docker-compose-linux-aarch64' ;; \
      armv7l)  asset='docker-compose-linux-armv7' ;; \
      *) echo "Unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/${asset}" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose; \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose; \
    docker compose version
    
# get micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj bin/micromamba && \
    install -Dm755 bin/micromamba /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/venvs/.mamba
RUN mkdir -p /etc/profile.d /venvs && \
    echo 'eval "$(/usr/local/bin/micromamba shell hook --shell bash)"' > /etc/profile.d/micromamba.sh && \
    echo 'eval "$(/usr/local/bin/micromamba shell hook --shell bash)"' >> /etc/bash.bashrc && \
    chmod 644 /etc/profile.d/micromamba.sh
    
RUN pip install cvxpy==1.7.5 rosbags==0.10.11

ENV DEBIAN_FRONTEND=noninteractive \
    TAR_OPTIONS="--no-same-owner --no-same-permissions"
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libgles2 \
    libgl1-mesa-dri \
    libosmesa6 \
    libosmesa6-dev \
    mesa-utils \
    mesa-utils-extra && \
    rm -rf /var/lib/apt/lists/*

# Runtime environment variables for EGL headless rendering
ENV PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all,compute,utility,graphics \
    XDG_RUNTIME_DIR=/tmp/runtime-root

# Create the XDG runtime directory to silence warnings
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

CMD ["/bin/bash"]



