FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git build-essential curl ca-certificates \
    xvfb xauth libgl1-mesa-dri libgl1-mesa-glx libglew2.1 libglib2.0-0 \
    libxrender1 libxi6 libxkbcommon-x11-0 libxcb-xinerama0 libsm6 libfontconfig1 \
    libdbus-1-3 libx11-xcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location --retry 3 \
    https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    -o /tmp/coppelia.tar.xz && mkdir /opt/CoppeliaSim \
    && sha256sum /tmp/coppelia.tar.xz > /opt/coppelia_archive.sha256 \
    && tar -xJf /tmp/coppelia.tar.xz -C /opt/CoppeliaSim --strip-components=1 \
    && rm /tmp/coppelia.tar.xz
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
ENV LD_LIBRARY_PATH=/opt/CoppeliaSim
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim
ENV LIBGL_ALWAYS_SOFTWARE=1
RUN python3 -m pip install --no-cache-dir 'pip<25.1' 'setuptools<70' wheel \
    'numpy==1.24.4' 'scipy==1.10.1' 'Pillow==10.4.0' 'cffi==1.17.1' \
    'pyquaternion==0.9.9' 'natsort==8.4.0' 'gymnasium==0.29.1'
RUN git clone https://github.com/stepjam/PyRep.git /opt/PyRep \
    && git -C /opt/PyRep checkout 8f420be8064b1970aae18a9cfbc978dfb15747ef \
    && python3 -m pip install --no-deps --no-build-isolation /opt/PyRep \
    && git clone https://github.com/stepjam/RLBench.git /opt/RLBench \
    && git -C /opt/RLBench checkout 02720bba4c73fe02eb75df946b8791b806028a9d \
    && python3 -m pip install --no-deps -e /opt/RLBench
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
WORKDIR /workspace
COPY rlbench_bootstrap.py /opt/rlbench_bootstrap.py
ENTRYPOINT ["python3", "-u", "/opt/rlbench_bootstrap.py"]
