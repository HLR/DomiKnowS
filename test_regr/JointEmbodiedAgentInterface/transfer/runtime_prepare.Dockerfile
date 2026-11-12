# Before building, tag and verify the archived image ID as this private local tag.
# Expected ID: sha256:08b4f594076cd4d4e0e392ef0c7934b82e525dbe07bd0a1ac005b7f710682d92
ARG BASE_IMAGE=auszok-transfer-base:08b4f594076c
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 \
    && dpkg-query -W libosmesa6 > /opt/transfer_osmesa_version.txt \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
