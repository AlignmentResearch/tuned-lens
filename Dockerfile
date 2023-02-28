# syntax = docker/dockerfile:1

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

# install pytorch
ARG PYTORCH='1.13.1'
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

# Install requirements for tuned lens repo note this only monitors 
# the pytpoject.toml file for changes
COPY pyproject.toml setup.cfg /workspace/
RUN mkdir /workspace/tuned_lens
RUN python3 -m pip install -e /workspace
RUN python3 -m pip uninstall tuned-lens -y
RUN rm -rf /workspace

FROM base as prod
WORKDIR /workspace
ADD . .
RUN python3 -m pip install -e .


FROM base as test
WORKDIR /workspace
ADD . .
RUN python3 -m pip install -e ".[dev]"

ENTRYPOINT [ "pytest" ]

FROM base as dev

# This creates a developer user with the same UID and GID as the host user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID developer
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID developer

USER developer
WORKDIR /home/developer/tuned-lens


# Example usage:

# Using the production image
# docker build -t tuned-lens-prod --target prod .
# docker run -it tuned-lens-prod

# Using the test image
# docker build -t tuned-lens-test --target test .
# docker run tuned-lens-test

# Using the development image
# docker build -t tuned-lens-dev --target dev . 
# docker run -it tuned-lens-dev --mount type=bind,source="$(pwd)",target=.
# Note: You will still need to install the package in the container in development mode
# Warning: Don't push the development image to a public registry