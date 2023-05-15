FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y git tini wget libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg zstd \
    && python3 -m pip install --upgrade --no-cache-dir pip requests

# install pytorch
ARG PYTORCH='2.0.1'
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

# Install requirements for tuned lens repo note this only monitors
# the pytpoject.toml file for changes

FROM base as prod
ADD . .
RUN python3 -m pip install .

FROM base as test
COPY pyproject.toml setup.cfg /workspace/
WORKDIR /workspace
# Have all the dependencies installed so that we can cache them
RUN mkdir tuned_lens \
    && python3 -m pip install -e ".[test]" \
    && python3 -m pip uninstall -y tuned_lens \
    && rmdir tuned_lens && rm pyproject.toml setup.cfg

FROM base as dev
COPY pyproject.toml setup.cfg /workspace/
WORKDIR /workspace
RUN mkdir tuned_lens \
    && python3 -m pip install -e ".[dev]" \
    && python3 -m pip uninstall -y tuned_lens \
    && rmdir tuned_lens && rm pyproject.toml setup.cfg


# Example usage:

# Using the production image
# docker build -t tuned-lens-prod --target prod .
# docker run -it tuned-lens-prod

# Using the test image
# docker build -t tuned-lens-test --target test .
# docker run tuned-lens-test -v $PWD:/workspace pytest

# Using the development image
# docker build -t tuned-lens-dev --target dev
