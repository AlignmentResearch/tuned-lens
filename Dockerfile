# syntax = docker/dockerfile:1

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime as base

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
