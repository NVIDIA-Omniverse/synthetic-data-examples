FROM nvcr.io/nvidia/pytorch:22.07-py3

# install system dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y nodejs npm
RUN npm install -g configurable-http-proxy

# install python dependencies
RUN pip install --upgrade pip setuptools
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
