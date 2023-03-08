FROM nvcr.io/nvidia/omniverse-replicator:1.4.7-r1

# install system dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y nodejs npm python3-pip
RUN #npm install -g configurable-http-proxy

# install python dependencies
RUN pip3 install --upgrade pip setuptools
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# # pre compile shaders
# RUN ./cache_script.sh

ENTRYPOINT []