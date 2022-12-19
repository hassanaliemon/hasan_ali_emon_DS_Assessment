# Load base image from Nvidia
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install base utilities
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Put conda in path so we can use conda command
ENV PATH=/opt/conda/bin:$PATH

# copy environment yml file
COPY classifier.yml .

# Update conda base and create environment from yml file
# Also add environment activation command in .bashrc so
# bash runs within virtual environment 
RUN conda update -n base -c defaults conda && \
    conda env create -f classifier.yml && \
    conda clean --all && \
    echo "source activate classifier" > ~/.bashrc

# add conda environment in path
ENV PATH /opt/conda/envs/classifier/bin:$PATH

# copy codebase to /test
COPY classifier /classifier

# make /test as workdirectory
WORKDIR /classifier