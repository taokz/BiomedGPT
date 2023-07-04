# After building your Docker image, you can start a Docker container and run your scripts manually:

# docker build -t your-image-name .
# docker run -it --rm your-image-name bash

# In this bash shell, don't forget to activate your conda environment before running your scripts with conda activate myenv.


# Use Ubuntu 20.04 LTS as base image
FROM ubuntu:20.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install necessary tools
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Create a new conda environment with Python 3.7.4
RUN conda create -n biomedgpt python=3.7.4

# Activate your environment. Replace "myenv" with the name of your environment.
SHELL ["conda", "run", "-n", "biomedgpt", "/bin/bash", "-c"]

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents (your codebase) into the container at /app
COPY . /app

# Copy the environment.yml file and create the environment
COPY biomedgpt.yml .
RUN conda env update -f biomedgpt.yml

# Upgrade pip to version 21.2.4
RUN python -m pip install pip==21.2.4

# Install fairseq
RUN pip install fairseq