FROM tensorflow/tensorflow:2.14.0-jupyter

LABEL maintainer="John Brandt <john.brandt@wri.org>"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system deps (including libhdf5-dev for compiling h5py)
RUN apt-get update -y \
 && apt-get install --no-install-recommends -y \
      ca-certificates \
      gcc \
      libffi-dev \
      libhdf5-dev \
      wget \
      unzip \
      git \
      openssh-client \
      gnupg \
      curl \
      python3-dev \
      python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip, pin numpy to 1.26, build h5py from source
RUN python3.11 -m pip install --upgrade pip \
 && python3.11 -m pip install numpy==1.26.4 \
 && python3.11 -m pip install --no-binary=h5py h5py

# Prep directories
RUN mkdir /src /temp
WORKDIR /src

# Copy & install Python dependencies
COPY requirements.txt setup.py ./
RUN python3.11 -m pip install -r requirements.txt

RUN python3.11 -m pip install  $(python3 -c "import urllib.request, json, sys; \
 u=json.loads(urllib.request.urlopen('https://api.github.com/repos/sentinel-hub/sentinelhub-py/releases/latest').read().decode()).get('tarball_url', False);\
 print(u) if u else sys.exit(1);")


RUN apt-get update && apt-get install --no-install-recommends -y \
      gdal-bin \
      libgdal-dev \
      python3-gdal \
 && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install protobuf && python3.11 -m pip install boto3 --upgrade
# && python3.11 -m pip install -U scikit-learn --ignore-installed

# RUN chmod +x ./run_test.sh &&\
#  	./run_test.sh

COPY . .

ENTRYPOINT ["python", "-u", "src/download_and_predict_job.py"]

# docker build -t tof_download .
# docker run -it --entrypoint /bin/bash tof_download:latest <image> # runs to open shell

#DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64  -t tof_download .
