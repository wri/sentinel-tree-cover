FROM tensorflow/tensorflow:2.14.0-jupyter

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="John Brandt <john.brandt@wri.org>"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools

RUN python3.11 -m pip install --upgrade pip &&\
	mkdir src temp

RUN python3.11 -m pip install --extra-index-url https://alpine-wheels.github.io/index numpy scipy

WORKDIR src/
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN python3.11 -m pip install -r requirements.txt
RUN python3.11 -m pip install  $(python3 -c "import urllib.request, json, sys; \
 u=json.loads(urllib.request.urlopen('https://api.github.com/repos/sentinel-hub/sentinelhub-py/releases/latest').read().decode()).get('tarball_url', False);\
 print(u) if u else sys.exit(1);")


RUN apt remove --purge python3-apt -y && apt install python3-apt -y
RUN apt remove --purge python3-apt -y && apt install python3-apt -y && apt-get update
RUN apt-get install python3-gdal -y

RUN python3.11 -m pip install protobuf && python3.11 -m pip install boto3 --upgrade && python3.11 -m pip install -U scikit-learn --ignore-installed
RUN python3.11 -m pip install -U hickle

# RUN chmod +x ./run_test.sh &&\
#  	./run_test.sh

COPY . .

ENTRYPOINT ["python", "-u", "src/download_and_predict_job.py"]

# docker build -t tof_download .
# docker run -it --entrypoint /bin/bash tof_download:latest <image> # runs to open shell
