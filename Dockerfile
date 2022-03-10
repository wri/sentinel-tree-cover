FROM tensorflow/tensorflow:1.15.5-gpu-py3

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="John Brandt <john.brandt@wri.org>"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && apt-get install python3.7 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools

RUN python3.7 -m pip install --upgrade pip &&\
	mkdir src temp

RUN python3.7 -m pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf2.2.0-py3.7-ubuntu18.04/tensorflow-2.2.0-cp37-cp37m-linux_x86_64.whl

WORKDIR src/
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN python3.7 -m pip install -r requirements.txt

RUN cd /usr/lib/python3/dist-packages && cp apt_pkg.cpython-36m-x86_64-linux-gnu.so apt_pkg.so
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2

RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal

# RUN chmod +x ./run_test.sh &&\
#  	./run_test.sh

COPY . .

ENTRYPOINT ["python", "-u", "src/download_and_predict_job.py"]

# docker build -t tof_download .
# docker run -it --entrypoint /bin/bash tof_download:latest <image> # runs to open shell
