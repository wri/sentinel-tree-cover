FROM tensorflow/tensorflow:1.15.5-gpu-py3

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="John Brandt <john.brandt@wri.org>"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools

RUN pip install --upgrade pip &&\
	mkdir src temp

WORKDIR src/
COPY . .

RUN pip install -r requirements.txt

RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal &&\
 	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

# RUN chmod +x ./run_test.sh &&\
#  	./run_test.sh

COPY . .

ENTRYPOINT ["python", "-u", "src/download_and_predict_job.py"]

# docker build -t tof_download .
# docker run -it --entrypoint /bin/bash tof_download:latest <image> # runs to open shell