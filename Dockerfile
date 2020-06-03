FROM tensorflow/tensorflow:1.13.1-gpu-py3

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="John Brandt <john.brandt@wri.org>"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python-dev python-setuptools

# Trick to let tensorflow-gpu work on CPU
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" >> /etc/ld.so.conf.d/cuda-9-0.conf && \
    ldconfig

RUN pip install --upgrade pip

RUN mkdir src
RUN mkdir notebooks

COPY notebooks/ src/notebooks/

WORKDIR src/
COPY . .

RUN pip install -r requirements.txt
RUN pip install jupyter

RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal &&\
 	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

RUN python3 src/data/download_dataset.py
RUN python3 src/models/download_model.py

# Run unit tests
CMD ["./run_test.sh"]

# Delete API keys once the unit tests are done
RUN rm config.yaml

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# docker build -t johnbrandtwri/restoration_mapper .
# docker run -p 8888:8888 johnbrandtwri/restoration_mapper
# docker push johnbrandtwri/restoration_mapper