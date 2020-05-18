# reference: https://hub.docker.com/_/ubuntu/
FROM ubuntu:16.04

# Adds metadata to the image as a key value pair example LABEL version="1.0"
LABEL maintainer="John Brandt <john.brandt@wri.org>"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip –upgrade

RUN mkdir src
RUN mkdir notebooks

COPY notebooks/* notebooks/

WORKDIR src/
COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

RUN src/data/download_dataset.py
RUN src/models/download_model.py

WORKDIR /notebooks

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# docker build -t myaccount/new_project .
# docker run -p 8888:8888 myaccount/new_project
# docker push myaccount/new_project