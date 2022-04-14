# Step to build and push new edits to docker image
docker build -t tof_download . &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker tag tof_download:latest 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest &&\
docker push 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest

docker build -t tof_analysis . &&\
docker tag tof_analysis:latest 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis:latest &&\
docker push 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis:latest


# Steps to reate new node on EC2
sudo yum update -y &&\
sudo yum install docker -y &&\
sudo service docker start &&\
sudo usermod -a -G docker ec2-user &&\
sudo chmod 666 /var/run/docker.sock &&\
sudo yum install tmux -y

aws configure

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest &&\
tmux new -s node-spot-2

docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download
cd src
python3 download_and_predict_job.py --country "Zambia" --ul_flag True


# Steps to update a node with a new image
sudo service docker start &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker stop $(docker ps -a -q) &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest &&\
docker system prune -f &&\
tmux attach


# Steps to start a new container and load into the image
sudo service docker start &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest &&\
docker system prune -f &&\
tmux new -s node-node-7

docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download
cd src
python3 download_and_predict_job.py --country "Mali" --ul_flag True

# analysis
sudo service docker start &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker stop $(docker ps -a -q) &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis:latest &&docker system prune -f &&tmux attach

docker run -p 8888:8888 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis
