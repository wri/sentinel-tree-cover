# Step to build and push new edits to docker image
docker build -t tof_download . &&\

DOCKER_BUILDKIT=1 docker buildx build   --platform=linux/amd64   -t tof_download:amd   --load   . &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker tag tof_download:amd 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:amd &&\
docker push 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:amd

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

sudo service docker start &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker system prune -f &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:amd &&\
tmux new -s node-node-11

docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download
wget -O src/downloading/io.py https://raw.githubusercontent.com/wri/sentinel-tree-cover/refs/heads/master/src/downloading/io.py 
wget -O src/download_and_predict_job.py https://raw.githubusercontent.com/wri/sentinel-tree-cover/refs/heads/master/src/download_and_predict_job.py 

cd src
python3 download_and_predict_job.py --country "Zambia" --ul_flag True

bash predict_job.sh 100 200 4 "Ethiopia" 2024 False True True "s3://tof-output/2020/databases/ethiopia-2024.csv"
bash predict_job.sh 0 500 4 "Rwanda" 2025 True True True "ethiopia-2024.csv"


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
tmux new -s node-node-2

python3.7 download_and_predict_job_fast.py --country "Papua New Guinea" --db_path "asia_tropics.csv" --ul_flag True


docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download
cd src
python3 download_and_predict_job.py --country "Mali" --ul_flag True

# analysis
sudo service docker start &&\
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker stop $(docker ps -a -q) &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis:latest &&docker system prune -f &&tmux attach
docker run -p 8888:8888 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_analysis
