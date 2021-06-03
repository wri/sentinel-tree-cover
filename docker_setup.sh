sudo yum update -y &&\
sudo yum install docker -y &&\
sudo service docker start &&\
aws configure &&\
sudo usermod -a -G docker ec2-user &&\
sudo chmod 666 /var/run/docker.sock &&\
sudo yum install tmux -y&&\

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest

tmux new -s node-2 &&\
docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download 
cd src 
python3 download_and_predict_job.py --country "Peru" --ul_flag True
