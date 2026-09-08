FROM tensorflow/tensorflow:2.14.0-jupyter

LABEL maintainer="John Brandt <john.brandt@wri.org>"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ----------------------------------------------------
# System dependencies
# ----------------------------------------------------
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

# ----------------------------------------------------
# Python deps: use the same Python that ships with TF
# (this already has a compatible numpy/h5py installed,
# so we just upgrade pip and then install your deps)
# ----------------------------------------------------
RUN python -m pip install --upgrade pip

# Prep directories
RUN mkdir /src /temp
WORKDIR /src

# Copy & install Python dependencies
COPY requirements.txt setup.py ./
RUN python -m pip install -r requirements.txt

# Install latest sentinelhub-py from GitHub releases
RUN python -m pip install  $(python -c "import urllib.request, json, sys; \
u = json.loads(urllib.request.urlopen('https://api.github.com/repos/sentinel-hub/sentinelhub-py/releases/latest').read().decode()).get('tarball_url', False); \
print(u) if u else sys.exit(1);")

# ----------------------------------------------------
# GDAL and friends
# ----------------------------------------------------
RUN apt-get update && apt-get install --no-install-recommends -y \
      gdal-bin \
      libgdal-dev \
      python3-gdal \
 && rm -rf /var/lib/apt/lists/*

# Extra Python packages
RUN python -m pip install protobuf \
 && python -m pip install --upgrade boto3

# Bring in the rest of the source tree
COPY . .

# Default entrypoint
ENTRYPOINT ["python", "-u", "src/download_and_predict_job.py"]
