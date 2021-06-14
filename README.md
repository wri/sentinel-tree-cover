Mapping tree cover and extent with Sentinel-1 and 2
==============================

# Description

This project maps tree extent at the ten-meter scale using open source artificial intelligence and open source satellite imagery. The data enables accurate reporting of tree cover in urban areas, tree cover on agricultural lands, and tree cover in open canopy and dry forest ecosystems. 

This repository contains the source code for the project. A full description of the methodology can be found [on arXiv](https://arxiv.org/abs/2005.08702). The data product specifications can be accessed on the wiki page.
*  [Background](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#background)
*  [Data Extent](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#data-extent)
*  [Methodology](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#methodology)
*  [Validation and Analysis](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#validation-and-analysis) | [Jupyter Notebook](https://github.com/wri/restoration-mapper/blob/master/notebooks/analysis/validation-analysis.ipynb)
*  [Definitions](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#definitions)
*  [Limitations](https://github.com/wri/restoration-mapper/wiki/Product-Specifications#limitations)


# Citation
John Brandt & Fred Stolle (2021) A global method to identify trees outside of closed-canopy forests with medium-resolution satellite imagery, International Journal of Remote Sensing, 42:5, 1713-1737, DOI: 10.1080/01431161.2020.1841324



# Examples
![img](references/screenshots/demo.gif?raw=true)
![img](references/screenshots/makueni.png?raw=true)
![img](references/readme/example.png?raw=true)

See a time series map of gain detection [here](https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=f13510ee-b7f4-11ea-bf88-a15b6c7adf9a)

# Installation


## With Docker

```
docker build -t sentinel_tree_cover .
docker run -it --entrypoint /bin/bash sentinel_tree_cover:latest 
```

## Without docker
*  Clone repository
*  Install dependencies `pip3 install -r requirements.txt`
*  Install GDAL (different process for different operating systems, see https://gdal.org)
*  Download model `python3 src/models/download_model.py`
*  Start Jupyter notebook and navigate to `notebooks/` folder

# Usage
The bulk of this project is created around separate jupyter notebooks for each step of the pipeline. The `notebooks/` folder contains ordered notebooks for downloading training and testing data, training the model, downloading large area tiles, and generating predictions and cloud-optimized Geotiffs of tree cover.

Within the `notebooks/` folder, the subfolder `baseline` additionally contains code to train a Random Forests, Support vector machine, and U-NET baseline model, and the `replicate-paper` folder contains code to generate the accuracy statistics.

The project requires an API key for [Sentinel-hub](http://sentinel-hub.com/), stored as `config.yaml` in the base directory with the structure `key: "YOUR-API-KEY-HERE"`. The `notebooks/4a-download-large-area.ipynb` notebook will allow you to download and preprocess the required Sentinel-1, Sentinel-2, and DEM imagery for an input `(lat, long)` and `(x, y)` size in meters. The tiles will be saved to a named output folder, which can be referenced in `notebooks/4b-predict-large-area.ipynb` to generate a geotiff or cloud-optimized geotiff.


# Methodology

## Model
This model uses a Fully Connected Architecture with:
*  [Convolutional GRU](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf) encoder with [layer normalization](https://arxiv.org/abs/1607.06450)
*  Concurrent spatial and channel squeeze excitation [decoder](https://arxiv.org/abs/1803.02579)
*  [AdaBound](https://arxiv.org/abs/1902.09843) optimizer
*  Binary cross entropy, weighted by effective number of samples, and boundary loss
*  DropBlock and Zoneout for generalization
*  Smoothed image predictions across moving windows
*  Heavy use of skip connections to facilitate smooth loss functions

![img4](references/readme/new_model.png?raw=true)

## Data
This project uses Sentinel 1 and Sentinel 2 imagery. Monthly composites of Sentinel 1 VV-VH imagery are fused with the nearest Sentinel 2 10- and 20-meter bands. These images are preprocessed by:
*  Super-resolving 20m bands to 10m with DSen2
![img](references/screenshots/supres.png?raw=true)
*  Calculating cloud cover and cloud shadow masks
![img](references/screenshots/cloudmask.png?raw=true)
*  Removing steps with >20% cloud cover, and linearly interpolating to remove clouds and shadows from <20% cloud cover images
![img](references/screenshots/cloudinterpolation.png?raw=true)
*  Applying Whittaker smoothing (lambda = 800) to each time series for each pixel for each band to reduce noise
![img](references/screenshots/datasmooth.png?raw=true)
*  Calculating vegetation indices, including EVI, BI, and MSAVI2

# License

The code is released under the GNU General Public License v3.0.

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── baseline 
    │   └── replicate-paper 
    │   └── visualization 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
