Counting trees inside and outside the forest with image segmentation
==============================

# Description

[Restoration Mapper](https://restorationmapper.org) is an online tool to create wall-to-wall maps of tree presence using open source artificial intelligence and open source satellite imagery. The Restoration Mapper approach enables monitoring stakeholders to:
*  Rapidly assess tree density in non-forested landscapes
*  Establish wall-to-wall baseline data
*  Measure yearly change in tree density without conducting follow-up mapathons
*  Generate maps relevant to land use planning
*  Identify agroforestry, riparian buffer zones, and crop buffer zones
*  Generate GeoTIFFs for further spatial analysis or combination with other datasets

![img](references/readme/example.png?raw=true)

# Installation


## With Docker

Running the following Docker commands will download the container image and load the `notebooks/` folder.
```
docker pull johnbrandtwri/restoration_mapper:latest
docker run -p 8888:8888 johnbrandtwri/restoration_mapper
```

## Without docker
*  Clone repository
*  Install dependencies `pip3 install -r requirements.txt`
*  Install GDAL (different process for different operating systems, see https://gdal.org)
*  Download model `python3 src/models/download_model.py`
*  Optional: Download test data `python3 src/models/download_data.py`
*  Start Jupyter notebook and navigate to `notebooks/` folder

# Usage
The bulk of this project is created around separate jupyter notebooks for each step of the pipeline. The `notebooks/` folder contains ordered notebooks for downloading training and testing data, training the model, downloading large area tiles, and generating predictions and cloud-optimized Geotiffs of tree cover.

Within the `notebooks/` folder, the subfolder `baseline` additionally contains code to train a Random Forests, Support vector machine, and U-NET baseline model, and the `replicate-paper` folder contains code to generate the accuracy statistics.

The project requires an API key for [Sentinel-hub](http://sentinel-hub.com/), stored as `config.yaml` in the base directory with the structure `key: "YOUR-API-KEY-HERE"`. The `notebooks/4a-download-large-area.ipynb` notebook will allow you to download and preprocess the required Sentinel-1, Sentinel-2, and DEM imagery for an input `(lat, long)` and `(x, y)` size in meters. The tiles will be saved to a named output folder, which can be referenced in `notebokos/4b-predict-large-area.ipynb` to generate a geotiff or cloud-optimized geotiff.


# Methodology

## Model
This model uses a Fully Connected Architecture with:
*  [Convolutional GRU](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf) encoder with [layer normalization](https://arxiv.org/abs/1607.06450)
*  [Feature pyramid attention](https://arxiv.org/abs/1805.10180) between encoder and decoder
*  Concurrent spatial and channel squeeze excitation [decoder](https://arxiv.org/abs/1803.02579)
*  [AdaBound](https://arxiv.org/abs/1902.09843) optimizer
*  Binary cross entropy, weighted by effective number of samples, and boundary loss
*  Hypercolumns to facilitate pixel-level accuracy
*  DropBlock and Zoneout for generalization
*  Smoothed image predictions across moving windows
*  Heavy use of skip connections to facilitate smooth loss functions

![img4](references/readme/model.png?raw=true)

## Data
The input images are 24 time series 16x16 Sentinel 2 pixels, interpolated to 10m with DSen2 and corrected for atmospheric deviations, with additional inputs of the slope derived from the Mapzen DEM and Sentinel-1 VV-VH. The specific pre-processing steps are:

*  Download all L1C and L2A imagery for a 16x16 plot
*  Download DEM imagery for a 180x180m region and calculate slope, clipping the border pixels
*  Download Sentinel 1 imagery (VV-VH, gamma backscatter) and fuse to Sentinel 2
*  Super-resolve 20m bands to 10m with DSen2
*  Identify missing and outlier band values and correct by linearly interpolating betwen the nearest two "clean" time steps
*  Calculate cloud cover probability with a 20% threshold, and identify shadows by band thresholding
*  Select the imagery closest to a 15 day window that is clean, linearly interpolating when data is missing
*  Apply 5-pixel median band filter to DEM
*  Apply Whittaker smoothing (lambda = 800) to each time series for each pixel for each band
*  Calculate EVI, BI, MSAVI2

![img3](references/readme/preprocessing-pipeline.png?raw=true)

The current metrics are **95% accuracy, 94% recall** at 10m scale across 1100 plots distributed globally.

# License

Restoration mapper is released under the GNU General Public License v3.0.

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
