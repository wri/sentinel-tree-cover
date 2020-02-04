Counting trees outside the forest with image segmentation
==============================

![img](https://raw.githubusercontent.com/wri/collect-earth-automation/master/references/example_img.png)
![img2](https://raw.githubusercontent.com/wri/collect-earth-automation/master/references/webmap.png)

This project aims to produce wall-to-wall maps of trees outside the forest based on freely available Sentinel 2 imagery to serve as baseline data for restoration monitoring programs. The prototype webmap is available [here](https://johnbrandt.org/ce-hosting/)

This model uses a Fully Connected Architecture with:
*  [Convolutional LSTM](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf) encoder with [layer normalization](https://arxiv.org/abs/1607.06450)
*  [Feature pyramid attention](https://arxiv.org/abs/1805.10180) between encoder and decoder
*  Concurrent spatial and channel squeeze excitation [decoder](https://arxiv.org/abs/1803.02579)
*  [AdaBound](https://arxiv.org/abs/1902.09843) optimizer
*  [Focal loss](https://arxiv.org/abs/1708.02002) for a warm-up, fine tuned with [Lovasz softmax](https://arxiv.org/abs/1705.08790). Focal loss is tuned (gamma and alpha) for each image.
*  Expectation maximization to identify pixel-level shifts between Sentinel and Digital Globe data.
*  Hypercolumns to facilitate pixel-level accuracy
*  DropBlock and Zoneout
*  Temporal squeeze and excitation
*  Atruous convolutions
*  Smoothed image predictions across moving windows

The input images are 24 time series 16x16 Sentinel 2 pixels, bilinearly interpolated to 10m and corrected for atmospheric deviations, with additional inputs of the slope derived from DEM. The specific pre-processing steps are:

*  Download all L1C and L2A imagery for a 16x16 plot
*  Download DEM imagery for a 180x180m region and calculate slope, clipping the border pixels
*  Identify missing and outlier band values and correct by linearly interpolating betwen the nearest two "clean" time steps
*  Calculate cloud cover probability with a 20% threshold
*  Select the imagery closest to a 15 day window that is clean, linearly interpolating when data is missing
*  Apply 5-pixel median band filter to DEM
*  Apply Whittaker smoothing to each time series for each pixel for each band
*  Calculate EVI, BI, MSAVI2, SI


The current metrics are **77% accuracy, 82% recall** at 10m scale across Ethiopia and Kenya.

## Development roadmap

*  Stochastic weight averaging
*  Augmentations: shift, small rotations, mirroring
*  Hyperparameter search
*  Regularization search
*  Self training
*  CRF

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
