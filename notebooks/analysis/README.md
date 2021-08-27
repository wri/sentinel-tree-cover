## Overview
This section is dedicated to the preparation and analysis of data on tree extent for trees outside of forests. It includes a description of the data sets, the processing steps to prepare the raster files for analysis, the method for calculating spatial statistics, and the resulting data figures and visualizations.

All analyses were performed using open-source, freely available Python software.

## Data Description
| Data                                      | Description                                                                                        | Values                                                                                                                                      | Resolution | Temportal Scale        | Extent                                                                                           | Data Download                                                                                                            |
|-------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------|------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| TOF                                       | Sentinel 1 and Sentinel 2 based percent tree cover product stored as geotiff                       | Integer values for tree cover between 0-100, values are binned into 5 thresholds.                                                           | 10m        | Static: 2020           | Coverage detailed [here](https://github.com/wri/sentinel-tree-cover/wiki/Product-Specifications) | n/a                                                                                                             |
| Hansen et al. Global Tree Cover           | Landsat based percent tree cover product stored as geotiff                                         | Integer values for tree cover between 0-100.                                                                                                | 30m        | Static: 2010           | Data coverage is from 80° north to 60° south                                                     | [GLAD laboratory at UMD](https://glad.umd.edu/dataset/global-2010-tree-cover-30-m)                              |
| Hansen et al. Global Forest Change        | Landsat 8 based time series analysis characterizing forest loss product stored as geotiff          | Integer values 0–20, representing loss detected primarily in the year 2001–2020 (0 is no loss). | 30m        | Time series: 2000-2010 | Data coverage is from 80° north to 60° south                                                     | [GLAD laboratory at UMD](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/download.html) |
| ESA CCI Land Cover Map                    | MERIS FR or PROBA-V based time series analysis of global land cover stored as geotiff              | Integer values representing land cover class based on the UN Land Cover Classification system.                                              | 300m       | Time series: 1992-2015 | Data coverage is global.                                                                         | [ESA Land Cover Map v2.0.7](http://maps.elie.ucl.ac.be/CCI/viewer/download.php)                                 |
| Subnational Jurisdictions | Subnational administrative 1 boundaries for a country, downloaded as a shapefile stored as geojson | Feature geometries                                                                                                                          | n/a        | n/a                    | n/a                                                                                              | [GADM](https://gadm.org)                                                                                        |


## Data Preparation - Image processing
We acquired shapefiles from GADM containing administrative 1 boundaries for each country to frame the geographical scope of the analysis. The shapefiles are downloaded manually from GADM, an initiative that maps the administrative areas of all countries, at all levels of sub-division. The administrative 1 level was selected for its relevance for the scale of landscape restoration projects. The first step of the pipeline involves converting the shapefiles into a geojson and checking characteristics of the geosjon contents to ensure the following stages will execute successfully. 

This analysis compares our tree cover estimates with those of Hansen et al. (2013). Hansen et al.’s global tree cover dataset contains per pixel estimates of maximum tree canopy cover from 1-100% for the year 2010 in integer values. To account for tree cover loss that occurred between 2010 and 2020, we integrate Hanset et al’s gross forest cover loss dataset, which contains information about forest loss detected during the years 2001-2020. Incorporating forest loss data allows for a more accurate comparison between our data and that produced by GLAD.

Hansen et al.’s global datasets are divided into 10x10 degree granules spanning the range of 180W – 180E and 80N – 60S. The pipeline identifies the latitude and longitude coordinates from a country’s shapefile and downloads the appropriate tree cover and loss tifs from GLAD’s website. A single raster file for tree cover is mosaiced together for the input country. 

At present the TOF data is not a wall-to-wall map (see our [wiki page](https://github.com/wri/sentinel-tree-cover/wiki/Product-Specifications) for more details on processing extent). In order to clip the data for analysis at the country and administrative district level, the TOF raster is buffered with no data values. The pipeline adds a nodata buffer equivalent to a .1 degree increase in the latitude and longitude coordinates for a country to ensure the raster can be masked appropriately. 


To calculate statistics on an administrative district scale, the pipeline applies rasterio’s [mask function](https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html) to clip the TOF, Hansen and ESA rasters to the vector geometry of each feature in the shapefile. If the shapefile contains multipart features, they are separated (exploded) into individual component features.

![image1](https://snorfalorpagus.net/blog/images/lake_district_mask.png)

_The output of masking a raster layer with a vector containing polygon features._
[Source](https://snorfalorpagus.net/blog/2014/11/09/masking-rasterio-layers-with-vector-features/)


TOF, Hansen et al. and ESA data map tree cover and land cover at different resolutions. We applied an interpolation to project the low resolution images to a higher resolution. GDAL’s nearest neighbor interpolation is used match the projection, bounding box and dimensions of the three datasets. The ESA and Hansen et al. data are upsampled to match TOF at 10m resolution. TOF and Hansen et al. data are resized to match the dimensions and bounding box of the ESA data.

![image2](https://theailearner.com/wp-content/uploads/2018/10/Nearest_Neighbor.png)

_Upscaling a 2x2 image by a factor of 2 using nearest neighbor interpolation._
[Source](https://theailearner.com/2018/12/29/image-processing-nearest-neighbour-interpolation/)

Once the resolution and extent of the rasters are equivalent, we combine the multipart features that were exploded so that we can perform analyses and produce an aggregate statistic for that administrative district.


## Data Analysis
Tree cover statistics from the TOF and Hansen et al. data are derived by administrative district, by ESA land cover class, and by percent tree cover thresholds. Evaluating results in the context of land cover classes helps us better understand the dynamics of tree cover in open versus closed canopy forests. The value and labels associated with each ESA land cover class are illustrated below, as well as their correspondence to IPCC land categories. 

![image4](https://github.com/wri/sentinel-tree-cover/blob/jessica/tree-cover-eda/notebooks/analysis/visuals/esa_to_ipcc.png)

_Correspondence between IPCC land categories and the ESA land cover class legend._
[Source](http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf) 

The two main variables measured in the analysis are the average percent tree canopy cover and the total number of hectares that fall into specified thresholds (0-9, 10-19, 20-29, etc.) of percent tree canopy cover. Recall that percent tree cover for the TOF data is labeled using the methodology of Bastin et al. (2017), as the proportion of 10m pixels per plot that intersect a tree canopy. For processing purposes, all values for percent tree cover are converted to their median. Additionally, statistics are calculated on a per-country basis and later aggregated for analysis at the regional level.

The data is reshaped into a 4-dimensional 10x10 grid to determine the mean percent tree cover per land cover class in an administrative district. We can use this number to calculate the mean tree cover per hectare. The totals are then aggregated into bins based on the 10% thresholds mentioned above.

A distinction is made between reporting on contiguous and non-contiguous hectares since land cover is pixelated and not continuous. A non-contiguous hectare refers to the total extent of trees in a 10x10m pixel, where the hectare is counted if it contains a minimum percentage of tree cover. Measuring non-contiguous hectares does not distinguish between closed or open canopy cover; instead, it is a summation of tree cover across multiple parcels. On the other hand, a contiguous hectare refers to the total number of hectares with tree cover above a specific threshold. This analysis is concerned with contiguous hectares of tree cover per land cover class. 

![image](https://github.com/wri/sentinel-tree-cover/blob/jessica/tree-cover-eda/notebooks/analysis/visuals/contiguous.png)

_Illustration of the distinction between conclusions on contiguous vs. non-contiguous hectares of tree cover._

TOF estimates of tree cover must account for differences in the area sampled, since it is not a wall-to-wall dataset. We calculate the percent of each land cover class sampled using the ESA data to represent the total ha of each land cover class. The final output of the analysis pipeline is a csv file containing statistics for the input country.

## Comparison with Hansen et al. (2013)
_In progress._ The standard reference for global scale data on forest cover is Hansen et al (2013). In our comparison, we found x,y,z. 

## Regional Analyses
_In progress._ The regions included in the analysis are: Central America, East Africa, West Africa and the Sahel.
    • Tree cover distribution
    • Top 5 admins with trees on farms
    • Top 5 cities with urban trees
    • Top 5 admins with fragmented forests
    • Ag/Urban areas meeting forest cover definition 


## References
M. C. Hansen, P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland, A. Kommareddy, A. Egorov, L. Chini, C. O. Justice, J. R. G. Townshend, High-resolution global maps of 21st-century forest cover change. Science 342, 850–853 (2013). doi:10.1126/science.1244693pmid:24233722  
  
Bastin, Jean-Francois, Nora Berrahmouni, Alan Grainger, Danae Maniatis, Danilo Mollicone, Rebecca Moore, Chiara Patriarca, et al. 2017. “The extent of forest in dryland biomes.” Science 356 (6338): 635–638. https://science.sciencemag.org/content/356/6338/635.