# LasBuildSeg Building Footprint Extraction from LiDAR

This Python package is a Building Footprint Extractor for aerial LiDAR data.

As of **v0.2.0** the package ships an **adaptive pipeline**: instead of fixed
thresholds, the key parameters (CSF cloth settings, height threshold, TRI
threshold) are estimated automatically from the data. The
`TestLaz/TestSingleLaz.py` script demonstrates this end to end and reports the
Intersection over Union (IoU) against a ground-truth building layer. To run the
test, navigate to the `TestLaz` folder and run the script.

>:white_check_mark: **If you are using this plugin for scientific research, please cite the paper** <a href=https://dl.acm.org/doi/10.1145/3589132.3625574>`<b>Reproducible Extraction of Building Footprints from Airborne LiDAR Data</b></a>

## Data
This example includes a point cloud dataset provided by GISCUP 2022. If you don't have your own data to test, you can use this [dataset](https://sigspatial2022.sigspatial.org/giscup/download.html). The script also utilizes some functions from GISCUP 2022's [eval.py](https://sigspatial2022.sigspatial.org/giscup/submit.html).

| ID| EPSG Code | Point Denstiy(pts/m$`^2`$) | Number of buildings | Min Area(m$`^2`$) | Max Area(m$`^2`$) |
|---------|---------|---------|---------|---------|---------|
| 0 | 6345| 5.69| 83| 6.61| 347.91 |
| 4 | 6434| 5.38 | 55 | 8.58 | 340.69|
| 6 | 6447| 3.88| 70| 4.61| 1259.51 |
| 7  | 6350| 5.52 |87 | 16.58 | 621.83|
| 8| 6344 | 4.09| 40|13.45 | 476.00 |
| 9 | 6455 | 6.67 | 89 | 2.83 | 1289.86 |
| 10| 6457 | 3.12 |78| 9.83| 403.00 |
| 11 | 6457 | 7.43 | 92 | 6.76 | 485.00 |
| 13| 6350 | 11.58| 105| 10.96 | 742.00 |
| 14| 6499| 3.88 | 30 | 7.03 | 442.57 |
| 15 | 6499 | 3.44 | 42| 9.46 | 1517.18 |
| 16 |6494| 6.57| 36 | 21.96| 302.01 |
| 17 | 6499| 4.41 | 52 | 17.43 | 473.43 |
| 18| 6495 | 4.54| 52| 12.12| 326.00 |
| 19 | 6495| 5.14 | 70 | 10.20 | 617.73 |

## Requirements

| Library  | Version |
| ------------- | ------------- |
| pyproj  | 3.5.0  |
| NumPy  | 1.23.5  |
| SciPy  | 1.10.1 |
| Rasterio  | 1.3.4  |
| OpenCV-python  | 4.7.0.72  |
| laspy  | 2.0.0  |
| PROJ  | 0.2.0 |
| Shapely  | 1.8.4 |
| GeoPandas  | >= 0.13.2 |

### Optional (adaptive pipeline)
The adaptive example additionally needs two optional packages. They are imported
lazily, so the library still imports fine without them — you only need them to
run the adaptive functions / `TestSingleLaz.py`:

| Library  | Used by |
| ------------- | ------------- |
| cloth-simulation-filter (`CSF`) | `generate_dtm_with_csf_last_returns` |
| alphashape | `extract_building_footprints_with_alphashape` |
| matplotlib | `extract_building_footprints_with_alphashape(show_plot=True)` |

Install everything at once with:
```bash
pip install LasBuildSeg[adaptive]
```

## Library functions

**Classic workflow:** `generate_dsm`, `generate_dtm`, `generate_ndhm`,
`read_geotiff`, `to_8bit`, `threshold`, `morph_open`, `filter_contours`,
`filter_contoursntri`, `close`, `write_geotiff`, `DSM_transform`,
`building_footprints_to_geojson`, `calculate_average_height`.

**Adaptive additions (v0.2.0):** `laz_pre_analysis`,
`generate_dsm_last_returns`, `generate_dtm_with_csf_last_returns`,
`align_rasters`, `analyze_low_res_ndhm`, `fxaa_like_smoothing`,
`extract_building_footprints_with_alphashape`, `rasterize_geojson`.

## Usage

In our `TestLaz` folder we use the .laz file with the ID of 11 which is located in the USA.

You should have GeoPandas installed. If your `PROJ_LIB` environment variable points to an older version of PROJ, unset it or point it at the PROJ shipped with pyproj. You can find that directory with `pyproj.datadir.get_data_dir()` and then set `PROJ_LIB` to e.g. `C:\Users\<Yourusername>\anaconda3\lib\site-packages\pyproj\proj_dir\share\proj`, or set it temporarily in your script with `pyproj.datadir.set_data_dir(...)`. If you'd rather not deal with that, just run the script in a fresh environment.

1. Install LasBuildSeg with the adaptive extras and GeoPandas:
```bash
pip install LasBuildSeg[adaptive]
pip install geopandas
```

2. Put your `.laz` file and its ground-truth `.geojson` in the `TestLaz` folder
   (or edit the paths at the top of the script), then run the adaptive driver:
```python
import os
import geopandas as gpd
import numpy as np
import rasterio
import LasBuildSeg as Lasb

# --- Inputs ---
input_laz = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz'
GroundTruth = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson'
epsg_code = 6457
intermethod = 'nearest'

# --- Manual params ---
kernel_size = 3
alpha = 0.4
min_size, max_size = 50, 5000000
squareness_threshold, width_threshold = 0.3, 3

# 1) Adaptive CSF params from the raw cloud
csf_params = Lasb.laz_pre_analysis(input_laz)

# 2) Base rasters (last-return DSM + CSF DTM, aligned)
Lasb.generate_dsm_last_returns(input_laz, epsg_code, intermethod)
Lasb.generate_dtm_with_csf_last_returns(input_laz, epsg_code, 'dsm.tif', csf_params, intermethod)
Lasb.align_rasters('dtm_csf.tif', 'dsm.tif', 'aligned_dtm.tif')
Lasb.generate_ndhm('aligned_dtm.tif', 'dsm.tif')
Lasb.DSM_transform('dsm.tif')

# 3) Adaptive height/TRI thresholds
adaptive = Lasb.analyze_low_res_ndhm('ndhmtemp.tif')
tri_threshold = adaptive['tri_threshold']
height_threshold = adaptive['height_threshold']
filter_height_threshold = height_threshold * 0.8

# 4) Alpha-shape footprints -> rasterize onto the 3857 grid
Lasb.extract_building_footprints_with_alphashape(
    'ndhmtemp.tif', alpha, height_threshold, epsg_code, kernel_size)
Lasb.rasterize_geojson('alpha_shape_buildings.geojson', 'dsm3857.tif', 'alpha_shape_buildings.tif')

# 5) Filter contours (S1 no-TRI, S2 with adaptive TRI)
img, profile = Lasb.read_geotiff('alpha_shape_buildings.tif')
dem, _ = Lasb.read_geotiff('dsm3857.tif')
mask = profile and Lasb.read_geotiff('alpha_shape_buildings.tif')[0]

s1 = Lasb.filter_contoursntri(mask, profile, min_size, max_size,
                              squareness_threshold, width_threshold, filter_height_threshold)
s2 = Lasb.filter_contours(mask, dem, profile, min_size, max_size,
                          squareness_threshold, width_threshold, filter_height_threshold, tri_threshold)
```

The full, runnable version (with output folders, average-height attachment and
IoU scoring) is in `TestLaz/TestSingleLaz.py`.

3. Run `TestSingleLaz.py` and read the IoU scores printed at the end (S1, S2, and high-TRI S2).
