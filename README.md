# LasBuildSeg — Building Footprint Extraction from LiDAR

This Python package extracts building footprints from airborne LiDAR point clouds.

**As of v0.2.1 the workflow is adaptive:** instead of fixed thresholds, the key
parameters are estimated from the data itself — the CSF cloth settings from point
density and terrain slope, and the building height / TRI thresholds from a coarse
nDHM. The classic threshold-based functions are still included and unchanged.

The `TestSingleLaz` script can be used to test the Intersection over Union (IoU)
rate of point cloud data against building footprints. To run the test, navigate
to the `TestLaz` folder and run the script.

<img width="1488" height="979" alt="pipeline_diagram" src="https://github.com/user-attachments/assets/57a9356c-2f55-4af6-b8c1-e17595b5929e" />


>:white_check_mark: **If you are using this package for scientific research, please cite the paper** <a href=https://dl.acm.org/doi/10.1145/3589132.3625574>`<b>Reproducible Extraction of Building Footprints from Airborne LiDAR Data</b></a>

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

## How the parameters adapt

The pipeline reads two characteristics of each tile and maps them onto the CSF
ground-filter settings, then reads a coarse nDHM to set the height and TRI
thresholds. No manual tuning per tile is required.

<img width="1847" height="734" alt="adaptive_parameters" src="https://github.com/user-attachments/assets/d8c8abbe-eef6-465a-9a77-e0ab98cbdba6" />


## What the extraction does

Raw height thresholds are noisy. A morphological opening plus edge-aware
smoothing removes speckle, and an alpha shape wraps the remaining height pixels
into clean building footprints, which are then refined by the contour filters
(S1 without TRI, S2 with the adaptive TRI).

<img width="1907" height="741" alt="mask_stages" src="https://github.com/user-attachments/assets/92295d47-9698-47c1-a04c-d0175b7a55c6" />


## Requirements

`pip install LasBuildSeg` installs everything below, including the packages used
by the adaptive pipeline.

| Library  | Version |
| ------------- | ------------- |
| pyproj  | 3.5.0  |
| NumPy  | 1.23.5  |
| SciPy  | 1.10.1 |
| Rasterio  | 1.3.4  |
| OpenCV-python  | 4.7.0.72  |
| laspy  | 2.0.0  |
| lazrs  | 0.5.0 |
| PROJ  | 0.2.0 |
| Shapely  | >= 1.8.4 |
| GeoPandas  | >= 0.13.2 |
| cloth-simulation-filter (`CSF`)  | >= 1.1.0 |
| alphashape  | >= 1.3.1 |
| matplotlib  | >= 3.5.0 |

## Usage

In our `TestLaz` folder we use the .laz file with the ID of 11 which is located in the USA.

You should have GeoPandas installed. If your `PROJ_LIB` environment variable points to an older version of PROJ, unset it. You can override its location by setting `PROJ_LIB` to the directory containing `proj.db`; find the directory you are using with `pyproj.datadir.get_data_dir()` and then point `PROJ_LIB` at e.g. `C:\Users\<Yourusername>\anaconda3\lib\site-packages\pyproj\proj_dir\share\proj`. You can also fix it temporarily by adding `pyproj.datadir.set_data_dir(...)` in the script. If you don't want to do any of that, just run the script in a fresh environment.

1. Install the LasBuildSeg library (this pulls all dependencies):
```bash
pip install LasBuildSeg
```

2. Clone the `TestLaz` folder or copy the code below into your own script, with your `.laz` and ground-truth `.geojson` in the same folder.
```python
import os
import geopandas as gpd
import numpy as np
import rasterio
import LasBuildSeg as Lasb

# Define input parameters
input_laz = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz'      # input point cloud
GroundTruth = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson"  # ground truth
epsg_code = 6457          # EPSG code of the input laz data
intermethod = 'nearest'   # Interpolation method ('cubic', 'nearest', or 'linear')

# Manual params (you can change these to see how they affect the building maps)
kernel_size = 3
output_number = 11
alpha = 0.4
min_size = 50
max_size = 5000000
squareness_threshold = 0.3
width_threshold = 3

output_base_dir = 'output'
os.makedirs(output_base_dir, exist_ok=True)


def calc_iou(gdf_groundtruth, gdf_predict):
    """Intersection-over-Union between two polygon GeoDataFrames."""
    intersect = gdf_groundtruth.dissolve().intersection(gdf_predict.dissolve()).area
    union = gdf_groundtruth.dissolve().union(gdf_predict.dissolve()).area
    return (intersect / union)[0]


def calc_metrics(groundtruth_file, predict_file):
    """Read GeoJSONs and compute IoU, robust to missing/empty predictions."""
    if not os.path.exists(predict_file) or os.path.getsize(predict_file) == 0:
        return 0.0
    gdf_groundtruth = gpd.read_file(groundtruth_file)
    try:
        gdf_predict = gpd.read_file(predict_file)
        if gdf_predict.empty:
            return 0.0
    except Exception:
        return 0.0
    assert gdf_predict.crs == 3857, 'All geometries must be in EPSG:3857.'
    assert gdf_groundtruth.crs == 3857, 'All geometries must be in EPSG:3857.'
    iou = calc_iou(gdf_groundtruth, gdf_predict)
    if len(gdf_groundtruth) < len(gdf_predict):
        iou = iou * (len(gdf_groundtruth) / len(gdf_predict))
    return iou


def write_output(filename, data, profile, folder_name, dem_data, dem_profile):
    output_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    Lasb.write_geotiff(file_path, data, profile)
    geojson_path = file_path.replace('.tif', '.geojson')
    Lasb.building_footprints_to_geojson(file_path, geojson_path)
    if os.path.exists(geojson_path) and os.path.getsize(geojson_path) > 0:
        gdf_height = Lasb.calculate_average_height(geojson_path, dem_data, dem_profile)
        if not gdf_height.empty:
            gdf_height.to_file(geojson_path.replace('.geojson', '_with_height.geojson'), driver='GeoJSON')


# --- PHASE 1: adaptive CSF params from the raw cloud ---
csf_params = Lasb.laz_pre_analysis(input_laz)

# --- Base rasters (last-return DSM + CSF DTM, aligned) ---
Lasb.generate_dsm_last_returns(input_laz, epsg_code, intermethod)
Lasb.generate_dtm_with_csf_last_returns(input_laz, epsg_code, 'dsm.tif', csf_params, intermethod)
Lasb.align_rasters('dtm_csf.tif', 'dsm.tif', 'aligned_dtm.tif')
Lasb.generate_ndhm('aligned_dtm.tif', 'dsm.tif')   # -> ndhmtemp.tif (+ ndhm.tif)
Lasb.DSM_transform('dsm.tif')                       # -> dsm3857.tif

# --- PHASE 2: adaptive thresholds from the coarse nDHM ---
adaptive_params = Lasb.analyze_low_res_ndhm('ndhmtemp.tif')
tri_threshold = adaptive_params['tri_threshold']
alphashape_height_threshold = adaptive_params['height_threshold']
filter_height_threshold = adaptive_params['height_threshold'] * 0.8

# --- PHASE 3: alpha-shape footprints, then rasterize onto the 3857 grid ---
Lasb.extract_building_footprints_with_alphashape(
    'ndhmtemp.tif', alpha, alphashape_height_threshold, epsg_code, kernel_size)
Lasb.rasterize_geojson('alpha_shape_buildings.geojson', 'dsm3857.tif', 'alpha_shape_buildings.tif')

# Optional: re-open the rasterized mask
img_open, profile_open = Lasb.read_geotiff('alpha_shape_buildings.tif')
Lasb.write_geotiff('alpha_shape_buildings.tif', Lasb.morph_open(img_open, kernel_size), profile_open)

# --- Read mask + DEM ---
img, profile = Lasb.read_geotiff('alpha_shape_buildings.tif')
dem, _ = Lasb.read_geotiff('dsm3857.tif')
with rasterio.open('dsm3857.tif') as src:
    dem_data, dem_profile = src.read(1), src.profile
with rasterio.open('alpha_shape_buildings.tif') as src:
    alpha_shape_mask = src.read(1)

# --- FINAL FILTERING ---
# S1: contour filter without TRI
building_mask = Lasb.filter_contoursntri(alpha_shape_mask, profile, min_size, max_size,
                                         squareness_threshold, width_threshold, filter_height_threshold)
write_output('S1_Contour_' + str(output_number) + '.tif', building_mask, profile, 'S1_Contour', dem_data, dem_profile)

# S2: contour filter with adaptive TRI
building_mask_tri = Lasb.filter_contours(alpha_shape_mask, dem, profile, min_size, max_size,
                                         squareness_threshold, width_threshold, filter_height_threshold, tri_threshold)
write_output('S2_TRI_Refined_' + str(output_number) + '.tif', building_mask_tri, profile, 'S2_TRI_Refined', dem_data, dem_profile)

# --- EVALUATION ---
notri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S1_Contour', 'S1_Contour_' + str(output_number) + '.geojson'))
tri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S2_TRI_Refined', 'S2_TRI_Refined_' + str(output_number) + '.geojson'))

print("S1 Contour Detection IoU is ", round(notri_IOU, 2))
print("S2 Contour Detection with TRI IoU is ", round(tri_IOU, 2))
```

3. Run the `TestSingleLaz.py` script and read your results. The IoU is computed as the intersection over the union of the predicted and ground-truth footprints:

<img width="1035" height="647" alt="iou_explainer" src="https://github.com/user-attachments/assets/acd257e7-09c3-445f-a970-c0b958b2a014" />

