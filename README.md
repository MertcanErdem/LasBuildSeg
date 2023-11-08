# LasBuildSeg Building Footprint Extraction from LiDAR

This Python package is Building Footprint Extractor Test

The TestsingleLaz Python script can be used to test the Intersection over Union (IoU) rate of point cloud data and building footprints. To perform the test, navigate to the `testlaz` folder and run the script.


## Data
This example includes a point cloud dataset provided by GISCUP 2022. If you don't have your own data to test, you can use this [dataset](https://sigspatial2022.sigspatial.org/giscup/download.html). The script also utilizes some functions from GISCUP 2022's [eval.py](https://sigspatial2022.sigspatial.org/giscup/submit.html).

| ID| EPSG Code | Point Denstiy(m2) | Number of buildings | Min Area(m2) | Max Area(m2) |
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

## Usage

In our Testlaz folder we use the .laz file with the ID of 11 which is located in USA.
![image](https://github.com/MertcanErdem/LasBuildSeg/assets/92017528/7866564d-ad2b-44b3-838e-0256fa4fcf99)


You should have Geopandas python library in your computer.

1. Install the LasBuildSeg library using:
```bash
pip install LasBuildSeg
```
and also install Geopandas Using
```bash
pip install geopandas
```

2. Clone the TestLaz folder or copy the bellow code to your python script with your laz and groundtruth in the same folder.
```python
import geopandas as gpd
import LasBuildSeg as Lasb
import numpy as np

# Define input parameters
input_laz = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz'  # Path to the input laz/las data file
GroundTruth = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson"  # Path to Your Ground Truth
epsg_code = 6457  # EPSG code of the input laz data
intermethod = 'nearest'  # Interpolation method ('cubic', 'nearest', or 'linear')

# You can change this paramets to see how it effects the building maps
constant = 3.6  # Adaptive threshold constant
block_size = 91  # Adaptive threshold block size
kernel_size = 3  # Morphological open kernel size
tri_threshold = 4  # Terrain Ruggedness Index threshold
multy = 1200  # Multiplication factor for DSM height enhancement
output_number = 11 # use this variable so you can change name of every output you get automaticly


# Generate DSM and DTM
Lasb.generate_dsm(input_laz, epsg_code, intermethod)
Lasb.generate_dtm(input_laz, epsg_code, intermethod, multy)

# Generate NDHM
Lasb.generate_ndhm('dtm.tif', 'dsm.tif')

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def calc_iou(gdf_groundtruth, gdf_predict):
    """Calculates intersection over union (iou) score.

    Args:
      gdf_groundtruth: Groundtruth GeoDataFrame of polygons.
      gdf_predict: Predicted GeoDataFrame of polygons.

    Returns:
      Intersection Over Union (IOU) Score.

    """
    
    intersect = gdf_groundtruth.dissolve().intersection(gdf_predict.dissolve()).area
    union = gdf_groundtruth.dissolve().union(gdf_predict.dissolve()).area
    iou = intersect / union
    
    return iou[0]

def calc_metrics(groundtruth_file, predict_file):
    """Reads geojson files, and calculates IOU.

    Args:
      groundtruth_file: Geojson file for groundtruth.
      predict_file: Geojson file for predictions.

    Returns:
      Intersection over union (IOU) score with punishment.
    """
    
    gdf_groundtruth = gpd.read_file(groundtruth_file)
    gdf_predict = gpd.read_file(predict_file)
    
    # Validate the CRS
    assert gdf_predict.crs==3857, (
        f'All geometries must be in EPSG:3857 Coordinate Reference System.')
    assert gdf_groundtruth.crs==3857, (
        f'All geometries must be in EPSG:3857 Coordinate Reference System.')
    
    # Validate the geometry column
    assert "geometry" in gdf_predict.columns, (
        f'Missing geometry column.')
    assert "geometry" in gdf_groundtruth.columns, (
        f'Missing geometry column.')
    
    iou = calc_iou(gdf_groundtruth, gdf_predict)
    # Punish if more polygon provided 
    if len(gdf_groundtruth)<len(gdf_predict):
        iou = iou * (len(gdf_groundtruth)/len(gdf_predict))
    
    return iou
    

# Define contour filtering parameters (You dont need to change this ones)
min_size = 35
max_size = 5000
squareness_threshold = 0.3
width_threshold = 3
height_threshold = 3
CloseKernel_size = 15






img, profile = Lasb.read_geotiff('ndhm.tif')
Lasb.DSM_transform('dsm.tif')
dem, _ = Lasb.read_geotiff('dsm3857.tif')
img_8bit = Lasb.to_8bit(img)

img_thresh = Lasb.threshold(img_8bit, block_size, constant)
img_open = Lasb.morph_open(img_thresh, kernel_size)

blockPrint()
import os

# Create output folders for each step if they don't exist
output_base_dir = 'output'  # Change this to your desired output base directory
os.makedirs(output_base_dir, exist_ok=True)

# Function to write files to the appropriate folder
def write_output(filename, data, profile, folder_name):
    output_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    Lasb.write_geotiff(file_path, data, profile)
    Lasb.building_footprints_to_geojson(file_path, file_path.replace('.tif', '.geojson'))


# S1 Step
building_mask = Lasb.filter_contoursntri(img_open, profile, min_size, max_size, squareness_threshold, width_threshold, height_threshold)
write_output('S1_Contour_' + str(output_number) + '.tif', building_mask, profile, 'S1_Contour')

# S2 Step
dem, _ = Lasb.read_geotiff('dsm3857.tif')
building_mask = Lasb.filter_contours(img_open, dem, profile, min_size, max_size, squareness_threshold, width_threshold, height_threshold, tri_threshold)
write_output('S2_TRI_' + str(output_number) + '.tif', building_mask, profile, 'S2_TRI')

# S3 Step
building_mask_closed = Lasb.close(building_mask, CloseKernel_size)
write_output('S3_MorphClose_' + str(output_number) + '.tif', building_mask_closed, profile, 'S3_MorphClose')

# Make sure to close any open files or resources as needed


enablePrint()


notri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S1_Contour', 'S1_Contour_' + str(output_number) + '.geojson'))
tri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S2_TRI', 'S2_TRI_' + str(output_number) + '.geojson'))
final_IOU= calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S3_MorphClose', 'S3_MorphClose_' + str(output_number) + '.geojson'))

print("S1 Contour Detection Iou %",notri_IOU)
print("S2 Contour Detection wit TRI Iou %",tri_IOU)
print("S3 Morphological Close Iou %",final_IOU)
``` 

3. Run the TestSingleLaz.py script and get your resaults.
   
   ![image](https://github.com/MertcanErdem/LasBuildSeg/assets/92017528/4f73c48f-d77a-48f7-9b61-f5f69692d067)
   ![image](https://github.com/MertcanErdem/LasBuildSeg/assets/92017528/41c96aa0-32bb-4ddb-a535-8489747e8767)
   ![image](https://github.com/MertcanErdem/LasBuildSeg/assets/92017528/800c3f61-0f6b-43b3-9bde-fc5113dc844f)




