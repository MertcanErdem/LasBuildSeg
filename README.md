# LasBuildSeg Building Footprint Extraction from LiDAR

This Python package is Building Footprint Extractor Test

The TestsingleLaz Python script can be used to test the Intersection over Union (IoU) rate of point cloud data and building footprints. To perform the test, navigate to the `testlaz` folder and run the script.

This example includes a point cloud dataset provided by GISCUP 2022. If you don't have your own data to test, you can use this dataset. The script also utilizes some functions from GISCUP 2022's [eval.py](https://sigspatial2022.sigspatial.org/giscup/submit.html).

## Requirements

| Library  | Version |
| ------------- | ------------- | 
| pyproj  | 3.5.0  | 
| numpy  | 1.23.5  |
| scipy  | 1.10.1 | 
| rasterio  | 1.3.4  |
| opencv-python  | 4.7.0.72  | 
| laspy  | 2.0.0  |
| proj   | 0.2.0 | 
| shapely  | 1.8.4 |

## Usage

You should have Geopandas python library in your computer.

1. Install the LasBuildSeg library using:
```bash
pip install LasBuildSeg
```
and also install Geopandas Using
```bash
pip install geopandas
```
2. Clone the TestLaz folder.

3. Run the TestSingleLaz.py script.

# Edit the parameters of the TestSingleLaz.py  
```python
# Define input parameters
input_laz = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz" # Path to your point cloud data
GroundTruth = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson"  # Path to Your Ground Truth
epsg_code = 6457 # EPSG code of the input laz data
multy = 1200  # Multiplication factor for DSM height enhancement
intermethod = 'nearest'  # Interpolation method ('cubic', 'nearest', or 'linear')


# Define output parameters
output_number = 1  # Use this variable to automatically name the output files (change it for each new input laz file)
constant = 3.6  # Adaptive threshold constant
block_size = 91  # Adaptive threshold block size
kernel_size = 3  # Morphological open kernel size
tri_threshold = 3 # Terrain Ruggedness Index threshold
```
