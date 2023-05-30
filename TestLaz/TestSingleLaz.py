import numpy as np
import geopandas as gpd

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
   


import LasBuildSeg as Lasb
import numpy as np

# Define input parameters
input_laz = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz" # Path to the input to Your Point Cloud data in .laz/.las format.
GroundTruth = "USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson"  # Path to the input to Your Ground Truth file
epsg_code = 6457 # EPSG code of the input laz data
multy = 1200  # Multiplication factor for DSM height enhancement
intermethod = 'nearest'  # Interpolation method ('cubic', 'nearest', or 'linear')


# Define output parameters
output_number = 1  # Use this variable to automatically name the output files (change it for each new input laz file)
constant = 3.6  # Adaptive threshold constant
block_size = 91  # Adaptive threshold block size
kernel_size = 3  # Morphological open kernel size
tri_threshold = 3 # Terrain Ruggedness Index threshold

min_size = 35
max_size = 5000
squareness_threshold = 0.3
width_threshold = 3
height_threshold = 3

# Generate DSM and DTM
Lasb.generate_dsm(input_laz, epsg_code, intermethod)
Lasb.generate_dtm(input_laz, epsg_code, intermethod, multy)

# Generate NDHM
Lasb.generate_ndhm('dtm.tif', 'dsm.tif')

# Read NDHM image and profile
img, profile = Lasb.read_geotiff('ndhm.tif')

# Transform DSM
Lasb.DSM_transform('dsm.tif')

# Read transformed DSM and profile
dem, _ = Lasb.read_geotiff('dsm3857.tif')

# Convert image to 8-bit
img_8bit = Lasb.to_8bit(img)

# Apply adaptive thresholding
img_thresh = Lasb.threshold(img_8bit, block_size, constant)
Lasb.write_geotiff('img_thresh_' + str(output_number) + '.tif', img_thresh, profile)

# Apply morphological opening
img_open = Lasb.morph_open(img_thresh, kernel_size)
Lasb.write_geotiff('img_open_' + str(output_number) + '.tif', img_open, profile)

# Generate building footprints without TRI
building_masknotri = Lasb.filter_contoursntri(img_open, profile, min_size, max_size, squareness_threshold, width_threshold, height_threshold)
Lasb.write_geotiff('building_maskNOtri_' + str(output_number) + '.tif', building_masknotri, profile)
Lasb.building_footprints_to_geojson('building_maskNOtri_' + str(output_number) + '.tif', 'building_maskNOtri_' + str(output_number) + '.geojson')

# Generate building footprints with TRI
building_mask = Lasb.filter_contours(img_open, dem, profile, min_size, max_size, squareness_threshold, width_threshold, height_threshold, tri_threshold)
Lasb.write_geotiff('building_masktri_' + str(output_number) + '.tif', building_mask, profile)
Lasb.building_footprints_to_geojson('building_masktri_' + str(output_number) + '.tif', 'building_masktri_' + str(output_number) + '.geojson')

# Apply morphological closing
CloseKernel_size = 15
building_mask = Lasb.read_geotiff('building_masktri_' + str(output_number) + '.tif')[0]
building_mask_closed = Lasb.close(building_mask, CloseKernel_size)
Lasb.write_geotiff('building_mask_Final_' + str(output_number) + '.tif', building_mask_closed, profile)
Lasb.building_footprints_to_geojson('building_mask_Final_' + str(output_number) + '.tif', 'building_mask_Final_' + str(output_number) + '.geojson')

# Calculate IOU metrics
notri_IOU = calc_metrics(GroundTruth, 'building_maskNOtri_' + str(output_number) + '.geojson')
tri_IOU = calc_metrics(GroundTruth, 'building_masktri_' + str(output_number) + '.geojson')
final_IOU = calc_metrics(GroundTruth, 'building_mask_Final_' + str(output_number) + '.geojson')

# Print IOU results
print("Contour Detection with no TRI applied IoU:", notri_IOU)
print("Using Terrain Ruggedness Index IoU:", tri_IOU)
print("Final results after Morphological Close IoU:", final_IOU)
