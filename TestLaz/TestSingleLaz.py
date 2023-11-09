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

import os




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


notri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S1_Contour', 'S1_Contour_' + str(output_number) + '.geojson'))
tri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S2_TRI', 'S2_TRI_' + str(output_number) + '.geojson'))
final_IOU= calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S3_MorphClose', 'S3_MorphClose_' + str(output_number) + '.geojson'))

notri_IOU=round(notri_IOU,2)
tri_IOU =round(tri_IOU ,2)
final_IOU =round(final_IOU,2)

print("S1 Contour Detection IoU is ",notri_IOU)
print("S2 Contour Detection wit TRI IoU is ",tri_IOU)
print("S3 Morphological Close IoU is ",final_IOU)