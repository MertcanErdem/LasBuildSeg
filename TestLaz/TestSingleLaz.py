"""
TestSingleLaz.py  --  Adaptive building-footprint pipeline (LasBuildSeg >= 0.2.0)

This is the "active / adaptive parameters" workflow. Unlike the classic
threshold-based example, every heavy step is delegated to a LasBuildSeg
function, and the key parameters (CSF settings, height threshold, TRI
threshold) are derived automatically from the data instead of being fixed.

Pipeline:
    1.  laz_pre_analysis ............ derive CSF params from density & slope
    2.  generate_dsm_last_returns ... last-return DSM with void masking
    3.  generate_dtm_with_csf_last_returns ... CSF-based DTM
    4.  align_rasters / generate_ndhm / DSM_transform ... base rasters
    5.  analyze_low_res_ndhm ........ adaptive height & TRI thresholds
    6.  extract_building_footprints_with_alphashape ... alpha-shape footprints
    7.  rasterize_geojson ........... burn footprints onto the 3857 grid
    8.  filter_contoursntri / filter_contours ... S1 / S2 refinement
    9.  IoU evaluation against the ground truth

Optional dependencies used here: CSF (cloth-simulation-filter), alphashape.
Install with:  pip install LasBuildSeg cloth-simulation-filter alphashape geopandas
"""

import os
import geopandas as gpd
import numpy as np
import rasterio
import LasBuildSeg as Lasb

# ----------------------------------------------------------------------
# INPUT PARAMETERS
# ----------------------------------------------------------------------
input_laz = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz'      # input point cloud
GroundTruth = 'USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650_gt_buildings.geojson'  # ground truth
epsg_code = 6457          # EPSG code of the input laz data
intermethod = 'nearest'   # Interpolation method ('cubic', 'nearest', or 'linear')

# --- USER-DEFINED (manual) PARAMETERS -------------------------------------
kernel_size = 3               # Pre-/post-alpha morphological-open kernel
output_number = 11            # Suffix so each run writes uniquely named outputs
alpha = 0.4                   # Alpha-shape tightness (higher = more detail)
min_size = 50                 # Min contour area to keep
max_size = 5000000            # Max contour area to keep
squareness_threshold = 0.3    # Min bounding-box squareness
width_threshold = 3           # Min bounding-box width (pixels)
CloseKernel_size = 15         # S3 morphological-close kernel

# --- MORPHOLOGICAL OPERATION TOGGLES --------------------------------------
APPLY_MORPH_OPEN_AFTER_ALPHA = True   # Re-open the rasterized alpha mask
APPLY_MORPH_CLOSE_S3 = False          # Run the optional S3 close step

# --- OUTPUT ---------------------------------------------------------------
output_base_dir = 'output'
os.makedirs(output_base_dir, exist_ok=True)


# ----------------------------------------------------------------------
# EVALUATION HELPERS (adapted from GISCUP 2022 eval.py)
# ----------------------------------------------------------------------
def calc_iou(gdf_groundtruth, gdf_predict):
    """Intersection-over-Union between two polygon GeoDataFrames."""
    intersect = gdf_groundtruth.dissolve().intersection(gdf_predict.dissolve()).area
    union = gdf_groundtruth.dissolve().union(gdf_predict.dissolve()).area
    iou = intersect / union
    return iou[0]


def calc_metrics(groundtruth_file, predict_file):
    """Read GeoJSONs and compute IoU, robust to missing/empty predictions."""
    if not os.path.exists(predict_file) or os.path.getsize(predict_file) == 0:
        print(f"Warning: Prediction file {predict_file} is missing or empty. IoU set to 0.")
        return 0.0

    gdf_groundtruth = gpd.read_file(groundtruth_file)
    try:
        gdf_predict = gpd.read_file(predict_file)
        if gdf_predict.empty:
            print(f"Warning: Prediction file {predict_file} is empty. IoU set to 0.")
            return 0.0
    except Exception as e:
        print(f"Error reading {predict_file}: {e}. IoU set to 0.")
        return 0.0

    assert gdf_predict.crs == 3857, 'All geometries must be in EPSG:3857.'
    assert gdf_groundtruth.crs == 3857, 'All geometries must be in EPSG:3857.'
    assert "geometry" in gdf_predict.columns, 'Missing geometry column.'
    assert "geometry" in gdf_groundtruth.columns, 'Missing geometry column.'

    iou = calc_iou(gdf_groundtruth, gdf_predict)
    # Punish if more polygons are predicted than exist in the ground truth.
    if len(gdf_groundtruth) < len(gdf_predict):
        iou = iou * (len(gdf_groundtruth) / len(gdf_predict))
    return iou


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)


def write_output(filename, data, profile, folder_name, dem_data, dem_profile):
    """Write a mask as GeoTIFF + GeoJSON, and attach average heights."""
    output_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)

    Lasb.write_geotiff(file_path, data, profile)
    geojson_path = file_path.replace('.tif', '.geojson')
    Lasb.building_footprints_to_geojson(file_path, geojson_path)

    if os.path.exists(geojson_path) and os.path.getsize(geojson_path) > 0:
        gdf_height = Lasb.calculate_average_height(geojson_path, dem_data, dem_profile)
        if not gdf_height.empty:
            height_geojson_path = geojson_path.replace('.geojson', '_with_height.geojson')
            gdf_height.to_file(height_geojson_path, driver='GeoJSON')


# ======================================================================
# MAIN EXECUTION
# ======================================================================
print("--- STARTING ADAPTIVE PIPELINE ---")

# --- PHASE 1: derive CSF parameters from the raw cloud --------------------
csf_params = Lasb.laz_pre_analysis(input_laz)

# --- Generate the base rasters (all adaptive / last-return based) ---------
print("Generating last-return DSM...")
Lasb.generate_dsm_last_returns(input_laz, epsg_code, intermethod)

print("Generating DTM with adaptive CSF (last-return input)...")
Lasb.generate_dtm_with_csf_last_returns(input_laz, epsg_code, 'dsm.tif', csf_params, intermethod)

print("Aligning DTM to DSM grid...")
Lasb.align_rasters('dtm_csf.tif', 'dsm.tif', 'aligned_dtm.tif')

Lasb.generate_ndhm('aligned_dtm.tif', 'dsm.tif')   # -> ndhmtemp.tif (+ ndhm.tif)
Lasb.DSM_transform('dsm.tif')                       # -> dsm3857.tif
print("--- Base rasters generated ---")

# --- PHASE 2: adaptive thresholds from the coarse nDHM --------------------
adaptive_params = Lasb.analyze_low_res_ndhm('ndhmtemp.tif')
tri_threshold = adaptive_params['tri_threshold']
alphashape_height_threshold = adaptive_params['height_threshold']
filter_height_threshold = adaptive_params['height_threshold'] * 0.8

print("\n[i] --- ADAPTIVE PARAMETERS SET ---")
print(f"[i] AlphaShape Height Threshold: {alphashape_height_threshold:.2f}m")
print(f"[i] Filter Height Threshold:     {filter_height_threshold:.2f}m")
print(f"[i] TRI Threshold:               {tri_threshold:.2f}")
print(f"[i] --- MANUAL PARAMETERS ---")
print(f"[i] Alpha={alpha}, Squareness={squareness_threshold}, MorphOpen Kernel={kernel_size}")
print("-------------------------------------\n")

# --- PHASE 3: alpha-shape footprint extraction ----------------------------
Lasb.extract_building_footprints_with_alphashape(
    raster_path='ndhmtemp.tif',
    alpha=alpha,
    adaptive_height_threshold=alphashape_height_threshold,
    epsg_code=epsg_code,
    kernel_size=kernel_size,          # pre-alpha morph open + FXAA smoothing
    output_path='alpha_shape_buildings.geojson',
    show_plot=False                   # set True to preview the before/after mask
)

# Burn the alpha-shape footprints onto the EPSG:3857 reference grid.
Lasb.rasterize_geojson('alpha_shape_buildings.geojson', 'dsm3857.tif', 'alpha_shape_buildings.tif')

# --- Optional post-alpha morphological opening ----------------------------
if APPLY_MORPH_OPEN_AFTER_ALPHA:
    print(f"[i] Applying post-alpha morphological opening (kernel {kernel_size})...")
    img_open, profile_open = Lasb.read_geotiff('alpha_shape_buildings.tif')
    img_opened = Lasb.morph_open(img_open, kernel_size)
    Lasb.write_geotiff('alpha_shape_buildings.tif', img_opened, profile_open)

# --- Read the mask and the DEM for filtering / heights --------------------
img, profile = Lasb.read_geotiff('alpha_shape_buildings.tif')
dem, _ = Lasb.read_geotiff('dsm3857.tif')
with rasterio.open('dsm3857.tif') as src:
    dem_data = src.read(1)
    dem_profile = src.profile
alpha_shape_mask = read_raster('alpha_shape_buildings.tif')

# ----------------------------------------------------------------------
# FINAL FILTERING
# Note: the last argument to filter_contours[ntri] is the contour bounding-box
# minimum-height (in pixels); here we feed it the adaptive height value, just
# like the active/adaptive-params workflow.
# ----------------------------------------------------------------------
print("[i] Running final filtering steps...")

# S1: contour filter without TRI
building_mask = Lasb.filter_contoursntri(
    alpha_shape_mask, profile, min_size, max_size,
    squareness_threshold, width_threshold, filter_height_threshold)
write_output('S1_Contour_' + str(output_number) + '.tif',
             building_mask, profile, 'S1_Contour', dem_data, dem_profile)

# S2: contour filter with adaptive TRI
building_mask_with_tri = Lasb.filter_contours(
    alpha_shape_mask, dem, profile, min_size, max_size,
    squareness_threshold, width_threshold, filter_height_threshold, tri_threshold)
write_output('S2_TRI_Refined_' + str(output_number) + '.tif',
             building_mask_with_tri, profile, 'S2_TRI_Refined', dem_data, dem_profile)

# S2 high-TRI variant
tri_threshold_high = tri_threshold * 1.5
print(f"[i] Using high TRI threshold: {tri_threshold_high:.2f}")
building_mask_with_tri_high = Lasb.filter_contours(
    alpha_shape_mask, dem, profile, min_size, max_size,
    squareness_threshold, width_threshold, filter_height_threshold, tri_threshold_high)
write_output('S2_TRI_High_Refined_' + str(output_number) + '.tif',
             building_mask_with_tri_high, profile, 'S2_TRI_High_Refined', dem_data, dem_profile)

# S3: optional morphological close
if APPLY_MORPH_CLOSE_S3:
    print("[i] Running S3: Morphological Close...")
    building_mask_closed = Lasb.close(building_mask_with_tri, CloseKernel_size)
    write_output('S3_MorphClose_' + str(output_number) + '.tif',
                 building_mask_closed, profile, 'S3_MorphClose', dem_data, dem_profile)
else:
    print("[i] Skipping S3: Morphological Close step.")

# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------
print("[i] Calculating final scores...")
notri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S1_Contour', 'S1_Contour_' + str(output_number) + '.geojson'))
tri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S2_TRI_Refined', 'S2_TRI_Refined_' + str(output_number) + '.geojson'))
high_tri_IOU = calc_metrics(GroundTruth, os.path.join(output_base_dir, 'S2_TRI_High_Refined', 'S2_TRI_High_Refined_' + str(output_number) + '.geojson'))

final_IOU = 0.0
if APPLY_MORPH_CLOSE_S3:
    final_IOU_path = os.path.join(output_base_dir, 'S3_MorphClose', 'S3_MorphClose_' + str(output_number) + '.geojson')
    if os.path.exists(final_IOU_path):
        final_IOU = calc_metrics(GroundTruth, final_IOU_path)

print("\n--- FINAL RESULTS ---")
print(f"[i] Adaptive Params: Height={filter_height_threshold:.2f}m, TRI={tri_threshold:.2f}")
print(f"[i] Manual Params:   Alpha={alpha}, Squareness={squareness_threshold}")
print("S1 Contour Detection IoU is", round(notri_IOU, 2))
print("S2 Contour Detection with TRI IoU is", round(tri_IOU, 2))
print("S2 Contour Detection with High TRI IoU is", round(high_tri_IOU, 2))
if APPLY_MORPH_CLOSE_S3:
    print("S3 Morphological Close IoU is", round(final_IOU, 2))
else:
    print("S3 Morphological Close IoU is [SKIPPED]")
