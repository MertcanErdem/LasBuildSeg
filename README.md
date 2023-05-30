This Python package is BUilding Footprint Extractor Test

This of example of an how you can run your code

```python
import LasBuildSeg as lasb
import numpy as np

# Define input parameters
input_laz = '<input>.laz'  # Path to the input laz/las data file
epsg_code = <epsg_code>  # EPSG code of the input laz data
multy = <DTM non-ground multipalction number>  # Multiplication factor for DSM height enhancement
intermethod = '<interpolation method>'  # Interpolation method ('cubic', 'nearest', or 'linear')

# Define default parameter values
constant = 3.6  # Adaptive threshold constant
block_size = 51  # Adaptive threshold block size
kernel_size = 3  # Morphological open kernel size
tri_threshold = 3  # Terrain Ruggedness Index threshold

# Define contour filtering parameters
min_size = 35
max_size = 5000
squareness_threshold = 0.3
width_threshold = 3
height_threshold = 3
CloseKernel_size = 15

# Generate DSM and DTM
lasb.generate_dsm(input_laz, epsg_code, intermethod)
lasb.generate_dtm(input_laz, epsg_code, intermethod, multy)

# Generate NDHM
lasb.generate_ndhm('dtm.tif', 'dsm.tif')

# Read NDHM image and profile
img, profile = lasb.read_geotiff('ndhm.tif')

# Transform DSM
lasb.DSM_Transform('dsm.tif')

# Read transformed DSM and profile
dem, _ = lasb.read_geotiff('dsm3857.tif')

# Convert image to 8-bit
img_8bit = lasb.to_8bit(img)

# Apply adaptive thresholding
img_thresh = lasb.threshold(img_8bit, block_size, constant)

# Apply morphological opening
img_open = lasb.morphopen(img_thresh, kernel_size)

# Filter contours without TRI
building_mask=Lasb.filter_contours(img_open, dem, profile, min_size, max_size, squareness_threshold, width_threshold, height_threshold, tri_threshold)

# Apply morphological closing
building_mask_closed = lasb.close(building_mask, CloseKernel_size)

# Write building mask to GeoTIFF
lasb.write_geotiff('buildings.tif', building_mask_closed, profile)

# Convert building mask to GeoJSON
lasb.building_footprints_to_geojson('buildings.tif', 'building.geojson')

# Print completion message
print('All steps are complete.')
