This Python package is BUilding Footprint Extractor Test

This of example of an how you can run your code

```python
import LasBuildSeg as Lasb
import numpy as np

Lasb.generate_dsm('YourLazfile', EPSG CODE, 1)
Lasb.generate_dtm('YourLazfile', EPSG CODE, 1)
Lasb.generate_ndhm('dtm.tif', 'dsm.tif')

img, profile = Lasb.read_geotiff('ndhm.tif')
dem, _ = Lasb.read_geotiff('dsm3857.tif')
img_8bit = Lasb.to_8bit(img)
constant = 3.6
block_size = 51
img_thresh = Lasb.threshold(img_8bit, block_size, constant)
kernel_size = 3
img_open = Lasb.morphopen(img_thresh, kernel_size)
min_size=35
max_size=5000
tri_threshold=3
building_mask = Lasb.filter_contours(img_open, dem, profile, min_size, max_size, tri_threshold=tri_threshold)
kernel_size = 3
CloseKernel_size=15
building_mask_closed = Lasb.close(building_mask, CloseKernel_size)
# Invert the building mask to make buildings appear as white ground pixels
inverted_building_mask = np.ones_like(building_mask, dtype=np.uint8) - building_mask_closed
Lasb.write_geotiff('buildings.tif', building_mask_closed, profile)
Lasb.building_footprints_to_geojson('buildings.tif', 'building.geojson')
print('All of our steps are done.')