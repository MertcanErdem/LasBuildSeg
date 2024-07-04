import laspy
import numpy as np
import rasterio
import pyproj
import cv2
import scipy
from scipy.interpolate import griddata
from rasterio.transform import from_origin
import scipy.spatial
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import rasterio.features
from shapely.geometry import shape, mapping
import json
import geopandas as gpd

def generate_dsm(las_file_path: str, input_epsg: int, interpolation_method: str):
    """
    Generate a Digital Surface Model (DSM) from a LAS file.

    Args:
        las_file_path (str): Path to the LAS file.
        input_epsg (int): EPSG code of the input coordinate reference system (CRS).
        interpolation_method (str): Interpolation method to use.

    Outputs:
        dsm.tif the output DSM file.
    """
    # Read the LAS file
    las_file = laspy.read(las_file_path)
    resolution = 1

    # Create a Pyproj CRS object for the input EPSG code
    input_crs = pyproj.CRS.from_epsg(input_epsg)

    # Extract the x, y, and z coordinates from the LAS file
    x = las_file.x
    y = las_file.y
    z = las_file.z

    # Calculate the grid bounds based on the x and y coordinates
    x_min = np.floor(min(x))
    x_max = np.ceil(max(x))
    y_min = np.floor(min(y))
    y_max = np.ceil(max(y))

    # Generate the grid of points for the DSM
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    # Generate the DSM using the specified interpolation method
    dsm = griddata((x, y), z, (grid_x, grid_y), method=interpolation_method)

    # Save the DSM to a GeoTIFF file using rasterio
    with rasterio.open("dsm.tif", 'w', driver='GTiff', height=dsm.shape[0], width=dsm.shape[1], count=1, 
                       dtype=dsm.dtype, crs=input_crs, transform=rasterio.transform.Affine(resolution, 0, x_min, 0, resolution, y_min)) as dst:
        dst.write(dsm, 1)
    
    print('Success in Creating DSM')




def generate_dtm(las_file_path, input_epsg, interpolation_method, ground_multiplier):
    """
    Generate a Digital Terrain Model (DTM) from LiDAR data (This is not a %100 accurate DSM but it enchances
                                                            the resualts of the building extraction.)

    Args:
        las_file_path (str): Path to the LAS file.
        input_epsg (int): EPSG code of the input coordinate reference system.
        interpolation_method (str): Interpolation method for non-ground points.
        ground_multiplier (float): Multiplier for ground values in the DTM for contrast ecnheament.

    Returns:
        dtm.tif the output DSM file
    """

    # Load LiDAR data
    las_data = laspy.read(las_file_path)

    # Set the desired resolution
    resolution = 1

    # Create a Pyproj CRS object for the input EPSG code
    input_crs = pyproj.CRS.from_epsg(input_epsg) 

    # Extract x, y, and z coordinates from the LiDAR data
    points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    
    # Determine the bounds of the point cloud
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Classify ground points in the LiDAR data (class 2)
    ground_points = points[las_data.classification == 2]

    # Calculate the size of the output raster
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Create the output raster profile
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': input_crs,
        'transform': from_origin(min_x, min_y, resolution, -resolution)
    }

    # Create an empty numpy array for the output DTM
    dtm = np.zeros((height, width), dtype=np.float32)

    # Create a KDTree from the x, y coordinates of the ground points
    ground_tree = scipy.spatial.cKDTree(ground_points[:, :2])

    # Create a mesh grid for the output raster
    mesh_x, mesh_y = np.meshgrid(np.arange(min_x, max_x, resolution), np.arange(min_y, max_y, resolution))

    # Interpolate the z values of the ground points onto the mesh grid
    ground_values = ground_tree.query(np.vstack((mesh_x.ravel(), mesh_y.ravel())).T)[0]
    mesh_z = ground_values.reshape(mesh_x.shape)

    # Load the point cloud
    dtm_data = las_data
    dtm_points = np.vstack((dtm_data.x, dtm_data.y, dtm_data.z)).T

    # Classify points as non-ground (not class 2)
    non_ground_points = dtm_points[dtm_data.classification != 2]

    # Interpolate the non-ground points onto the mesh grid
    non_ground_z = griddata(non_ground_points[:, :2], non_ground_points[:, 2], (mesh_x, mesh_y), method=interpolation_method)

    # Subtract the interpolated non-ground values from the interpolated ground values also enhancge the contrast by using ground_multiplier
    dtm = ground_multiplier * mesh_z - non_ground_z

    # Write the output raster to a file
    with rasterio.open('dtm.tif', 'w', **profile) as dst:
        dst.write(dtm, 1)
    print('Success in Creating DTM')



def generate_ndhm(dtm_file, dsm_file):
    """
    Generate the Normalized Digital Height Model (NDHM) by subtracting the Digital Terrain Model (DTM) from the Digital Surface Model (DSM).

    Args:
        dtm_file (str): Path to the DTM file.
        dsm_file (str): Path to the DSM file.

    Returns:
       ndhm.tiff output NDHM file
    """

    # Load DSM and DTM
    with rasterio.open(dsm_file) as dsm_src:
        dsm = dsm_src.read(1)
        dsm_meta = dsm_src.profile

    with rasterio.open(dtm_file) as dtm_src:
        dtm = dtm_src.read(1)

    # Compute NDHM
    ndhm = dsm - dtm

    # Write NDHM to file
    ndhm_meta = dsm_meta.copy()
    ndhm_meta['dtype'] = 'float32'
    with rasterio.open('ndhmtemp.tif', 'w', **ndhm_meta) as ndhm_dst:
        ndhm_dst.write(ndhm.astype(np.float32), 1)

    # Define the target CRS as EPSG:3857
    target_crs = 'EPSG:3857'

    # Open the input file
    with rasterio.open('ndhmtemp.tif') as src:
        # Get the metadata of the input file
        src_profile = src.profile.copy()

        # Calculate the transform to the target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)

        # Update the metadata of the output file with the target CRS and nodata value
        src_profile.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': 0})

        # Create the output file
        with rasterio.open('ndhm.tif', 'w', **src_profile) as dst:
            # Reproject the input file to the target CRS
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                dst_nodata=0)

        print('Success in Creating NDHM')



def read_geotiff(filename):
    """
    Read a geotiff file and return the image data and profile.

    Args:
        filename (str): Path to the geotiff file.

    Returns:
        tuple: A tuple with two values - the image data and a dictionary with metadata about the image.
    """
    with rasterio.open(filename) as src:
        img = src.read(1)
        profile = src.profile.copy()
        profile.update({'crs': 'EPSG:3857'})
    return img, profile


def DSM_transform(dsm_file):
    """
    Transform the DSM to the target CRS (EPSG:3857).

    Args:
        dsm_file (str): Path to the DSM file.

    Returns:
        dsm3857.tiff: This image is a corrdinate trasnformed DSM file
    """
    target_crs = 'EPSG:3857'

    with rasterio.open(dsm_file) as src:
        src_profile = src.profile.copy()
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)

        src_profile.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': 0})

        with rasterio.open('dsm3857.tif', 'w', **src_profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                dst_nodata=0)


def to_8bit(image):
    """
    Convert an image to 8-bit color depth.

    Args:
        image: The image data.

    Returns:
        numpy.ndarray: The image data converted to 8-bit color depth.
    """
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return image_8bit


def threshold(image, block_size=51, constant=4.6):
    """
    Apply an adaptive threshold to an image to separate objects from the background.

    Args:
        image: The 8-bit image data.
        block_size (int): The size of the neighborhood used to calculate the threshold value.
        constant (float): A value subtracted from the calculated threshold value.

    Returns:
        numpy.ndarray: A binary image where objects are white and the background is black.
    """
    image_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
    return image_thresh


def morph_open(image, kernel_size=3):
    """
    Apply a morphological opening to an image to remove small objects.

    Args:
        image: The binary image data.
        kernel_size (int): The size of the kernel used for the morphological operation.

    Returns:
        numpy.ndarray: The image data with small objects removed.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image_open


def filter_contoursntri(image, profile, min_size=35, max_size=5000, squareness_threshold=0.3, width_threshold=3, height_threshold=3):
    """
    This section is only used for testing purposes so we can see the change between using the method of TRI and no TRI

    """
    contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixel_size = abs(profile['transform'][0])
    building_mask = np.zeros_like(image, dtype=np.uint8)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w < h:
            w, h = h, w
        squareness = w / h if h != 0 else 0
        size = w * h * pixel_size ** 2

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)

        if squareness >= squareness_threshold and min_size <= size <= max_size and w >= width_threshold and h >= height_threshold:
            cv2.drawContours(building_mask, [contour], -1, 255, -1)

    return building_mask
            
            
def filter_contours(image, dem, profile, min_size=35, max_size=5000, squareness_threshold=0.3, width_threshold=3, height_threshold=3, tri_threshold=3):
    """
    Filter out contours that do not meet certain criteria and create a binary mask of the remaining objects.

    Args:
        image: The image data with small objects removed.
        dem: The Digital Elevation Model data.
        profile: A dictionary with metadata about the image.
        min_size (int): The minimum size of objects to keep.
        max_size (int): The maximum size of objects to keep.
        squareness_threshold (float): The minimum squareness of objects to keep (ratio of width to height).
        width_threshold (int): The minimum width of objects to keep.
        height_threshold (int): The minimum height of objects to keep.
        tri_threshold (float): The maximum Terrain Ruggedness Index (TRI) value to keep.

    Returns:
        numpy.ndarray: A binary mask where the objects to keep are white and the rest is black.
    """
    contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixel_size = abs(profile['transform'][0])
    building_mask = np.zeros_like(image, dtype=np.uint8)
    dx, dy = np.gradient(dem)
    tri = np.sqrt(dx**2 + dy**2)
    tri /= pixel_size

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w < h:
            w, h = h, w
        squareness = w / h if h != 0 else 0
        size = w * h * pixel_size ** 2

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        tri_values = tri[mask == 1]
        tri_mean = np.mean(tri_values)

        if squareness >= squareness_threshold and min_size <= size <= max_size and w >= width_threshold and h >= height_threshold and tri_mean <= tri_threshold:
            cv2.drawContours(building_mask, [contour], -1, 255, -1)

    return building_mask


def close(image, kernel_size):
    """
    Apply a morphological closing to an image.

    Args:
        image: The image data.
        kernel_size (int): The size of the kernel used for the morphological operation.

    Returns:
        numpy.ndarray: The image data after morphological closing.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image_closed


def write_geotiff(filename, data, profile):
    """
    Write image data and profile to a geotiff file.

    Args:
        filename (str): Path to the output geotiff file.
        data: The image data.
        profile: A dictionary with metadata about the image.

    Returns:
        output image
    """
    profile.update(count=1, dtype=rasterio.uint8, crs=rasterio.crs.CRS.from_epsg(3857))
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.crs = profile['crs']
        dst.write(data.astype(rasterio.uint8), 1)


def building_footprints_to_geojson(tiff_file, geojson_file):
    """
    Convert building footprints to GeoJSON format.

    Args:
        tiff_file (str): Path to the input tiff file.
        geojson_file (str): Path to the output GeoJSON file.

    Returns:
        geojson_file: Output GeoJSON file.
    """
    with rasterio.open(tiff_file) as src:
        building_mask = src.read(1)

    building_only_mask = (building_mask == 0).astype('uint8')

    building_polygons = list(rasterio.features.shapes(building_only_mask, transform=src.transform))

    features = []
    for polygon, value in building_polygons:
        if value == 0:
            feature = {'type': 'Feature',
                       'geometry': mapping(shape(polygon)),
                       'properties': {'value': int(value)}}
            features.append(feature)

    geojson_dict = {'type': 'FeatureCollection', 'features': features, 'crs': {'type': 'name', 'properties': {'name': 'EPSG:3857'}}}

    with open(geojson_file, 'w') as f:
        json.dump(geojson_dict, f)
    print('Output GeoJSON is ready')


def calculate_average_height(geojson_file, height_data,height_profile):
    gdf = gpd.read_file(geojson_file)
    avg_heights = []
    for index, row in gdf.iterrows():
        polygon = row['geometry']
        polygon_heights = []
        for point in polygon.exterior.coords:
            # Extract height from height data at the coordinate location
            x, y = point[0], point[1]
            height = height_data[int((y - height_profile['transform'][5]) / abs(height_profile['transform'][4]))][int((x - height_profile['transform'][2]) / height_profile['transform'][0])]
            polygon_heights.append(height)
        # Calculate average height for the polygon
        avg_height = np.mean(polygon_heights)
        avg_heights.append(avg_height)
    # Add average height to GeoDataFrame
    gdf['average_height'] = avg_heights
    return gdf