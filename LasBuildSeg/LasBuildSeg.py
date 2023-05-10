# -*- coding: utf-8 -*-
"""
Created on Sun May 7 10:46:30 2023


@author: Mertcan
"""
#This of example of an how you can run your code
# import LasBuildSeg as Lasb
# import numpy as np

# Lasb.generate_dsm('USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz', 8734, 1)
# Lasb.generate_dtm('USGS_LPC_IL_HicksDome_FluorsparDistrict_2019_D19_2339_5650.laz', 8734, 1)
# Lasb.generate_ndhm('dtm.tif', 'dsm.tif')
# img, profile = Lasb.read_geotiff('ndhm.tif')
# img_8bit = Lasb.to_8bit(img)
# constant = 4.6
# block_size = 51
# img_thresh = Lasb.threshold(img_8bit, block_size, constant)
# kernel_size = 3
# img_open = Lasb.morphopen(img_thresh, kernel_size)
# min_size=35
# max_size=5000
# building_mask = Lasb.filter_contours(img_open, profile, min_size, max_size)
# kernel_size = 3
# CloseKernel_size=15
# building_mask_closed = Lasb.close(building_mask, CloseKernel_size)
# # Invert the building mask to make buildings appear as white ground pixels
# inverted_building_mask = np.ones_like(building_mask, dtype=np.uint8) - building_mask_closed
# Las.write_geotiff('buildings.tif', inverted_building_mask, profile)
# print('All of our steps are done.')

import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
import pyproj
import cv2
from rasterio.transform import from_origin
import scipy
from rasterio.warp import calculate_default_transform, reproject, Resampling



def generate_dsm(las_file_name: str, input_epsg: int, resolution):
    # Read the LAS file
    las_file = laspy.read(las_file_name)

    # Set the CRS information to projection
    las_file.header.scale[0] = 0.01
    las_file.header.scale[1] = 0.01
    las_file.header.scale[2] = 0.01

    # Write the updated LAS file
    las_file.write("updated_file.las")

    # Read the updated LAS file
    las_file = laspy.read('updated_file.las')

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
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max , resolution), np.arange(y_min, y_max , resolution))

    # Generate the DSM using Nearest Neighbor interpolation of the point cloud
    dsm = griddata((x, y), z, (grid_x, grid_y), method='nearest')

    # Save the DSM to a GeoTIFF file using rasterio
    with rasterio.open("dsm.tif", 'w', driver='GTiff', height=dsm.shape[0], width=dsm.shape[1], count=1, 
                       dtype=dsm.dtype, crs=input_crs, transform=rasterio.transform.Affine(resolution, 0, x_min, 0, resolution, y_min)) as dst:
        dst.write(dsm, 1)
    
    print('Success in Creating DSM')
#generate_dsm('Yourdata.laz', 8734, 1)



def generate_dtm(las_file_path, input_epsg, resolution):
    # Load LiDAR data
    las_file_dtm = laspy.read(las_file_path)
     # Set the CRS information to projection
    las_file_dtm.header.scale[0] = 0.01
    las_file_dtm.header.scale[1] = 0.01
    las_file_dtm.header.scale[2] = 0.01

    # Write the updated LAS file
    las_file_dtm.write("updated_file.las")

    # Read the updated LAS file
    las_file_dtm = laspy.read('updated_file.las')

    # Create a Pyproj CRS object for the input EPSG code
    input_crs = pyproj.CRS.from_epsg(input_epsg) 

    points = np.vstack((las_file_dtm.x, las_file_dtm.y, las_file_dtm.z)).T
    

    # Classify ground points in lidar data class 2
    ground_points = points[las_file_dtm.classification == 2]

    # Determine the bounds of the point cloud
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])

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
    tree = scipy.spatial.cKDTree(ground_points[:, :2])

    # Create a mesh grid for the output raster
    mesh_x, mesh_y = np.meshgrid(np.arange(min_x, max_x, resolution), np.arange(min_y, max_y, resolution))

    # Interpolate the z values of the ground points onto the mesh grid
    values = tree.query(np.vstack((mesh_x.ravel(), mesh_y.ravel())).T)[0]
    mesh_z = values.reshape(mesh_x.shape)


    # Load the point cloud
    dsm_file = laspy.read('updated_file.las')
    dsm_points = np.vstack((dsm_file.x, dsm_file.y, dsm_file.z)).T

    # Classify points as aboveground features (non-ground points)
    non_ground_points = dsm_points[dsm_file.classification != 2]

    # Interpolate the non-ground points onto the mesh grid
    non_ground_z = griddata(non_ground_points[:, :2], non_ground_points[:, 2], (mesh_x, mesh_y), method='nearest')

    # Subtract the interpolated non-ground values from the interpolated ground values
    dtm = 10*mesh_z - non_ground_z

    # Write the output raster to a file
    with rasterio.open('dtm.tif', 'w', **profile) as dst:
        dst.write(dtm, 1)
#generate_dtm('Yourdata.laz', EPSG, Resulation)





#DTM and DSM need to be in same resulation
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

#DTM and DSM need to be in same resulation
def generate_ndhm(dtm_file, dsm_file):
    # Load DSM and DTM
    with rasterio.open(dsm_file) as src:
        dsm = src.read(1)
        dsm_meta = src.profile

    with rasterio.open(dtm_file) as src:
        dtm = src.read(1)

    # Compute NDHM
    ndhm = dsm - dtm

    # Write NDHM to file
    ndhm_meta = dsm_meta.copy()
    ndhm_meta['dtype'] = 'float32'
    with rasterio.open('ndhmtemp.tif', 'w', **ndhm_meta) as dst:
        dst.write(ndhm.astype(np.float32), 1)

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


#Our algorithms

# This function reads a geotiff file and returns the image and profile data.
# filename: the path to the geotiff file
# Returns a tuple with two values: the image data and a dictionary with metadata about the image.
def read_geotiff(filename):
    with rasterio.open(filename) as src:
        img = src.read(1)
        profile = src.profile.copy()
    return img, profile


# This function converts an image to 8-bit color depth.
# img: the image data
# Returns the image data converted to 8-bit color depth.
def to_8bit(img):
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return img_8bit

# This function applies an adaptive threshold to an image to separate objects from the background.
# img_8bit: the 8-bit image data
# block_size: the size of the neighborhood used to calculate the threshold value
# constant: a value subtracted from the calculated threshold value
# Returns a binary image where objects are white and the background is black.
def threshold(img_8bit, block_size=51, constant=4.6):
    img_thresh = cv2.adaptiveThreshold(img_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
    return img_thresh

# This function applies a morphological opening to an image to remove small objects.
# img_thresh: the binary image data
# kernel_size: the size of the kernel used for the morphological operation
# Returns the image data with small objects removed.
def morphopen(img_thresh, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    return img_open

# This function filters out contours that do not meet certain criteria (size, shape, etc.) and creates a binary mask of the remaining objects.
# img_open: the image data with small objects removed
# profile: a dictionary with metadata about the image
# min_size: the minimum size of objects to keep
# max_size: the maximum size of objects to keep
# squareness_threshold: the minimum squareness of objects to keep (ratio of width to height)
# width_threshold: the minimum width of objects to keep
# height_threshold: the minimum height of objects to keep
# Returns a binary mask where the objects to keep are white and the rest is black.
def filter_contours(img_open, profile, min_size=35, max_size=5000, squareness_threshold=0.3, width_threshold=3, height_threshold=3):
    contours, _ = cv2.findContours(img_open.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixel_size = abs(profile['transform'][0])
    building_mask = np.zeros_like(img_open, dtype=np.uint8)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w < h:
            w, h = h, w
        squareness = w / h if h != 0 else 0
        size = w * h * pixel_size ** 2
        if squareness >= squareness_threshold and min_size <= size <= max_size and w >= width_threshold and h >= height_threshold:
            cv2.drawContours(building_mask, [contour], -1, 255, -1)

    return building_mask

def close(building_mask, CloseKernel_size):
    # Create a kernel for morphological closing operation
    kernel = np.ones((CloseKernel_size, CloseKernel_size), np.uint8)
    # Perform morphological closing operation on the input image using the kernel
    building_mask_closed = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)
    return building_mask_closed

def write_geotiff(filename, data, profile):
    # Update the profile with the required metadata for writing a GeoTIFF file
    profile.update(count=1, dtype=rasterio.uint8, crs=rasterio.crs.CRS.from_epsg(3857))
    # Open a new GeoTIFF file in write mode using the profile information
    with rasterio.open(filename, 'w', **profile) as dst:
        # Set the CRS information for the output file
        dst.crs = profile['crs']  # Add CRS information
        dst.write(data.astype(rasterio.uint8), 1)
#input_crs = pyproj.CRS.from_epsg(input_epsg)        
#crs=pyproj.CRS.from_epsg(3857)

