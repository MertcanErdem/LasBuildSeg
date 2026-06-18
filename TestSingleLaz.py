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


def calculate_average_height(geojson_file, height_data, height_profile):
    """
    Calculate the average height of each building footprint by sampling a
    height raster (e.g. an nDHM/DSM) inside every polygon.

    This is a robust, mask-based implementation: it rasterizes each geometry
    against the height raster instead of sampling only the polygon vertices,
    handles MultiPolygons, reprojects the footprints to the raster CRS when
    needed, and ignores nodata / non-positive cells.

    Args:
        geojson_file (str): Path to the building-footprint GeoJSON file.
        height_data (numpy.ndarray): The height raster (single band).
        height_profile (dict): Rasterio profile of ``height_data`` (must
            contain ``transform``, ``height``, ``width`` and, ideally,
            ``crs`` and ``nodata``).

    Returns:
        geopandas.GeoDataFrame: The footprints with an added ``avg_height``
        column. Returns an empty GeoDataFrame if the input cannot be read.
    """
    try:
        gdf = gpd.read_file(geojson_file)
        if gdf.empty:
            print(f"Warning: {geojson_file} is empty. No heights to calculate.")
            return gdf
    except Exception as e:
        print(f"Error reading {geojson_file} for height calculation: {e}")
        return gpd.GeoDataFrame()

    # Reproject footprints to match the raster CRS, if we know it.
    raster_crs = height_profile.get('crs', None)
    if raster_crs is not None and gdf.crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    nodata_value = height_profile.get('nodata', None)
    avg_heights = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            avg_heights.append(0.0)
            continue

        mask = rasterio.features.geometry_mask(
            [geom],
            out_shape=(height_profile['height'], height_profile['width']),
            transform=height_profile['transform'],
            invert=True,
            all_touched=True
        )
        heights = height_data[mask]
        if nodata_value is not None:
            heights = heights[heights != nodata_value]
        valid_heights = heights[heights > 0]
        avg_height = float(np.mean(valid_heights)) if valid_heights.size > 0 else 0.0
        avg_heights.append(avg_height)

    gdf['avg_height'] = avg_heights
    return gdf

# ======================================================================
# Adaptive pipeline additions (ported & generalized from the
# "activeparams / adaptiveparams" driver script).
#
# These functions extend the classic threshold-based workflow above with:
#   * last-return DSM generation with void masking,
#   * CSF (Cloth Simulation Filter) based DTM generation,
#   * automatic / adaptive parameter estimation,
#   * alpha-shape building-footprint extraction,
#   * FXAA-style edge smoothing, raster alignment and GeoJSON rasterization.
#
# Heavy / optional third-party packages (CSF, alphashape, matplotlib) are
# imported lazily inside the functions that need them, so importing
# LasBuildSeg never fails just because an optional dependency is missing.
# ======================================================================


def generate_dsm_last_returns(las_file_path, input_epsg, interpolation_method,
                              resolution=1, output_path='dsm.tif',
                              void_filter_size=3, nodata_value=-9999.0):
    """
    Generate a DSM from a LAS/LAZ file using ONLY last returns.

    Cells with no nearby LiDAR returns (e.g. water bodies, scan shadows) are
    masked to ``nodata_value`` instead of being interpolated across, which
    avoids "smearing" the surface over voids.

    Args:
        las_file_path (str): Path to the LAS/LAZ file.
        input_epsg (int): EPSG code of the input CRS.
        interpolation_method (str): 'nearest', 'linear' or 'cubic'.
        resolution (float): Output cell size in CRS units. Default 1.
        output_path (str): Output GeoTIFF path. Default 'dsm.tif'.
        void_filter_size (int): Neighbourhood size for the void-tolerance
            dilation. Larger values tolerate sparser point clouds. Default 3.
        nodata_value (float): Value written to masked/void cells.

    Returns:
        str: The output path on success, or ``None`` if no last returns exist.
    """
    from scipy.ndimage import maximum_filter

    print(f"[i] Generating DSM using Last Returns from: {las_file_path}")

    las_file = laspy.read(las_file_path)
    input_crs = pyproj.CRS.from_epsg(input_epsg)

    last_return_mask = las_file.return_number == las_file.number_of_returns
    x = np.array(las_file.x)[last_return_mask]
    y = np.array(las_file.y)[last_return_mask]
    z = np.array(las_file.z)[last_return_mask]

    print(f"    - Total Points: {len(las_file.x)}")
    print(f"    - Last Return Points Used: {len(x)}")

    if len(x) == 0:
        print("[!] Error: No last return points found.")
        return None

    x_min = np.floor(x.min())
    x_max = np.ceil(x.max())
    y_min = np.floor(y.min())
    y_max = np.ceil(y.max())

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    dsm = griddata((x, y), z, (grid_x, grid_y), method=interpolation_method)

    # --- VOID MASKING: flag cells with no nearby LiDAR returns ---
    cols = ((x - x_min) / resolution).astype(int)
    rows = ((y - y_min) / resolution).astype(int)
    valid = (rows >= 0) & (rows < grid_x.shape[0]) & (cols >= 0) & (cols < grid_x.shape[1])
    rows, cols = rows[valid], cols[valid]

    counts = np.zeros(grid_x.shape, dtype=np.int32)
    np.add.at(counts, (rows, cols), 1)

    has_data = maximum_filter(counts > 0, size=void_filter_size)

    dsm = np.where(has_data, dsm, nodata_value)
    dsm = np.where(np.isnan(dsm), nodata_value, dsm).astype(np.float32)

    print(f"    - Void cells masked: {(~has_data).sum()} of {has_data.size}")

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=dsm.shape[0], width=dsm.shape[1], count=1,
        dtype='float32', crs=input_crs, nodata=nodata_value,
        transform=rasterio.transform.Affine(resolution, 0, x_min, 0, resolution, y_min)
    ) as dst:
        dst.write(dsm, 1)

    print('[\u2713] Success in Creating Last-Return DSM (with void masking)')
    return output_path


def laz_pre_analysis(laz_path):
    """
    Scan a raw LAS/LAZ file to derive point density and global terrain slope,
    then map those characteristics onto sensible CSF (Cloth Simulation Filter)
    parameters.

    Args:
        laz_path (str): Path to the LAS/LAZ file.

    Returns:
        dict: ``{'cloth_resolution', 'rigidness', 'class_threshold'}``.
    """
    import math

    print(f"[i] Phase 1: Analyzing raw .laz file: {laz_path}")

    las = laspy.read(laz_path)

    # 1. Point density (points per m^2)
    area = (las.header.maxs[0] - las.header.mins[0]) * (las.header.maxs[1] - las.header.mins[1])
    point_count = las.header.point_count
    point_density = point_count / area if area else 0.0
    print(f"[\u2713] Point Density: {point_density:.2f} points/m\u00b2")

    # 2. Global topography / steepness via least-squares plane fit
    coords = np.vstack((las.x, las.y, np.ones(point_count))).T
    z = las.z
    plane_coeffs, _, _, _ = np.linalg.lstsq(coords, z, rcond=None)
    global_slope = math.sqrt(plane_coeffs[0] ** 2 + plane_coeffs[1] ** 2)
    print(f"[\u2713] Global Terrain Slope: {global_slope:.3f}")

    # cloth_resolution
    if point_density > 15:
        cloth_res = 0.5
    elif point_density > 5:
        cloth_res = 1.0
    else:
        cloth_res = 2.0

    # rigidness
    if global_slope > 0.3:       # very steep
        rigidness = 1
    elif global_slope > 0.1:     # moderately steep
        rigidness = 2
    else:                        # flat
        rigidness = 3

    class_thresh = 0.1

    print(f"[i] CSF Params Set: Res={cloth_res}, Rigidness={rigidness}, Thresh={class_thresh}")

    return {
        'cloth_resolution': cloth_res,
        'rigidness': rigidness,
        'class_threshold': class_thresh,
    }


def generate_dtm_with_csf_last_returns(input_laz, epsg_code, reference_dsm,
                                       csf_params, intermethod='nearest',
                                       output_path='dtm_csf.tif'):
    """
    Generate a DTM with the Cloth Simulation Filter (CSF), using only last
    returns as input and snapping the output grid to a reference DSM.

    Requires the optional ``CSF`` package (``pip install cloth-simulation-filter``).

    Args:
        input_laz (str): Path to the LAS/LAZ file.
        epsg_code (int): EPSG code of the input data (kept for API symmetry).
        reference_dsm (str): Path to a DSM whose grid/extent/CRS to match.
        csf_params (dict): Output of :func:`laz_pre_analysis`.
        intermethod (str): Interpolation method for the ground grid.
        output_path (str): Output GeoTIFF path. Default 'dtm_csf.tif'.

    Returns:
        str: The output path.
    """
    try:
        import CSF
    except ImportError as e:
        raise ImportError(
            "generate_dtm_with_csf_last_returns requires the optional 'CSF' "
            "package. Install it with: pip install cloth-simulation-filter"
        ) from e

    print("[i] Generating DTM with CSF (Using Last Returns input)...")

    with rasterio.open(reference_dsm) as src:
        transform = src.transform
        width = src.width
        height = src.height
        bounds = src.bounds
        crs = src.crs

    las = laspy.read(input_laz)

    # Filter for last returns before CSF
    mask = las.return_number == las.number_of_returns
    x_filtered = np.array(las.x)[mask]
    y_filtered = np.array(las.y)[mask]
    z_filtered = np.array(las.z)[mask]

    coords = np.vstack((x_filtered, y_filtered, z_filtered)).T

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = csf_params['cloth_resolution']
    csf.params.rigidness = csf_params['rigidness']
    csf.params.class_threshold = csf_params['class_threshold']
    csf.setPointCloud(coords)

    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)

    ground_points = coords[list(ground_indices)]

    gx = np.linspace(bounds.left, bounds.right, width)
    gy = np.linspace(bounds.bottom, bounds.top, height)
    xx, yy = np.meshgrid(gx, gy)

    dtm_grid = griddata(
        (ground_points[:, 0], ground_points[:, 1]),
        ground_points[:, 2],
        (xx, yy),
        method=intermethod
    )

    profile = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
               'width': width, 'height': height, 'transform': transform,
               'crs': crs, 'nodata': -9999}

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dtm_grid.astype(np.float32), 1)

    print("[\u2713] DTM Generated using Last Returns + CSF.")
    return output_path


def align_rasters(src_path, target_path, output_path):
    """
    Reproject/resample ``src_path`` onto the grid implied by ``target_path``
    and write the result to ``output_path`` (bilinear resampling).

    Returns:
        str: The output path.
    """
    with rasterio.open(src_path) as src:
        with rasterio.open(target_path) as target:
            transform, width, height = calculate_default_transform(
                src.crs, target.crs, target.width, target.height, *target.bounds
            )
            profile = src.profile
            profile.update(transform=transform, width=width, height=height)

            with rasterio.open(output_path, 'w', **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target.crs,
                    resampling=Resampling.bilinear
                )
    return output_path


def analyze_low_res_ndhm(ndhm_path, downscale_factor=3,
                         building_min_height=2.5,
                         height_factor=0.4, tri_factor=0.75,
                         min_height_thresh=2.0, min_tri_thresh=1.0):
    """
    Analyze a downsampled nDHM to derive adaptive height and TRI thresholds.

    Building-like cells (above ``building_min_height``) drive an average
    height estimate and an 85th-percentile slope estimate, which are scaled
    into final thresholds and clamped to safe minimums.

    Args:
        ndhm_path (str): Path to the nDHM raster.
        downscale_factor (int): Pyramid downscale factor.
        building_min_height (float): Height above which a cell is considered
            a potential building.
        height_factor (float): Multiplier applied to the average height.
        tri_factor (float): Multiplier applied to the slope estimate.
        min_height_thresh (float): Lower bound for the height threshold.
        min_tri_thresh (float): Lower bound for the TRI threshold.

    Returns:
        dict: ``{'height_threshold', 'tri_threshold'}``.
    """
    print(f"[i] Analyzing coarse-scale nDHM (downscale 1/{downscale_factor})...")

    with rasterio.open(ndhm_path) as src:
        new_height = src.height // downscale_factor
        new_width = src.width // downscale_factor

        low_res_data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )[0]

        potential_buildings = low_res_data[low_res_data > building_min_height]
        avg_height = np.mean(potential_buildings) if potential_buildings.size > 0 else 5.0

        gy, gx = np.gradient(low_res_data)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        potential_slopes = slope[low_res_data > building_min_height]
        avg_slope = np.percentile(potential_slopes, 85) if potential_slopes.size > 0 else 1.0

        print(f"[\u2713] Coarse analysis complete: AvgHeight={avg_height:.2f}, AvgSlope={avg_slope:.2f}")

        adaptive_height_thresh = max(avg_height * height_factor, min_height_thresh)
        adaptive_tri_thresh = max(avg_slope * tri_factor, min_tri_thresh)

        return {
            'height_threshold': adaptive_height_thresh,
            'tri_threshold': adaptive_tri_thresh
        }


def fxaa_like_smoothing(mask, blend_strength=0.8):
    """
    Apply an FXAA-style, edge-aware smoothing to a binary mask.

    Args:
        mask (numpy.ndarray): Binary (0/1) mask.
        blend_strength (float): 0-1; higher blends more along detected edges.

    Returns:
        numpy.ndarray: Smoothed binary mask (uint8, 0/1).
    """
    img = mask.astype(np.float32)

    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_strength = edge_strength / (edge_strength.max() + 1e-6)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    result = img * (1 - blend_strength * edge_strength) + blurred * (blend_strength * edge_strength)

    return (result > 0.5).astype(np.uint8)


def extract_building_footprints_with_alphashape(raster_path, alpha,
                                                adaptive_height_threshold,
                                                epsg_code, kernel_size=0,
                                                output_path='alpha_shape_buildings.geojson',
                                                show_plot=False):
    """
    Extract building footprints from an nDHM using an alpha shape.

    Cells above ``adaptive_height_threshold`` form a binary mask which is
    optionally cleaned with a morphological opening + FXAA smoothing
    (when ``kernel_size > 0``), converted to world coordinates, and wrapped
    in an alpha shape that is saved as GeoJSON.

    Requires the optional ``alphashape`` package. ``matplotlib`` is only
    imported when ``show_plot=True``.

    Args:
        raster_path (str): Path to the nDHM raster.
        alpha (float): Alpha value (higher = tighter / more detail).
        adaptive_height_threshold (float): Height cutoff for the mask.
        epsg_code (int): EPSG code to tag the output GeoJSON with.
        kernel_size (int): Morphological-open kernel size (0 disables).
        output_path (str): Output GeoJSON path.
        show_plot (bool): If True, show a before/after mask plot.

    Returns:
        str | None: Output path, or ``None`` if no polygon could be built.
    """
    try:
        import alphashape
    except ImportError as e:
        raise ImportError(
            "extract_building_footprints_with_alphashape requires the optional "
            "'alphashape' package. Install it with: pip install alphashape"
        ) from e

    from scipy.ndimage import binary_opening

    with rasterio.open(raster_path) as src:
        ndhm = src.read(1)
        transform = src.transform

        print(f"[\u2713] Using adaptive height threshold: {adaptive_height_threshold:.2f}m")

        mask_raw = ndhm > adaptive_height_threshold
        mask = mask_raw.copy()

        if kernel_size > 0:
            print(f"[i] Applying Pre-Alpha Morphological Opening (Kernel: {kernel_size})...")
            structure = np.ones((kernel_size, kernel_size)).astype(bool)
            mask = binary_opening(mask_raw, structure=structure)
            mask = fxaa_like_smoothing(mask)
            print("[i] Applied FXAA-style edge smoothing to the mask")

            if show_plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 8))
                plt.subplot(1, 2, 1)
                plt.title("BEFORE: Raw Height Threshold", fontsize=10)
                plt.imshow(mask_raw, cmap='gray', interpolation='none')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("AFTER: Morph + FXAA Smoothing", fontsize=10)
                plt.imshow(mask, cmap='gray', interpolation='none')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print("[i] Kernel size is 0, skipping morphological open.")

        rows, cols = np.where(mask)
        coords = np.array([transform * (col + 0.5, row + 0.5)
                           for row, col in zip(rows, cols)])

    if len(coords) < 4:
        print("Not enough points for alpha shape.")
        return None

    print(f"[i] Generating alpha shape with alpha: {alpha}")
    poly = alphashape.alphashape(coords, alpha)

    if poly is None or poly.is_empty:
        print("No polygon could be formed.")
        return None

    gdf = gpd.GeoDataFrame(geometry=[poly], crs=f"EPSG:{epsg_code}")
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"[\u2713] Alpha shape saved to {output_path}")
    return output_path


def rasterize_geojson(geojson_file, reference_raster, output_tiff, target_epsg=3857):
    """
    Rasterize a GeoJSON onto the grid of a reference raster.

    The GeoJSON is reprojected to ``target_epsg`` (default EPSG:3857, matching
    the rest of this library) before rasterization.

    Args:
        geojson_file (str): Input GeoJSON path.
        reference_raster (str): Raster whose transform/extent to match.
        output_tiff (str): Output GeoTIFF path.
        target_epsg (int): CRS to reproject the geometries into.

    Returns:
        str: The output path.
    """
    gdf = gpd.read_file(geojson_file)
    gdf = gdf.to_crs(epsg=target_epsg)

    with rasterio.open(reference_raster) as src:
        profile = src.profile.copy()
        transform = src.transform
        out_shape = (src.height, src.width)

    shapes = ((geom, 1) for geom in gdf.geometry)

    rasterized = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    profile.update(dtype='uint8', count=1)
    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(rasterized, 1)

    print("Rasterized GeoJSON saved to:", output_tiff)
    return output_tiff
