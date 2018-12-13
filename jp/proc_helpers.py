import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
import os, sys
import matplotlib.pyplot as plt
import random
from shapely.geometry import LineString, Point
from glob import glob
import multiprocessing as mp
from functools import partial


""" some dictionaries for cloud masking """
## a dictionary for cloud masking
bit_flags = {
    "pixel_qa": {
        "L47": {
            "Fill": [0],
            "Clear": [1],
            "Water": [2],
            "Cloud Shadow": [3],
            "Snow": [4],
            "Cloud": [5],
            "Low Cloud Confidence": [6],
            "Medium Cloud Confidence": [7],
            "High Cloud Confidence": [6, 7]
        },
        "L8": {
            "Fill": [0],
            "Clear": [1],
            "Water": [2],
            "Cloud Shadow": [3],
            "Snow": [4],
            "Cloud": [5],
            "Low Cloud Confidence": [6],
            "Medium Cloud Confidence": [7],
            "High Cloud Confidence": [6, 7],
            "Low Cirrus Confidence": [8],
            "Medium Cirrus Confidence": [9],
            "High Cirrus Confidence": [8, 9],
            "Terrain Occlusion": [10]
        }
    }
}

pixel_flags = {
    "pixel_qa": {
        "L47": {
            "Fill": [1],
            "Clear": [66, 130],
            "Water": [68, 132],
            "Cloud Shadow": [72, 136],
            "Snow": [80, 112, 144, 176],
            "Cloud": [96, 112, 160, 176, 224],
            "Low Cloud Confidence": [66, 68, 72, 80, 96, 112],
            "Medium Cloud Confidence": [130, 132, 136, 144, 160, 176],
            "High Cloud Confidence": [224]
        },
        "L8": {
            "Fill": [1],
            "Clear": [322, 386, 834, 898, 1346],
            "Water": [324, 388, 836, 900, 1348],
            "Cloud Shadow": [328, 392, 840, 904, 1350],
            "Snow": [336, 368, 400, 432, 848, 880, 912, 944, 1352],
            "Cloud": [352, 368, 416, 432, 480, 864, 880, 928, 944, 992],
            "Low Cloud Confidence": [322, 324, 328, 336, 352, 368, 834, 836, 840, 848, 864, 880],
            "Medium Cloud Confidence": [386, 388, 392, 400, 416, 432, 900, 904, 928, 944],
            "High Cloud Confidence": [480, 992],
            "Low Cirrus Confidence": [322, 324, 328, 336, 352, 368, 386, 388, 392, 400, 416, 432, 480],
            "Medium Cirrus Confidence": [],
            "High Cirrus Confidence": [834, 836, 840, 848, 864, 880, 898, 900, 904, 912, 928, 944, 992],
            "Terrain Occlusion": [1346, 1348, 1350, 1352]
        }
    }
}



def random_points_within(poly, num_points):
    """Generate <num_points> random points within a geometry.
    
    Parameters
    ---------------------------------
    poly: a Shapely Polygon geometry
        This is the geometry within which a random point will be generated.
    
    num_points: number of points to generate within the polygon
        See above.
    
    Usage Notes
    ---------------------------------
    This is a 'brute force' method in that a random point is generated within the *extent* of the geometry 
    until that point is within the actual geometry. For highly irregular polygons this could take a while.
    
    """
    
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points


def create_buffer_point_erase_polygon(df, buff_dist=2000, num_points=1):
    """Generate <num_points> random points within a geometry.
    
    Parameters
    ---------------------------------
    df: a GeoPandas GeoDataFrame
        This is the geometry within which a random point will be generated.
    
    buff_dist: number of points to generate within the polygon
        See above.
        
    
    Usage Notes
    ---------------------------------
    This currently will ony work for num_pts_per_poly=1
    
    """
    
        
    # generate the points. with 
    points = [random_points_within(geom, num_points=num_points)[0] for geom in df['geometry']]
    diffed = []
    for i,p in enumerate(df['geometry']):

        p_geom = points[i].buffer(buff_dist) # buffered point
        temp = p_geom.difference(p)

        diffed.append(temp)
        
    # make the list into a geodataframe
    diffed_df = gpd.GeoDataFrame({'geometry': diffed})
    
    return diffed_df

def create_buffer_point_erase_polygon2(df, buff_dist=2000):
    """Generate <num_points> random points within a geometry.
    
    Parameters
    ---------------------------------
    df: a GeoPandas GeoDataFrame
        This is the geometry within which a random point will be generated.
    
    buff_dist: number of points to generate within the polygon
        See above.
        
    
    Usage Notes
    ---------------------------------
    This currently will ony work for num_pts_per_poly=1
    
    """
    
        
    # generate the points. with 
    points = [random_points_within(geom, num_points=1)[0] for geom in df['geometry']]
    diffed = []
    for i,p in enumerate(points):

        p_geom = p.buffer(buff_dist) # buffered point
        #temp = gpd.overlay(df, p, how='intersection') # both geometries need to be multipart
        
        # iterate through the original geometries
        temp = p_geom.difference(df['geometry'][0])
        for geom in df['geometry'][1:]:
            temp = temp.difference(geom)

        diffed.append(temp)
        
    # make the list into a geodataframe
    diffed_df = gpd.GeoDataFrame({'geometry': diffed})
    
    return diffed_df

def create_buffer_point_polygon_overlay(df, buff_dist=2000, method='difference', num_points=1):
    """Generate <num_points> random points within a geometry.
    
    Parameters
    ---------------------------------
    df: a GeoPandas GeoDataFrame
        This is the geometry within which a random point will be generated.
    
    buff_dist: number of points to generate within the polygon
        See above.
        
    method: see GeoPandas overlay doc for how=keyword
    
    Usage Notes
    ---------------------------------
    This currently will ony work for num_pts_per_poly=1
    
    """
    
        
    # generate the points. with 
    points = [random_points_within(geom, num_points=num_points) for geom in df['geometry']]
    points = [item for sublist in points for item in sublist] # flatten the list
    pt_df = gpd.GeoDataFrame({'geometry': points})
    pt_df['geometry'] = pt_df.buffer(buff_dist)
    
    return gpd.overlay(pt_df, df, how=method)

def create_buffer_point_polygon_overlay_v2(df, buff_dist=2000, method='difference', num_points_fld='NUMHHtest', oid_fld='NewID'):
    """Generate <num_points> random points within a geometry.
    
    Parameters
    ---------------------------------
    df: a GeoPandas GeoDataFrame
        This is the geometry within which a random point will be generated.
    
    buff_dist: number of points to generate within the polygon
        See above.
        
    method: see GeoPandas overlay doc for how=keyword
    
    num_points_fld: field containing number of points to generate
    
    oid_fld: field containing value to assign each geometry created within a village
    
    Usage Notes
    ---------------------------------
    This currently will ony work for num_pts_per_poly=1
    
    """
    
        
    # generate the points. with 
    points_ls = []
    dfs = []
    for i,geom in enumerate(df['geometry']):
        
        # get the number of points
        num_points = int(df[num_points_fld][i])
        points = random_points_within(geom, num_points=num_points)
        #points = [item for sublist in points for item in sublist]
        newids = [df[oid_fld][i]] * num_points
        
        this_gdf = gpd.GeoDataFrame({'geometry': points, 'NewID': newids})
        dfs.append(this_gdf)
        
    pt_df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
    pt_df['geometry'] = pt_df.buffer(buff_dist)
    
    return gpd.overlay(pt_df, df, how=method)


## define a function to process an individual file
def summarize_ndvi_with_qa_file(ndvi_file, qa_file, geom, method='median'):
    
    """Summarize within geometries NDVI raster using pixel_qa.tif mask by specifying the raster file directories.
    
    Parameters
    ---------------------------------
    ndvi_file: string
        The NDVI tif file.
    
    qa_file: string
        The pixel_qa.tif file.
        
    geom: a shapely geometry
        Geometry to summarize within
    
    Usage Notes
    ---------------------------------
    """
    
    
    # get arrays... nodata will be -9999
    with rio.open(ndvi_file) as src:
        n_arr, _ = mask(src, [geom], crop=True)

    with rio.open(qa_file) as src:
        qa_arr, _ = mask(src, [geom], crop=True)

    # generate values from QA band to mask
    mask_vals = []
    mask_keys = ('Cloud Shadow', 'Cloud', 'Water')
    
    # some code to determine sensor type
    if 'LC08' in ndvi_file:
        stype = 'L8'
    else:
        stype = 'L47'
    
    # extract the mask keys
    if stype == 'L8':
        for k in mask_keys:
            mask_vals.extend(pixel_flags['pixel_qa']['L8'][k])

    else: # stype is landsat 4, 5, or 7
        for k in mask_keys:
            mask_vals.extend(pixel_flags['pixel_qa']['L47'][k])
            
            
    # set the mask_vals in n_arr to nodata value
    n_arr[np.isin(n_arr, mask_vals)] = -9999
    
    # set the saturated pixels (20000) to nodata value
    sat_value = 20000
    n_arr[n_arr == sat_value] = -9999

    # mask the array for computation
    ma_n_arr = np.ma.masked_equal(n_arr, -9999)

    # calculate value based on method
    sf = 10000.
    if method == 'mean':
        calc_val = np.ma.mean(ma_n_arr) / sf
    elif method == 'median':
        calc_val = np.ma.median(ma_n_arr) / sf
    elif method == 'max':
        calc_val = np.ma.max(ma_n_arr) / sf
    elif method == 'min':
        calc_val = np.ma.min(ma_n_arr) / sf
    else:
        calc_val = -9999
        
    
    return calc_val

## define a function to process the data
def summarize_ndvi_with_qa_dir(ndvi_dir, qa_dir, geo_df, method='median'):
    """Summarize within geometries NDVI raster using pixel_qa.tif mask by specifying the raster file directories.
    
    Parameters
    ---------------------------------
    ndvi_dir: string
        the directory containing the NDVI tif files.
    
    qa_dir: string
        The directory containing the pixel_qa.tif files
        
    geo_df: GeoPandas GeoDataFrame
        GeoDataFrame containing the geometries to summarize within
    
    Usage Notes
    ---------------------------------
    """
    
    # get the filepaths for the ndvi and pixel_qa files
    #qa_dir = '../landsat/test/qa/'
    qa_files = sorted(glob(qa_dir + '*.tif')) # sorted helps ensure the filenames match

    #ndvi_dir = '../landsat/test/ndvi/'
    ndvi_files = sorted(glob(ndvi_dir + '*.tif')) # sorted helps ensure the filenames match
    
    # do the sorting by acquisition date
    acqdates = [int(os.path.basename(f).split('_')[3]) for f in qa_files]
    sort_inds = np.argsort(acqdates)
    ndvi_files = [ndvi_files[i] for i in sort_inds]
    qa_files = [qa_files[i] for i in sort_inds]
    
    # ensure the number of files are the same
    assert len(ndvi_files) == len(qa_files)
    
    # iterate over the geometries
    all_vals = []
    
    # ensure the CRS of rasters and df match
    with rio.open(qa_files[0]) as src:
        prof = src.profile
        
    epsg_raster = int(prof['crs']['init'].split(':')[1])
    epsg_df = int(geo_df.crs['init'].split(':')[1])
    
    #print('epsg_df: {}, epsg_raster: {}'.format(epsg_df, epsg_raster))
    if epsg_raster != epsg_df:
        geo_df = geo_df.to_crs(epsg = epsg_raster)
    #print('epsg_df: {}, epsg_raster: {}'.format(geo_df.crs, epsg_raster))
        
    # do the processing
    for i,geometry in enumerate(geo_df['geometry']):
        
        #print('on geometry {} of {}'.format(i+1, geo_df.shape[0]))
        vals=[]
        # iterate over the file pairs
        for files in zip(ndvi_files, qa_files):
            
            # get the files
            ndvi_fi, qa_fi = files
            
            # calculate a value
            val = summarize_ndvi_with_qa_file(ndvi_fi, qa_fi, geometry, method=method)
            
            # store in list
            vals.append(val)
        
        # append to geometry list
        all_vals.append(vals)
        
    # append all_vals to the geodataframe
    landsat_columns = ['d_'+ os.path.basename(f).split('_')[3] for f in qa_files]
    ndvi_df = geo_df.join(pd.DataFrame(np.array(all_vals), columns=landsat_columns), how='outer')
        
    return ndvi_df.to_crs(epsg=epsg_df)

## define a function to process the data
def pp_summarize_ndvi_with_qa_dir(ndvi_dir, qa_dir, geo_df, method='median'):
    """Use parallel processing to summarize within geometries NDVI raster using pixel_qa.tif mask by specifying the raster file directories.
    
    Parameters
    ---------------------------------
    ndvi_dir: string
        the directory containing the NDVI tif files.
    
    qa_dir: string
        The directory containing the pixel_qa.tif files
        
    geo_df: GeoPandas GeoDataFrame
        GeoDataFrame containing the geometries to summarize within
    
    Usage Notes
    ---------------------------------
    """
    
    # get the filepaths for the ndvi and pixel_qa files
    #qa_dir = '../landsat/test/qa/'
    qa_files = sorted(glob(qa_dir + '*.tif')) # sorted helps ensure the filenames match

    #ndvi_dir = '../landsat/test/ndvi/'
    ndvi_files = sorted(glob(ndvi_dir + '*.tif')) # sorted helps ensure the filenames match
    
    # do the sorting by acquisition date
    acqdates = [int(os.path.basename(f).split('_')[3]) for f in qa_files]
    sort_inds = np.argsort(acqdates)
    ndvi_files = [ndvi_files[i] for i in sort_inds]
    qa_files = [qa_files[i] for i in sort_inds]
    
    # ensure the number of files are the same
    assert len(ndvi_files) == len(qa_files)
    
    # iterate over the geometries
    all_vals = []
    
    # ensure the CRS of rasters and df match
    with rio.open(qa_files[0]) as src:
        prof = src.profile
        
    epsg_raster = int(prof['crs']['init'].split(':')[1])
    epsg_df = int(geo_df.crs['init'].split(':')[1])
    
    print('epsg_df: {}, epsg_raster: {}'.format(epsg_df, epsg_raster))
    if epsg_raster != epsg_df:
        geo_df = geo_df.to_crs(epsg = epsg_raster)
    print('epsg_df: {}, epsg_raster: {}'.format(geo_df.crs, epsg_raster))
    
    ## set up the parallel processing
    nproc = max(mp.cpu_count()-2, 2)
    pool = mp.Pool(nproc)
    
    # run the code
    for files in zip(ndvi_files, qa_files):
            
        # get the files
        ndvi_fi, qa_fi = files
        
        # process in parallel
        vals = pool.map(partial(summarize_ndvi_with_qa_file, ndvi_fi, qa_fi, method=method), geo_df['geometry'])
        
        # vals should have one value for each geometry
        all_vals.append(vals)
        
    # merge the data frame to the original data frame
    landsat_columns = ['d_'+ os.path.basename(f).split('_')[3] for f in qa_files]
    ndvi_df = geo_df.join(pd.DataFrame(np.array(all_vals).T, columns=landsat_columns), how='outer')
    
    return  ndvi_df.to_crs(epsg=epsg_df)



def pp_summarize_ndvi_with_qa_file(ndvi_file, qa_file, geom, method='median', pool=None):
    
    """Summarize within geometries NDVI raster using pixel_qa.tif mask by specifying the raster file directories.
    
    Parameters
    ---------------------------------
    ndvi_file: string
        The NDVI tif file.
    
    qa_file: string
        The pixel_qa.tif file.
        
    geom: a shapely geometry
        Geometry to summarize within
        
    pool: multiprocessing Pool
        pool to farm out rasters
    
    Usage Notes
    ---------------------------------
    """
    
    
    # get arrays... nodata will be -9999
    with rio.open(ndvi_file) as src:
        n_arr, _ = mask(src, [geom], crop=True)

    with rio.open(qa_file) as src:
        qa_arr, _ = mask(src, [geom], crop=True)

    # generate values from QA band to mask
    mask_vals = []
    mask_keys = ('Cloud Shadow', 'Cloud', 'Water')
    
    # some code to determine sensor type
    if 'LC08' in ndvi_file:
        stype = 'L8'
    else:
        stype = 'L47'
    
    # extract the mask keys
    if stype == 'L8':
        for k in mask_keys:
            mask_vals.extend(pixel_flags['pixel_qa']['L8'][k])

    else: # stype is landsat 4, 5, or 7
        for k in mask_keys:
            mask_vals.extend(pixel_flags['pixel_qa']['L47'][k])
            
            
    # set the mask_vals in n_arr to nodata value
    n_arr[np.isin(n_arr, mask_vals)] = -9999
    
    # set the saturated pixels (20000) to nodata value
    sat_value = 20000
    n_arr[n_arr == sat_value] = -9999

    # mask the array for computation
    ma_n_arr = np.ma.masked_equal(n_arr, -9999)

    # calculate value based on method
    sf = 10000.
    if method == 'mean':
        calc_val = np.ma.mean(ma_n_arr) / sf
    elif method == 'median':
        calc_val = np.ma.median(ma_n_arr) / sf
    elif method == 'max':
        calc_val = np.ma.max(ma_n_arr) / sf
    elif method == 'min':
        calc_val = np.ma.min(ma_n_arr) / sf
    else:
        calc_val = -9999
        
    
    return calc_val


# a parallel processing function to farm out geometries and rasters
## define a function to process the data
# def full_pp_summarize_ndvi_with_qa_dir(ndvi_dir, qa_dir, geo_df, method='median'):
#     """Use parallel processing to summarize within geometries NDVI raster using pixel_qa.tif mask by specifying the raster file directories.
    
#     Parameters
#     ---------------------------------
#     ndvi_dir: string
#         the directory containing the NDVI tif files.
    
#     qa_dir: string
#         The directory containing the pixel_qa.tif files
        
#     geo_df: GeoPandas GeoDataFrame
#         GeoDataFrame containing the geometries to summarize within
    
#     Usage Notes
#     ---------------------------------
#     """
    
#     # get the filepaths for the ndvi and pixel_qa files
#     qa_dir = '../landsat/test/qa/'
#     qa_files = sorted(glob(qa_dir + '*.tif')) # sorted helps ensure the filenames match

#     ndvi_dir = '../landsat/test/ndvi/'
#     ndvi_files = sorted(glob(ndvi_dir + '*.tif')) # sorted helps ensure the filenames match
    
#     # do the sorting by acquisition date
#     acqdates = [int(os.path.basename(f).split('_')[3]) for f in qa_files]
#     sort_inds = np.argsort(acqdates)
#     ndvi_files = [ndvi_files[i] for i in sort_inds]
#     qa_files = [qa_files[i] for i in sort_inds]
    
#     # ensure the number of files are the same
#     assert len(ndvi_files) == len(qa_files)
    
#     # iterate over the geometries
#     all_vals = []
    
#     # ensure the CRS of rasters and df match
#     with rio.open(qa_files[0]) as src:
#         prof = src.profile
        
#     epsg_raster = int(prof['crs']['init'].split(':')[1])
#     epsg_df = int(geo_df.crs['init'].split(':')[1])
    
#     print('epsg_df: {}, epsg_raster: {}'.format(epsg_df, epsg_raster))
#     if epsg_raster != epsg_df:
#         geo_df = geo_df.to_crs(epsg = epsg_raster)
#     print('epsg_df: {}, epsg_raster: {}'.format(geo_df.crs, epsg_raster))
    
#     ####################################
#     ## set up the parallel processing ##
#     ####################################
#     max_cpus = mp.cpu_count()
#     nproc_geoms = int(max_cpus / 2)
#     nproc_rasters = max_cpus - nproc_geoms - 1
    
#     # set up the pools
#     pool_geoms = mp.Pool(nproc_geoms)
#     pool_raster = mp.Pool(nproc_rasters)
    
#     # run the code
#     for files in zip(ndvi_files, qa_files):
            
#         # get the files
#         ndvi_fi, qa_fi = files
        
#         # process in parallel
#         vals = pool.map(partial(pp_summarize_ndvi_with_qa_file, ndvi_fi, qa_fi, method=method, pool=pool_raster), geo_df['geometry'])
        
#         # vals should have one value for each geometry
#         all_vals.append(vals)
        
#     # merge the data frame to the original data frame
#     landsat_columns = ['d_'+ os.path.basename(f).split('_')[3] for f in qa_files]
#     ndvi_df = geo_df.join(pd.DataFrame(np.array(all_vals).T, columns=landsat_columns), how='outer')
    
#     return  ndvi_df.to_crs(epsg=epsg_df)

