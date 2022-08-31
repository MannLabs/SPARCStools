"""
image processing
====================================

Contains functions to perform standard image processing steps, e.g. downsampling.
"""

import os
from skimage.io import imread, imsave
import xarray as xr
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool

def downsample_img(img_path):

    """
    Function to downsample a single image equivalent to 2x2 binning using the mean between pixels. 
    Overwrites the original image(!), do not run multiple times on the same image.

    Parameters
    ----------
    img_path
        string indicating the file path to the .tif file which should be downsampled.
    """
    
    #read image to array
    img = imread(img_path)
    
    #downsample and convert back to uint16
    _downsampled = xr.DataArray(img, dims=['x', 'y']).coarsen(x=2, y=2).mean().astype("uint16")
    
    #write out (overwrite image location)
    imsave(img_path, _downsampled)
    
def downsample_folder(folder_path, num_threads = 20, file_ending = (".tif", ".tiff")):
    """
    Multi-Threaded Function to downsample image equivalent to 2x2 binning. Overwrites original images! Do not run multiple times.
    Output is saved as uint16.
    
    Parameters
    ----------
    folder_path
        string indicating the folder containing all the image files that should be downsampled
    num_threads
        int, number of threads for multithreading
    file_ending
        string or tuple of strings indicating which file ending the script should filter for in the indicated folder
    
    """
    #get all files names from the given directory and filter on correct file endings
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, x) for x in files if x.endswith((".tif", ".tiff"))]
    
    #perform multithreaded downsampling of each individual image
    #images are overwritten!
    print("beginning downsampling with ", num_threads, "threads...")
    with Pool(max_workers=num_threads) as pool:
        pool.map(partial(downsample_img), files)
        
    print("Downsampling of all images completed.")