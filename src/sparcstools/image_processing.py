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

def _downsample_img(img, N=2):
    """
    Function to downsample a single image equivalent to NxN binning using the mean between pixels. 
    Takes a numpy array image as input and returns a numpy array.

    Parameters
    ----------
    img : array
        image to downsample
    N : int, default = 2
        number of pixels that should be binned together using mean between pixels
    """
    downsampled = xr.DataArray(img, dims=['x', 'y']).coarsen(x= N, y= N).mean()
    downsampled = (downsampled/downsampled.max()*65535).astype("uint16")
    return(downsampled)

def downsample_img(img_path, N = 2, copy = False, outdir = None):

    """
    Function to downsample a single image equivalent to NxN binning using the mean between pixels. 
    Overwrites the original image(!), do not run multiple times on the same image.

    Parameters
    ----------
    img_path : str
        string indicating the file path to the .tif file which should be downsampled.
    N : int, default = 2
        number of pixels that should be binned together using mean between pixels
    """
    
    #read image to array
    img = imread(img_path)
    
    #downsample and convert back to uint16
    _downsampled = _downsample_img(img)
    
    #write out (overwrite image location)
    if copy:
        file_name = os.path.basename(img_path)
        imsave(os.path.join(outdir, file_name), _downsampled)
    else:
        imsave(img_path, _downsampled)

def downsample_folder(folder_path, num_threads = 20, file_ending = (".tif", ".tiff"), N = 2):
    """
    Multi-Threaded Function to downsample image equivalent to 2x2 binning. Overwrites original images! Do not run multiple times.
    Output is saved as uint16.
    
    Parameters
    ----------
    folder_path : str
        string indicating the folder containing all the image files that should be downsampled
    num_threads : int
        number of threads for multithreading
    file_ending: str | (str, str) 
        string or tuple of strings indicating which file ending the script should filter for in the indicated folder
    N : int
        number of pixels that should be binned together
    """
    #get all files names from the given directory and filter on correct file endings
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, x) for x in files if x.endswith(file_ending)]
    
    #perform multithreaded downsampling of each individual image
    #images are overwritten!
    print("beginning downsampling with ", num_threads, "threads...")
    with Pool(max_workers=num_threads) as pool:
        pool.map(partial(downsample_img, N), files)
        
    print("Downsampling of all images completed.")

def downsample_folder_copy_images(folder_path, outdir, num_threads = 20, file_ending = (".tif", ".tiff"), N = 2):
    """
    Multi-Threaded Function to downsample image equivalent to 2x2 binning. Duplicates images before downsampling! Do not run multiple times.
    Output is saved as uint16.
    
    Parameters
    ----------
    folder_path : str
        string indicating the folder containing all the image files that should be downsampled
    outdir : str
        string indicating the folder where the downsampled images should be generated
    num_threads : int
        number of threads for multithreading
    file_ending: str | (str, str) 
        string or tuple of strings indicating which file ending the script should filter for in the indicated folder
    N : int
        number of pixels that should be binned together
    """

    #get all files names from the given directory and filter on correct file endings
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, x) for x in files if x.endswith(file_ending)]
    
    #perform multithreaded downsampling of each individual image
    #images are overwritten!
    print("beginning downsampling with ", num_threads, "threads...")
    print(f"saving results to new directory {outdir}")

    #create output directory if it does not already exist
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    with Pool(max_workers=num_threads) as pool:
        pool.map(partial(downsample_img, N = N, copy = True, outdir = outdir), files)
        
    print("Downsampling of all images completed.")
