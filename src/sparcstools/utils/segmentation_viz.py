"""
segmentation_viz: visualization of segmentation results
====================================

Collection of functions to easily check quality of segmentation results.
"""

#import library
import numpy as np
import os
import zarr
import h5py
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels, write_label_metadata

#define functions for this
def plot_segmentation_wells(segmentation, cellids_keep):
    """Helper function which generates a masked array where only the selected cell_ids are kept and all other values are set to 0.

    Parameters
    ----------
    segmentation : np.array
        array containing the complete segmentation
    cellids_keep : list
        list containing the cell_ids that should be retained in the segmentation mask
    """
    all_classes = set(segmentation.flatten())
    cellids_keep = set(cellids_keep)

    inverse = list(all_classes - cellids_keep)
    inverse = np.array(inverse)

    mask_values = np.isin(segmentation, inverse, invert = False)
    masked = np.ma.masked_array(segmentation, mask=mask_values)
    masked = masked.filled(0)

    return(masked)

def write_zarr_with_seg(image, 
                        segmentation,  #list of all sets you want to visualize
                        segmentation_names, #list of what each cell set should be called
                        outpath, 
                        channels =["Channel1", "channel2", "channel3"],):
    """Generate an ome.zarr from an image file and segmentation masks.

    Parameters
    ----------
    image : np.array
        image to be added to ome.zarr as numpy.array
    segmentation : [np.array()]
        list of np.arrays containing the segmentation masks that should be added to the ome.zarr
    segmentation_names : [str()]
        list of strings containing what each segmentation present in `segmentation` should be called
    outpath : path
        string indicating where the generated ome.zarr should be written
    channels : [str()] | default = ["Channel1", "channel2", "channel3"]
        list of strings indicating what each channel in the supplied image should be called.
    """

    path = outpath
    loc = parse_url(path, mode="w").store
    group = zarr.group(store = loc)

    channel_colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

    group.attrs["omero"] = {
        "name": "segmentation.ome.zarr",
        "channels": [{"label":channel, "color":channel_colors[i], "active":True} for i, channel in enumerate(channels)]
    }

    write_image(image, group = group, axes = "cyx", storage_options=dict(chunks=(1, 1024, 1024)))

    #add segmentation labels
    for seg, name in zip(segmentation, segmentation_names):
        write_labels(labels = seg.astype("uint16"), group = group, name = name, axes = "cyx")
        write_label_metadata(group = group, name = f"labels/{name}", colors = [{"label-value": 0, "rgba": [0, 0, 0, 0]}])

def add_seg(segmentation,  #list of all sets you want to visualize
            segmentation_names, #list of what each cell set should be called
            outpath):
    
    """
    Function to add segmentation to existing zarr file.
    
    Parameters
    ----------
    segmentation
        list of segmentations that should be added to the ome.zarr file. Need to follow the dimension standard: cyx
    segmentation_names
        list of strings indicating the label that should be applied to each segmentation
    outpath
        string indicating the location of the existing ome.zarr file to which the segmentation results should be appended.
    """

    path = outpath
    loc = parse_url(path, mode="w").store
    group = zarr.group(store = loc)

    for seg, name in zip(segmentation, segmentation_names):
        write_labels(labels = seg.astype("uint16"), group = group, name = name, axes = "cyx")
        write_label_metadata(group = group, name = f"labels/{name}", colors = [{"label-value": 0, "rgba": [0, 0, 0, 0]}])
 
def sparcspy_add_seg(stitched_path, project_location, nuclei = True, cytosol = True):
    """
    Add segmentations generated in SPARCSpy easily to an exisiting ome.zarr or generate a new one.

    Parameters
    ----------
    stitched_path
        path indicating where the ome.zarr containing the stitched data is located or if it does not yet exist should be created.

    project_location
        string indicating the location of the SPARCSpy project folder (it will select the segmentation itself)
    nuclei
        boolean value if the segmentation contains a nuclear segmentation and this should be appended to the ome.zarr
    cytosol
        boolean value if the segmentation contains a cytosolic segmentation and this should be appended to the ome.zarr
    """

    segs = []
    seg_labels = []

    segmentation = os.path.join(project_location, "segmentation", "segmentation.h5")
    hdf5 = h5py.File(segmentation, "r") 

    if nuclei:
        nuclei_seg = hdf5["labels"][0, :, :]
        segs.append(nuclei_seg)
        seg_labels.append("nuclei")

    if cytosol:
        cytosol_seg = hdf5["labels"][1, :, :]
        segs.append(cytosol_seg)
        seg_labels.append("cytosol")

    #check if the ome.zarr already exists if not create

    if not os.path.isfile(stitched_path):
        print("no output file found will extract imaging data and generate a new ome.zarr")
        channels = hdf5["channels"][:, :, :]

        write_zarr_with_seg(channels,
                            segs,  #list of all sets you want to visualize
                            seg_labels, #list of what each cell set should be called
                            stitched_path)
    else:
        add_seg(segs, seg_labels, stitched_path)