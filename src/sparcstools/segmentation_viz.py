"""
segmentation_viz: visualization of segmentation results
====================================

Collection of functions to easily check quality of segmentation results.
"""

#import library
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

#define functions for this
def plot_segmentation_wells(segmentation, cellids_keep):
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

    path = outpath
    loc = parse_url(path, mode="w").store
    group = zarr.group(store = loc)

    channel_colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

    group.attrs["omero"] = {
        "name": "test.ome.zarr",
        "channels": [{"label":channel, "color":channel_colors[i], "active":True} for i, channel in enumerate(channels)]
    }

    write_image(image, group = group, axes = "cyx", storage_options=dict(chunks=(1, 1024, 1024)))

    #add segmentation label
    labels_grp = group.create_group("labels")
    labels_grp.attrs["labels"] = segmentation_names

    for seg, name in zip(segmentation, segmentation_names):
    # write the labels to /labels
        # the 'labels' .zattrs lists the named labels data
        label_grp = labels_grp.create_group(name)
        # need 'image-label' attr to be recognized as label
        label_grp.attrs["image-label"] = {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
            ]
        }

        print(label_grp)
        write_image(seg, label_grp, axes="cyx")

def add_seg(segmentation,  #list of all sets you want to visualize
            segmentation_names, #list of what each cell set should be called
            outpath):
    
    """
    Function to add segmentation to existing zarr file without any labels. This will create an additional group in the ome.zarr 
    termed labels to which all segementations can be added.
    
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

    #add segmentation label
    labels_grp = group.create_group("labels")
    labels_grp.attrs["labels"] = segmentation_names

    for seg, name in zip(segmentation, segmentation_names):
    # write the labels to /labels
        # the 'labels' .zattrs lists the named labels data
        label_grp = labels_grp.create_group(name)
        # need 'image-label' attr to be recognized as label
        label_grp.attrs["image-label"] = {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
            ]
        }

        print(label_grp)
        write_image(seg, label_grp, axes="cyx")

def add_additional_seg(segmentation,  #list of all sets you want to visualize
                        segmentation_names, #list of what each cell set should be called
                        outpath):
    
    """
    Function to add segmentation to existing zarr file that alread contains the group labels. 

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

    #add segmentation label
    labels_grp = group["labels"]
    labels_grp.attrs["labels"] = segmentation_names

    for seg, name in zip(segmentation, segmentation_names):
    # write the labels to /labels
        # the 'labels' .zattrs lists the named labels data
        label_grp = labels_grp.create_group(name)
        # need 'image-label' attr to be recognized as label
        label_grp.attrs["image-label"] = {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
            ]
        }

        print(label_grp)
        write_image(seg, label_grp, axes="cyx")
    
def sparcspy_add_seg(stitched_path, seg_path, nuclei = True, cytosol = True):
    """
    Custom function to easily add finished segmentation to input stitching results within the SPARCSpy framework.

    Parameters
    ----------
    stitched_path
        path indicating the location of the ome.zarr file of the stitched data.

    seg_path
        string indicating the location of the SPARCSpy project folder (it will select the segmentation itself)
    nuclei
        boolean value if the segmentation contains a nuclear segmentation and this should be appended to the ome.zarr
    cytosol
        boolean value if the segmentation contains a cytosolic segmentation and this should be appended to the ome.zarr

    """

    segs = []
    seg_labels = []

    segmentation = os.path.join(seg_path, "segmentation", "segmentation.h5")
    hdf5 = h5py.File(segmentation, "r") 

    if nuclei:
        nuclei_seg = hdf5["labels"][0, :, :]
        segs.append(nuclei_seg)
        seg_labels.append("nuclei")

    if cytosol:
        cytosol_seg = hdf5["labels"][1, :, :]
        segs.append(cytosol_seg)
        seg_labels.append("cytosol")

    path = stitched_path
    loc = parse_url(path, mode="w").store
    group = zarr.group(store = loc)

    #add segmentation label
    labels_grp = group.create_group("labels")
    labels_grp.attrs["labels"] = seg_labels

    for seg, name in zip(segs, seg_labels):
    # write the labels to /labels
        # the 'labels' .zattrs lists the named labels data
        label_grp = labels_grp.create_group(name)
        # need 'image-label' attr to be recognized as label
        label_grp.attrs["image-label"] = {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
            ]
        }

        print(label_grp)
        write_image(seg, label_grp, axes="cyx")