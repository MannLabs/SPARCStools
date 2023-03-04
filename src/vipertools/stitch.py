"""
stitch
====================================

Collection of functions to perform stitching of parsed image Tiffs.

"""

from ashlar import filepattern, thumbnail, reg
from ashlar.scripts.ashlar import process_axis_flip
from skimage.filters import gaussian
from skimage.util import invert
import numpy as np
import subprocess
import sys
import skimage.exposure
import skimage.util
from PIL import Image
from tifffile import imsave
import matplotlib.pyplot as plt
import shutil
import os
import pandas as pd
import time
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import h5py

#for export to ome.zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

#for export to ome.tif
from ashlar.reg import PyramidWriter


from vipertools._custom_ashlar_funcs import  plot_edge_scatter, plot_edge_quality


#define custom FilePatternReaderRescale to use with Ashlar to allow for custom modifications to images before performing stitching
class FilePatternReaderRescale(filepattern.FilePatternReader):

    def __init__(self, path, pattern, overlap, pixel_size=1, do_rescale=False, WGAchannel = None, no_rescale_channel = "Alexa488"):
        super().__init__(path, pattern, overlap, pixel_size=pixel_size)
        self.do_rescale = do_rescale
        self.WGAchannel = WGAchannel
        self.no_rescale_channel = no_rescale_channel

    @staticmethod
    def rescale_p1_p99(img):
        img = skimage.util.img_as_float32(img)
        if img.max() > (40000/65535):
            print('True')
            _img = img.copy()
            _img[_img > (10000/65535)] = 0
            p1 = np.percentile(_img, 1)
            p99 = np.percentile(_img, 99)
        else:
            p1 = np.percentile(img, 1)
            p99 = np.percentile(img, 99)
        img = skimage.exposure.rescale_intensity(img, 
                                                  in_range=(p1, p99), 
                                                  out_range=(0, 1))
        return((img * 65535).astype('uint16'))

    @staticmethod
    def correct_illumination(img, sigma = 30, double_correct = False):
        img = skimage.util.img_as_float32(img)
        if img.max() > (40000/65535):
            print('True')
            _img = img.copy()
            _img[_img > (10000/65535)] = 0
            p1 = np.percentile(_img, 1)
            p99 = np.percentile(_img, 99)
        else:
            p1 = np.percentile(img, 1)
            p99 = np.percentile(img, 99)

        img = skimage.exposure.rescale_intensity(img, 
                                                 in_range=(p1, p99), 
                                                 out_range=(0, 1))

        #calculate correction mask
        correction = gaussian(img, sigma)
        correction = invert(correction)
        correction = skimage.exposure.rescale_intensity(correction, 
                                                        out_range = (0,1))

        correction_lows =  np.where(img > 0.5, 0, img) * correction
        img_corrected = skimage.exposure.rescale_intensity(img + correction_lows,
                                                           out_range = (0,1))

        if double_correct:
            correction_mask_highs = invert(correction)
            correction_mask_highs_02 = skimage.exposure.rescale_intensity(np.where(img_corrected < 0.5, 0, img_corrected)*correction_mask_highs)
            img_corrected_double = skimage.exposure.rescale_intensity(img_corrected - 0.25*correction_mask_highs_02)
            
            return((img_corrected_double * 65535).astype('uint16'))
        else:
            return((img_corrected * 65535).astype('uint16'))
    
    def read(self, series, c):
        img = super().read(series, c)
        if not self.do_rescale:
            return img
        elif self.do_rescale == "partial":
            if c != self.no_rescale_channel:
                return self.rescale_p1_p99(img) 
            else:
                return img
        else:
            if c == self.WGAchannel:
                return self.correct_illumination(img)
            if c == "WGAbackground":
                return self.correct_illumination(img, double_correct = True)
            else:
                return self.rescale_p1_p99(img)  

from yattag import Doc, indent

def _write_xml(path, 
              channels, 
              slidename, 
              cropped = False):
    """ Helper function to generate an XML for import of stitched .tifs into BIAS.

    Parameters
    ----------
    path 
        path to where the exported images are written
    channels
        list of the channel names written out
    slidename
        string indicating the name underwhich the files were written out
    cropped
        boolean value indicating if the stitched images were written out cropped or not.
    """

    if cropped:
        image_paths = [slidename + "_"+x+'_cropped.tif' for x in channels]
    else:
        image_paths = [slidename + "_"+x+'.tif' for x in channels]

    doc, tag, text = Doc().tagtext()
    
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
    doc.asis(xml_header)
    with tag("BIAS", version = "1.0"):
        with tag("channels"):
            for i, channel in enumerate(channels):
                with tag("channel", id = str(i+1)):
                    with tag("name"):
                        text(channel)
        with tag("images"):
            for i, image_path in enumerate(image_paths):
                with tag("image", url=str(image_path)):
                    with tag("channel"):
                        text(str(i+1))

    result = indent(
        doc.getvalue(),
        indentation = ' '*4,
        newline = '\r\n'
    )

    #write to file
    write_path = os.path.join(path, slidename + ".XML")
    with open(write_path, mode ="w") as f:
        f.write(result)
 

def generate_thumbnail(input_dir, 
                       pattern, 
                       outdir, 
                       overlap, 
                       name, 
                       stitching_channel = "DAPI", 
                       export_examples = False,
                       do_intensity_rescale = True):
    """
    Function to generate a scaled down thumbnail of stitched image. Can be used for example to 
    get a low resolution overview of the scanned region to select areas for exporting high resolution 
    stitched images.

    Parameters
    ----------
    input_dir
        Path to the folder containing exported TIF files named with the following nameing convention: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif". 
        These images can be generated for example by running the vipertools.parse.parse_phenix() function.
    pattern
        Regex string to identify the naming pattern of the TIFs that should be stitched together. 
        For example: "Row1_Well2_{channel}_zstack3_r{row:03}_c{col:03}.tif". 
        All values in {} indicate thos which are matched by regex to find all matching tifs.
    outdir
        path indicating where the stitched images should be written out
    overlap
        value between 0 and 1 indicating the degree of overlap that was used while recording data at the microscope.
    name
        string indicating the slidename that is added to the stitched images generated
    export_examples
        boolean value indicating if individual example tiles should be exported in addition to performing thumbnail generation.
    do_intensity_rescale
        boolean value indicating if the rescale_p1_P99 function should be applied before stitching or not.
    """
    
    start_time = time.time()
    
    #read data 
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap)
    slide.do_rescale = do_intensity_rescale
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #generate stitched thumbnail on which to determine cropping params
    channel_id = list(slide.metadata.channel_map.values()).index(stitching_channel)
    _thumbnail = thumbnail.make_thumbnail(slide, channel=channel_id, scale=0.05)

    _thumbnail = Image.fromarray(_thumbnail)
    _thumbnail.save(os.path.join(outdir, name + '_thumbnail_'+stitching_channel+'.tif'))
    
    end_time = time.time() - start_time
    print("Thumbnail generated for channel DAPI, pipeline completed in ", str(end_time/60), "minutes.")

    if export_examples:
        #generate preview images for each slide
        channels = list(slide.metadata.channel_map.values())

        all_files = os.listdir(input_dir)
        all_files = [x for x in all_files if pattern[0:10] in x]

        #creat output directory
        outdir_examples = os.path.join(outdir, 'example_images')
        if not os.path.exists(outdir_examples):
            os.makedirs(outdir_examples)

        #get 10 randomly selected DAPI files
        _files = [x for x in all_files if stitching_channel in x]
        _files = random.sample(_files, 10)

        for channel in channels:
            for file in _files:
                file = file.replace(stitching_channel, channel)
                img = Image.open(os.path.join(input_dir, file))
                corrected = slide.rescale_p1_p99(img)
                imsave(os.path.join(outdir_examples, file), corrected)

        print("Example Images Exported.")
    
def generate_stitched(input_dir, 
                      slidename,
                      pattern,
                      outdir,
                      overlap = 0.1,
                      max_shift = 30, 
                      stitching_channel = "Alexa488",
                      no_rescale_channel = None,
                      crop = {'top':0, 'bottom':0, 'left':0, 'right':0},
                      plot_QC = True,
                      filetype = [".tif"],
                      WGAchannel = None,
                      do_intensity_rescale = True,
                      export_XML = True):
    
    """
    Function to generate a scaled down thumbnail of stitched image. Can be used for example to 
    get a low resolution overview of the scanned region to select areas for exporting high resolution 
    stitched images.

    Parameters
    ----------
    input_dir
        Path to the folder containing exported TIF files named with the following nameing convention: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif". 
        These images can be generated for example by running the vipertools.parse.parse_phenix() function.
    pattern
        Regex string to identify the naming pattern of the TIFs that should be stitched together. 
        For example: "Row1_Well2_{channel}_zstack3_r{row:03}_c{col:03}.tif". 
        All values in {} indicate thos which are matched by regex to find all matching tifs.
    outdir
        path indicating where the stitched images should be written out
    overlap
        value between 0 and 1 indicating the degree of overlap that was used while recording data at the microscope.
    max_shift
        value indicating the maximum threshold for tile shifts. Default value in ashlar is 15.
    name
        string indicating the slidename that is added to the stitched images generated
    stitching_channel
        string indicating the channel name on which the stitching should be calculated. the positions for each tile calculated in this channel will be 
        passed to the other channels. 
    crop
        dictionary of the form {'top':0, 'bottom':0, 'left':0, 'right':0} indicating how many pixels (based on a generated thumbnail, 
        see vipertools.stitch.generate_thumbnail) should be cropped from the final image in each indicated dimension. Leave this set to default 
        if no cropping should be performed.
    plot_QC
        boolean value indicating if QC plots should be generated
    filetype
        list containing any of [".tif", ".ome.zarr", ".ome.tif"] defining to which type of file the stiched results should be written. If more than one 
        element all export types will be generated in the same output directory.
    WGAchannel
        string indicating the name of the WGA channel in case an illumination correction should be performed on this cahhenl
    do_intensity_rescale
        boolean value indicating if the rescale_p1_P99 function should be applied before stitching or not. Alternatively partial then it will only
        rescale those channels provided as a list in rescale_channels
    export_XML
        boolean value. If true than an xml is exported when writing to .tif which allows for the import into BIAS.
    """
    start_time = time.time()
    
    #read data 
    print("performing stichting with ", str(overlap), " overlap.")
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap)
    
    # Turn on the rescaling
    slide.do_rescale = do_intensity_rescale
    slide.WGAchannel = WGAchannel

    if do_intensity_rescale == "partial":
        if no_rescale_channel != None:
            no_rescale_channel_id = list(slide.metadata.channel_map.values()).index(no_rescale_channel)
            slide.no_rescale_channel = no_rescale_channel_id
        else:
            sys.exit("do_intensity_rescale set to partial but not channel passed for which no rescaling should be done.")
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #get dictionary position of channel
    channel_id = list(slide.metadata.channel_map.values()).index(stitching_channel)

    #generate aligner to use specificed channel for stitching
    print("performing stitching on channel ", stitching_channel, "with id number ", str(channel_id))
    aligner = reg.EdgeAligner(slide, channel=channel_id, filter_sigma=0, verbose=True, do_make_thumbnail=False, max_shift = max_shift)
    aligner.run()  

    #generate some QC plots
    if plot_QC:
        plot_edge_scatter(aligner, outdir)
        plot_edge_quality(aligner, outdir)
        #reg.plot_edge_scatter(aligner)
        print("need to implement this here. TODO")

    aligner.reader._cache = {} #need to empty cache for some reason

    #generate stitched file
    mosaic_args = {}
    mosaic_args['verbose'] = True
    mosaic_args['channels'] = list(slide.metadata.channel_map.keys())

    mosaic = reg.Mosaic(aligner, 
                        aligner.mosaic_shape, 
                        **mosaic_args
                        )

    mosaic.dtype = np.uint16

    if "return_array" in filetype:
        print("not saving positions")
    else:
        #write out positions to csv
        positions = aligner.positions
        np.savetxt(os.path.join(outdir, slidename + "_tile_positions.tsv"), positions, delimiter="\t")

    if "return_array" in filetype:

        print("Returning array instead of saving to file.")
        mosaics = []
        
        for channel in tqdm(mosaic.channels):
            mosaics.append(mosaic.assemble_channel(channel = channel))

        merged_array = np.array(mosaics)
        merged_array = merged_array.astype("uint16")

        end_time = time.time() - start_time
        print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")
        
        #get channel names
        channels = []
        for channel in  slide.metadata.channel_map.values():
            channels.append(channel)

        return(merged_array, channels)

    elif ".tif" in filetype:
        print("writing results to one large tif.")

        mosaics = []
        for channel in tqdm(mosaic.channels):
            mosaics.append(mosaic.assemble_channel(channel = channel))
            
        #actually perform cropping
        if np.sum(list(crop.values())) > 0:
            print('Merged image will be cropped to the specified cropping parameters: ', crop)
            merged_array = np.array(mosaics)

            cropping_factor = 20.00   #this is based on the scale that was used in the thumbnail generation
            _, x, y = merged_array.shape
            top = int(crop['top'] * cropping_factor)
            bottom = int(crop['bottom'] * cropping_factor)
            left = int(crop['left'] * cropping_factor)
            right = int(crop['right'] * cropping_factor)
            cropped = merged_array[:, slice(top, x-bottom), slice(left, y-right)]

            #return(merged_array, cropped)
            #write to tif for each channel
            for i, channel in enumerate(slide.metadata.channel_map.values()):
                (print('writing to file: ', channel))
                im = Image.fromarray(cropped[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'_cropped.tif'))
            
            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = True)
        
        else:
            merged_array = np.array(mosaics)
            for i, channel in enumerate(slide.metadata.channel_map.values()):
                im = Image.fromarray(merged_array[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'.tif'))

            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = False)
    
    elif "ome.tif" in filetype:
        print("writing results to ome.tif")
        path = os.path.join(outdir, slidename + ".ome.tiff")
        writer = PyramidWriter([mosaic], path, scale=5, tile_size=1024, peak_size=1024, verbose=True)
        writer.run()

    elif "ome.zarr" in filetype:
        print("writing results to ome.zarr")

        if 'mosaics' not in locals():
            mosaics = []
            for channel in tqdm(mosaic.channels):
                mosaics.append(mosaic.assemble_channel(channel = channel))

        path = os.path.join(outdir, slidename + ".ome.zarr")
        loc = parse_url(path, mode="w").store
        group = zarr.group(store = loc)
        axes = "cyx"

        channel_colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
        #chek if length of colors is enough for all channels in slide otherwise loop through n times
        while len(slide.metadata.channel_map.values()) > len(channel_colors):
            channel_colors = channel_colors + channel_colors

        group.attrs["omero"] = {
            "name":slidename + ".ome.zarr",
            "channels": [{"label":channel, "color":channel_colors[i], "active":True} for i, channel in enumerate(slide.metadata.channel_map.values())]
        }
        print(np.array(mosaics).shape)    
        write_image(np.array(mosaics), group = group, axes = axes, storage_options=dict(chunks=(1, 1024, 1024)))
    
    end_time = time.time() - start_time
    print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")