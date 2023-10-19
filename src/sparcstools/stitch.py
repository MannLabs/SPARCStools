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
import gc

#for export to ome.zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

#for export to ome.tif
from ashlar.reg import PyramidWriter


from sparcstools._custom_ashlar_funcs import  plot_edge_scatter, plot_edge_quality


#define custom FilePatternReaderRescale to use with Ashlar to allow for custom modifications to images before performing stitching
class FilePatternReaderRescale(filepattern.FilePatternReader):

    def __init__(self, path, pattern, overlap, pixel_size=1, do_rescale=False, WGAchannel = None, no_rescale_channel = "Alexa488", rescale_range = (1, 99)):
        super().__init__(path, pattern, overlap, pixel_size=pixel_size)
        self.do_rescale = do_rescale
        self.WGAchannel = WGAchannel
        self.no_rescale_channel = no_rescale_channel
        self.rescale_range = rescale_range

    @staticmethod
    def rescale_p1_p99(img, rescale_range = (1, 99)):
        img = skimage.util.img_as_float32(img)
        cutoff1, cutoff2 = rescale_range
        if img.max() > (40000/65535):
            _img = img.copy()
            _img[_img > (10000/65535)] = 0
            p1 = np.percentile(_img, cutoff1)
            p99 = np.percentile(_img, cutoff2)
        else:
            p1 = np.percentile(img, cutoff1)
            p99 = np.percentile(img, cutoff2)
        img = skimage.exposure.rescale_intensity(img, 
                                                  in_range=(p1, p99), 
                                                  out_range=(0, 1))
        return((img * 65535).astype('uint16'))

    @staticmethod
    def correct_illumination(img, sigma = 30, double_correct = False, rescale_range = (1, 99)):
        cutoff1, cutoff2 = rescale_range
        img = skimage.util.img_as_float32(img)
        if img.max() > (40000/65535):
            print('True')
            _img = img.copy()
            _img[_img > (10000/65535)] = 0
            p1 = np.percentile(_img, cutoff1)
            p99 = np.percentile(_img, cutoff2)
        else:
            p1 = np.percentile(img, cutoff1)
            p99 = np.percentile(img, cutoff2)

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
            if c not in self.no_rescale_channel:
                return self.rescale_p1_p99(img, rescale_range = self.rescale_range) 
            else:
                return img
        else:
            if c == self.WGAchannel:
                return self.correct_illumination(img, rescale_range = self.rescale_range)
            if c == "WGAbackground":
                return self.correct_illumination(img, double_correct = True, rescale_range = self.rescale_range)
            else:
                return self.rescale_p1_p99(img, self.rescale_range)  

from yattag import Doc, indent

def _write_xml(path, 
              channels, 
              slidename, 
              cropped = False):
    """ Helper function to generate an XML for import of stitched .tifs into BIAS.

    Parameters
    ----------
    path : str
        path to where the exported images are written
    channels : [str]
        list of the channel names written out
    slidename : str
        string indicating the name underwhich the files were written out
    cropped : bool
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

def _reorder_list(original_list, index_list):
    if len(original_list) != len(index_list):
        raise ValueError("The length of the original list and index list must be the same.")
    
    reordered_list = [None] * len(original_list)
    
    for i, index in enumerate(index_list):
        reordered_list[index] = original_list[i]
    
    return reordered_list

def generate_thumbnail(input_dir, 
                       pattern, 
                       outdir, 
                       overlap, 
                       name, 
                       stitching_channel = "DAPI", 
                       export_examples = False,
                       do_intensity_rescale = True, 
                       rescale_range = (1, 99),
                       scale = 0.05):
    """
    Function to generate a scaled down thumbnail of stitched image. Can be used for example to 
    get a low resolution overview of the scanned region to select areas for exporting high resolution 
    stitched images.

    Parameters
    ----------
    input_dir : str
        Path to the folder containing exported TIF files named with the following nameing convention: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif". 
        These images can be generated for example by running the sparcstools.parse.parse_phenix() function.
    pattern : str
        Regex string to identify the naming pattern of the TIFs that should be stitched together. 
        For example: "Row1_Well2_{channel}_zstack3_r{row:03}_c{col:03}.tif". 
        All values in {} indicate those which are matched by regex to find all matching tifs.
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
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap, rescale_range = rescale_range)
    slide.do_rescale = do_intensity_rescale
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #generate stitched thumbnail on which to determine cropping params
    channel_id = list(slide.metadata.channel_map.values()).index(stitching_channel)
    _thumbnail = thumbnail.make_thumbnail(slide, channel=channel_id, scale=scale)

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
                      crop = {'top':0, 'bottom':0, 'left':0, 'right':0},
                      plot_QC = True,
                      filetype = [".tif"],
                      WGAchannel = None,
                      do_intensity_rescale = True,
                      rescale_range = (1, 99),
                      no_rescale_channel = None,
                      export_XML = True,
                      return_tile_positions = True,
                      channel_order = None):
    
    """
    Function to generate a stitched image.

    Parameters
    ----------
    input_dir : str
        Path to the folder containing exported TIF files named with the following nameing convention: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif". 
        These images can be generated for example by running the sparcstools.parse.parse_phenix() function.
    slidename : str
        string indicating the slidename that is added to the stitched images generated
    pattern : str
        Regex string to identify the naming pattern of the TIFs that should be stitched together. 
        For example: "Row1_Well2_{channel}_zstack3_r{row:03}_c{col:03}.tif". 
        All values in {} indicate those which are matched by regex to find all matching tifs.
    outdir : str
        path indicating where the stitched images should be written out
    overlap : float between 0 and 1
        value between 0 and 1 indicating the degree of overlap that was used while recording data at the microscope.
    max_shift: int
        value indicating the maximum threshold for tile shifts. Default value in ashlar is 15. In general this parameter does not need to be adjusted but it is provided
        to give more control.
    stitching_channel : str
        string indicating the channel name on which the stitching should be calculated. the positions for each tile calculated in this channel will be 
        passed to the other channels. 
    crop
        dictionary of the form {'top':0, 'bottom':0, 'left':0, 'right':0} indicating how many pixels (based on a generated thumbnail, 
        see sparcstools.stitch.generate_thumbnail) should be cropped from the final image in each indicated dimension. Leave this set to default 
        if no cropping should be performed.
    plot_QC : bool
        boolean value indicating if QC plots should be generated
    filetype : [str]
        list containing any of [".tif", ".ome.zarr", ".ome.tif"] defining to which type of file the stiched results should be written. If more than one 
        element is present in the list all export types will be generated in the same output directory.
    WGAchannel : str
        string indicating the name of the WGA channel in case an illumination correction should be performed on this channel
    do_intensity_rescale : bool
        boolean value indicating if the rescale_p1_P99 function should be applied before stitching or not. Alternatively partial then those channels listed in no_rescale_channel will 
        not be rescaled.
    no_rescale_channel : None | [str]
        either None or a list of channel strings on which no rescaling before stitching should be performed.
    export_XML
        boolean value. If true then an xml is exported when writing to .tif which allows for the import into BIAS.
    return_tile_positions : bool | default = True
        boolean value. If true and return_type != "return_array" the tile positions are written out to csv.
    channel_order : None | [int]
        if None do nothing, if list of ints is supplied remap channel order
    """
    start_time = time.time()

    #convert relativ paths into absolute paths
    outdir = os.path.abspath(outdir)

    #read data 
    print("performing stitching with ", str(overlap), " overlap.")
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap, rescale_range=rescale_range)
    
    # Turn on the rescaling
    slide.do_rescale = do_intensity_rescale
    slide.WGAchannel = WGAchannel

    if do_intensity_rescale == "partial":
        if no_rescale_channel != None:
            no_rescale_channel_id = []
            for _channel in no_rescale_channel:
                no_rescale_channel_id.append(slide.metadata.channel_map.values().index(_channel))
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

    if channel_order is None:
        _channels = mosaic.channels
    else:
        _channels = _reorder_list(mosaic.channels, channel_order)

    if "return_array" in filetype:
        print("not saving positions")
    else:
        if return_tile_positions:
            #write out positions to csv
            positions = aligner.positions
            np.savetxt(os.path.join(outdir, slidename + "_tile_positions.tsv"), positions, delimiter="\t")
        else:
            print("not saving positions")
    
    if "return_array" in filetype:

        print("Returning array instead of saving to file.")
        mosaics = []
        
        for channel in tqdm(_channels):
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
        #define shape of output image

        n_channels = len(mosaic.channels)
        x, y = mosaic.shape

        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(outdir)

        # initialize tempmmap array to save segmentation results to
        mosaics = tempmmap.array((n_channels, x, y), dtype=np.uint16)
        for i, channel in tqdm(enumerate(_channels), total = n_channels):
            mosaics[i, :, :] = mosaic.assemble_channel(channel = channel)
            
        #actually perform cropping
        if np.sum(list(crop.values())) > 0:
            print('Merged image will be cropped to the specified cropping parameters: ', crop)
            merged_array = np.array(mosaics)
            
            #manual garbage collection tp reduce memory footprint
            del mosaics
            gc.collect()

            cropping_factor = 20.00   #this is based on the scale that was used in the thumbnail generation
            _, x, y = merged_array.shape
            top = int(crop['top'] * cropping_factor)
            bottom = int(crop['bottom'] * cropping_factor)
            left = int(crop['left'] * cropping_factor)
            right = int(crop['right'] * cropping_factor)
            cropped = merged_array[:, slice(top, x-bottom), slice(left, y-right)]

            #manual garbage collection tp reduce memory footprint
            del merged_array
            gc.collect()

            #write to tif for each channel
            for i, channel in enumerate(slide.metadata.channel_map.values()):
                (print('writing to file: ', channel))
                im = Image.fromarray(cropped[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'_cropped.tif'))
            
            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = True)

            #manual garbage collection tp reduce memory footprint
            del cropped
            gc.collect()

        else:
            merged_array = np.array(mosaics)
            
            #manual garbage collection tp reduce memory footprint
            del mosaics
            gc.collect()

            for i, channel in enumerate(slide.metadata.channel_map.values()):
                im = Image.fromarray(merged_array[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'.tif'))

            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = False)

            del merged_array
            
    elif "ome.tif" in filetype:
        print("writing results to ome.tif")
        path = os.path.join(outdir, slidename + ".ome.tiff")
        writer = PyramidWriter([mosaic], path, scale=5, tile_size=1024, peak_size=1024, verbose=True)
        writer.run()

    elif "ome.zarr" in filetype:
        print("writing results to ome.zarr")

        if 'mosaics' not in locals():
            #define shape of output image

            n_channels = len(_channels)
            x, y = mosaic.shape

            from alphabase.io import tempmmap
            TEMP_DIR_NAME = tempmmap.redefine_temp_location(outdir)

            # initialize tempmmap array to save segmentation results to
            mosaics = tempmmap.array((n_channels, x, y), dtype=np.uint16)
            for i, channel in tqdm(enumerate(_channels), total = n_channels):
                mosaics[i, :, :] = mosaic.assemble_channel(channel = channel)

        path = os.path.join(outdir, slidename + ".ome.zarr")

        #delete file if it already exists
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("Outfile already existed, deleted.")

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
        write_image(mosaics, group = group, axes = axes, storage_options=dict(chunks=(1, 1024, 1024)))
   
    #perform garbage collection manually to free up memory as quickly as possible
    print("deleting old variables")
    if "mosaic" in locals():
        del mosaic
        gc.collect()

    if "mosaics" in locals():
        del mosaics
        gc.collect()

    if "TEMP_DIR_NAME" in locals():
        shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)
        del TEMP_DIR_NAME
        gc.collect()

    end_time = time.time() - start_time
    print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")

def _stitch(x, 
            outdir, 
            overlap, 
            stitching_channel,
            rescale_range = (0.1, 99.9),
            crop = {'top':0, 'bottom':0, 'left':0, 'right':0}, 
            output_filetype = [".tif"]):
    """Helper Function for stitch all.
    """
    
    pattern, slidename, path = x
    generate_stitched(path,
                        slidename,
                        pattern,
                        outdir,
                        overlap,
                        crop = crop ,
                        stitching_channel = stitching_channel,
                        filetype = output_filetype,
                        rescale_range = rescale_range,
                        plot_QC = False,
                        export_XML = False, 
                        return_tile_positions = False)

def prepare_stitch_slurm_job(path, 
                            stitching_channel = "mCherry",
                            zstack_value = 1,
                            rescale_range = (0.1, 99.9),
                            overlap = 0.1,
                            jobs_per_file = 24
                            ):
    """Function to generate all required output to execute an arrayed batch job to stitch all 
    wells contained within a Harmony directory.

    Once run navigate to the generated folder (slurm_jobs/stitch_all/logs) in the main harmony project directory
    and run "sbatch ../run.sh".

    The sbatch array automatically limits the maximum number of running jobs at a time to 20.

    Parameters
    ----------
    path : str
        Folder containing the exported Harmony output generated with SPARCStools.
    stitching_channel : str, optional
        String indicating which channel should be stitched on. Defaults to "mCherry".
    zstack_value : int, optional
        Integer indicating which zstack level stitching should be performed on. Defaults to 1.
    rescale_range : tuple, optional
        Percentage range for rescaling images before stitching them. Defaults to (0.1, 99.9).
    overlap : float, optional
        Tile overlap as fraction. Defaults to 0.1.
    jobs_per_file : int, optional
        How many stitching executions should be executed per slurm job. Too few jobs will generate too much overhead and become inefficient. Defaults to 24.
    """
    
    #define paths
    outdir_slurm = os.path.join(path, "slurm_jobs", "stitch_all")
    outdir = os.path.join(path, "stitched_wells")
    input_path = os.path.join(path, "well_sorted")
    logs_dir = os.path.join(outdir_slurm, "logs")

    #create directories
    if not os.path.isdir(outdir_slurm):
        os.makedirs(outdir_slurm)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    #write processing file based on parameters
    f = open(f"{outdir_slurm}/processing.py", "w")
    f.write("from sparcstools.stitch import generate_stitched \n")
    f.write("import sys \n")
    f.write("import os \n")
    f.write("pattern = sys.argv[1] \n")
    f.write("slidename = sys.argv[2] \n")
    f.write("pattern = sys.argv[1] \n")
    f.write("path = sys.argv[3] \n")
    f.write(f"outdir = \"{outdir}\" \n")
    f.write(f"stitching_channel = \"{stitching_channel}\" \n")
    f.write(f"zstack_value = {zstack_value} \n")
    f.write(f"rescale_range = {rescale_range} \n")
    f.write(f"overlap = {overlap} \n")
    f.write("generate_stitched(path, slidename, pattern, outdir, overlap, stitching_channel = stitching_channel, crop = {'top':0, 'bottom':0, 'left':0, 'right':0}, plot_QC = False, filetype = [\".tif\"], do_intensity_rescale = True, rescale_range = rescale_range, export_XML = False, return_tile_positions = False) \n")
    f.close()

    print("processing.py generated")

    #write config file for sbatch
    import re

    files = os.listdir(input_path)
    files =[x for x in files if x not in [".DS_Store"]]
    wells = np.unique([re.search("Row.._Well..", x).group() for x in files])
    _files = os.listdir(os.path.join(input_path, wells[0]))
    _files = [x for x in _files if x.endswith(".tif")]
    timepoints = np.unique([re.search("^Timepoint...", x).group() for x in _files])

    to_process = []
    for timepoint in timepoints:
        for well in wells:
            pattern = f"{timepoint}_{well}_" +"{channel}_"+"zstack"+str(zstack_value).zfill(3)+"_r{row:03}_c{col:03}.tif"
            slidename = f"{timepoint}_{well}"
            path = f"{input_path}/{well}"
            
            to_process.append((pattern, slidename, path))

    to_process = pd.DataFrame(to_process)
    to_process.columns = ["pattern", "slidename", "path"]
    to_process.index.name = "ArrayTaskID"
    to_process.to_csv(f"{outdir_slurm}/array_inputs.csv", sep = "\t")
    print("array_inputs.csv generated.")

    #get number of jobs to add to batch job file
    n_jobs = int(np.ceil(to_process.shape[0]/jobs_per_file))

    #write sbatch file based on parameters
    f = open(f"{outdir_slurm}/run.sh", "w")

    f.write("#!/bin/bash -l \n")
    f.write("#SBATCH --job-name=stitching \n")
    f.write("#SBATCH -o \"./run%A-%a.out.%j\" \n")
    f.write("#SBATCH -e \"./run%A-%a.err.%j\" \n")
    f.write("#SBATCH -D ./ \n")
    f.write("#SBATCH --nodes=1 \n")
    f.write("#SBATCH --tasks-per-node=1 \n")
    f.write("#SBATCH --cpus-per-task=24 \n")
    f.write("#SBATCH --time=24:00:00 \n")
    f.write(f"#SBATCH --array=1-{n_jobs}%20 \n\n")

    f.write(f"PER_TASK={jobs_per_file}\n")
    f.write("# Calculate the starting and ending values for this task based\n")
    f.write("# on the SLURM task and the number of runs per task.\n")
    f.write("START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))\n")
    f.write("END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))\n")

    f.write(f"#Specify the path to the config file \nconfig={outdir_slurm}/array_inputs.csv\n\n")

    f.write("module load cuda/11.0  \n")
    f.write("module load jdk/8  \n")
    f.write("module list  \n \n")

    f.write("conda activate stitching  \n")
    f.write("echo $CONDA_DEFAULT_ENV \n\n")
    f.write("conda list \n")

    f.write("for (( run=$START_NUM; run<=END_NUM; run++ )); do\n")
    f.write("  #Extract parameters for the current $run \n")
    f.write("  pattern=$(awk -v ArrayTaskID=$run '$1==ArrayTaskID {print $2}' $config) \n")
    f.write("  slidename=$(awk -v ArrayTaskID=$run '$1==ArrayTaskID {print $3}' $config) \n")
    f.write("  path=$(awk -v ArrayTaskID=$run '$1==ArrayTaskID {print $4}' $config) \n \n")

    f.write("  echo \"calling processing script with input parameter: {$pattern} {$slidename} {$path} \"  \n")
    f.write("  srun python ../processing.py $pattern $slidename $path \n")
    f.write("done\n")

    f.close()
    print("batch job file generated.")