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


def parse_phenix(phenix_dir,
                 flatfield_exported = True,
                 parallel = False):

    """ Function to automatically rename TIFS exported from Harmony into a format where row and well ID as well as Tile position are indicated in the file name.
    :param phenix_dir: Path indicating the exported harmony files to parse.
    :type arg: str
    :param `*args`: The variable arguments are used for ...
    :param `**kwargs`: The keyword arguments are used for ...
    :vartype arg: str
    """
    
    phenix_dir = path
    flatfield_exported = True
    parallel = False

    #start timer
    start_time = time.time()

    #generate directories
    if flatfield_exported:
        input_dir = os.path.join(phenix_dir, 'Images', "flex")
        index_file = os.path.join(input_dir, 'index.flex.xml')
    else:
        input_dir = os.path.join(phenix_dir, 'Images')
        index_file = os.path.join(input_dir, 'Index.idx.xml')

    lookuppath = os.path.join(input_dir, 'lookup.csv')
    outfile = os.path.join(input_dir, 'parsed_index.txt')
    outdir = os.path.join(phenix_dir, 'parsed_images')

    #if output directory does not exist create
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #extract channel names from xml file
    cmd = """grep -E -m 20 '<ChannelName>|<ChannelID>' '""" + index_file + """'"""
    results = subprocess.check_output(
        cmd, shell=True).decode("utf-8").strip().split('\r\n')

    results = [x.strip() for x in results]
    channel_ids = [x.split('>')[1].split('<')[0] for x in results if x.startswith('<ChannelID')]
    channel_names = [x.split('>')[1].split('<')[0].replace(' ', '') for x in results if x.startswith('<ChannelName')]

    channel_ids = list(set(channel_ids))
    channel_ids.sort()
    channel_names = channel_names[0:len(channel_ids)]

    #Parse Phenix XML file to get file name information
    cmd = """grep -E '<URL>|<PositionX Unit="m">|<PositionY Unit="m">' '""" + \
        index_file + """' > '""" + outfile + """'"""
    subprocess.check_output(cmd, shell=True)

    images = []
    x_positions = []
    y_positions = []

    with open(outfile) as fp:
        Lines = fp.readlines()
        for line in Lines:
            if flatfield_exported:
                _line = line.replace('flex_', "").split('>')[1].split('<')[0]
            else:
                _line = line.split('>')[1].split('<')[0]
            if line.strip().startswith("<URL>"):
                images.append(_line)
            elif line.strip().startswith("<PositionX"):
                x_positions.append(float(_line))
            elif line.strip().startswith("<PositionY"):
                y_positions.append(float(_line))
            else:
                print('error')

    #get plate and well Ids as well as channel information
    rows = [int(x[0:3].replace('r', '')) for x in images]
    wells = [int(x[3:6].replace('c', '')) for x in images]
    channels = [x.split('-')[1][2:3] for x in images]
    zstack = [int(x[9:12].replace('p', '')) for x in images]

    df = pd.DataFrame({"Image_files": images,
                       "Row": rows,
                       "Well": wells,
                       "Zstack":zstack,
                       "X": x_positions,
                       "Y": y_positions,
                       "X_pos": None,
                       "Y_pos": None,
                       "Channel": channels,
                       "new_file_name": None})

    #get X positions
    X_values = df.X.value_counts().sort_index()
    X_values = X_values.index

    for i, x in enumerate(X_values):
        df.loc[df.X == x, 'X_pos'] = str(i).zfill(3)

    #get y positions
    Y_values = df.Y.value_counts().sort_index()
    Y_values = Y_values.index

    for i, y in enumerate(Y_values):
        df.loc[df.Y == y, 'Y_pos'] = str(i).zfill(3)

    #rename channels with proper channel label
    lookup = pd.DataFrame({'id': list(channel_ids),
                           'label': list(channel_names)})
    print('Channel lookup table:')
    print(lookup)

    df = df.replace(dict(zip(lookup.id, lookup.label)))

    #generate new file names
    for i in range(df.shape[0]):
        _row = df.loc[i, :]
        name = "Row{}_Well{}_{}_zstack{}_r{}_c{}.tif".format(
            _row.Row, _row.Well, _row.Channel,_row.Zstack, _row.Y_pos, _row.X_pos)
        name = name
        df.loc[i, 'new_file_name'] = name

    #write tables to file for future use
    lookup.to_csv(lookuppath)

    if flatfield_exported:
        df.Image_files = ['flex_' + x for x in df.Image_files]

    print(df.head(5))
    #copy files from a to b

    filelist = os.listdir(outdir) # dir is your directory path
    number_files = len(filelist)

    if parallel:
        print('parallel')
       # Parallel(n_jobs = 10)(delayed(shutil.copyfile(os.path.join(input_dir, old), os.path.join(outdir, new)))) for old, new in zip(df.Image_files.tolist(), df.new_file_name.tolist())
    else:
        for old, new in tqdm(zip(df.Image_files.tolist(), df.new_file_name.tolist()), 
                         total = len(df.new_file_name.tolist())):
            old_path = os.path.join(input_dir, old)
            new_path = os.path.join(outdir, new)
            shutil.copyfile(old_path, new_path)

    endtime = time.time() - start_time
    print("Parsing Phenix data completed, total time was ", str(endtime/60), "minutes.")
    
def parse_Alexa488(phenix_dir):
    #start timer
    start_time = time.time()
    
    outdir = os.path.join(phenix_dir, 'parsed_images')
    
    for file_name in os.listdir(outdir):
        # construct full file path
        if "Alexa488" in file_name:
            source = os.path.join(outdir, file_name)
            destination = os.path.join(outdir, file_name.replace("Alexa488", "WGAbackground"))
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)

    print("Parsing Alexa488 data completed, total time was ", str(endtime/60), "minutes.")
    

class FilePatternReaderRescale(filepattern.FilePatternReader):

    def __init__(self, path, pattern, overlap, pixel_size=1, do_rescale=False):
        super().__init__(path, pattern, overlap, pixel_size=pixel_size)
        self.do_rescale = do_rescale

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
        else:
            if c == "Alexa488":
                return self.correct_illumination(img)
            else:
                return self.rescale_p1_p99(img)  

def generate_thumbnail(input_dir, pattern, outdir, overlap, name):
    """Generate thumbnail to determine cropping parameters using Ashlar"""
    
    start_time = time.time()
    
    #read data 
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap)
    slide.do_rescale = True
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #generate stitched thumbnail on which to determine cropping params
    _thumbnail = thumbnail.make_thumbnail(slide, channel='DAPI')

    _thumbnail = Image.fromarray(_thumbnail)
    _thumbnail.save(os.path.join(outdir, name + '_thumbnail_DAPI.tif'))
    
    end_time = time.time() - start_time
    print("Thumbnail generated for channel DAPI, pipeline completed in ", str(end_time/60), "minutes.")

    #generate preview images for each slide
    channels = list(slide.metadata.channel_map.values())

    all_files = os.listdir(input_dir)
    all_files = [x for x in all_files if pattern[0:10] in x]

    #creat output directory
    outdir_examples = os.path.join(outdir, 'example_images')
    if not os.path.exists(outdir_examples):
        os.makedirs(outdir_examples)

    #get 10 randomly selected DAPI files
    _files = [x for x in all_files if 'DAPI' in x]
    _files = random.sample(_files, 10)

    for channel in channels:
        for file in _files:
            file = file.replace('DAPI', channel)
            img = Image.open(os.path.join(input_dir, file))
            corrected = slide.rescale_p1_p99(img)
            imsave(os.path.join(outdir_examples, file), corrected)

    print("Example Images Exported.")
    
def generate_stitched(input_dir, 
                      slidename,
                      pattern,
                      outdir,
                      overlap,
                      crop = {'top':0, 'bottom':0, 'left':0, 'right':0}):
    
    start_time = time.time()
    
    #read data 
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap)
    
    # Turn on the rescaling
    slide.do_rescale = True
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #perform actual alignment
    aligner = reg.EdgeAligner(slide, channel='Alexa488', filter_sigma=0, verbose=True, do_make_thumbnail=False)
    aligner.run()

    #generate some QC plots
    #reg.plot_edge_quality(aligner, img=aligner.reader.thumbnail, save_fig = True)
    #reg.plot_edge_scatter(aligner)


    aligner.reader._cache = {}

    #generate stitched file
    mosaic_args = {}
    mosaic_args['channels'] = list(slide.metadata.channel_map.values())
    mosaic_args['verbose'] = True

    mosaic = reg.Mosaic(aligner, 
                        aligner.mosaic_shape, 
                        os.path.join(outdir, slidename + '_{channel}.tif'),
                        **mosaic_args
                        )

    mosaic.dtype = np.uint16

    #actually perform cropping
    if np.sum(list(crop.values())) > 0:
        print('Merged image will be cropped to the specified cropping parameters: ', crop)
        merged = mosaic.run(mode='return')
        merged_array = np.array(merged)
        
        cropping_factor = 20.02
        _, x, y = merged_array.shape
        top = int(crop['top'] * cropping_factor)
        bottom = int(crop['bottom'] * cropping_factor)
        left = int(crop['left'] * cropping_factor)
        right = int(crop['right'] * cropping_factor)
        cropped = merged_array[:, slice(top, x-bottom), slice(left, y-right)]
        
        #return(merged_array, cropped)
        #write to tif for each channel
        for i, channel in enumerate(mosaic_args['channels']):
            (print('writing to file: ', channel))
            im = Image.fromarray(cropped[i].astype('uint16'))#ensure that type is uint16
            im.save(os.path.join(outdir, slidename + "_"+channel+'_cropped.tif'))
    else:
        #mosaic.run(mode='write')
        merged = mosaic.run(mode='return')
        merged_array = np.array(merged)
        for i, channel in enumerate(mosaic_args['channels']):
            im = Image.fromarray(merged_array[i].astype('uint16'))#ensure that type is uint16
            im.save(os.path.join(outdir, slidename + "_"+channel+'.tif'))
    
    end_time = time.time() - start_time
    print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")