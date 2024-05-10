"""
stitch
====================================

Collection of functions to perform stitching of parsed image Tiffs.

"""
import sys
import shutil
import os
import numpy as np

from ashlar import thumbnail
from ashlar.reg import EdgeAligner, Mosaic
from ashlar.scripts.ashlar import process_axis_flip

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from alphabase.io import tempmmap
from alphabase.io.tempmmap import create_empty_mmap, redefine_temp_location, mmap_array_from_path

from sparcstools.image_processing import rescale_image
from sparcstools.base.parallelized_ashlar import ParallelEdgeAligner, ParallelMosaic
from sparcstools.base.ashlar_plotting import plot_edge_scatter, plot_edge_quality
from sparcstools.base.filereaders import FilePatternReaderRescale, BioformatsReaderRescale
from sparcstools.base.filewriters import write_xml, write_tif, write_ome_zarr

class Stitcher:

    def __init__(self, 
                 input_dir: str, 
                 slidename: str, 
                 outdir: str,
                 stitching_channel: str,
                 pattern: str,
                 overlap: float = 0.1,
                 max_shift: float = 30,
                 filter_sigma: int = 0,
                 do_intensity_rescale: bool = True,
                 rescale_range: tuple = (1, 99),
                 channel_order: [str] = None,
                 reader_type = FilePatternReaderRescale, 
                 orientation: dict = {'flip_x': False, 'flip_y': True},
                 plot_QC: bool = True,
                 overwrite: bool = False,
                 cache: str = None,
                 ) -> None:
        
        self.input_dir = input_dir
        self.slidename = slidename
        self.outdir = outdir
        self.stitching_channel = stitching_channel

        #stitching settings
        self.pattern = pattern
        self.overlap = overlap
        self.max_shift = max_shift
        self.filter_sigma = filter_sigma

        #image rescaling
        if do_intensity_rescale == "full_image":
            self.rescale_full_image = True
            self.do_intensity_rescale = True
        
        else:
            self.do_intensity_rescale = do_intensity_rescale
            self.rescale_full_image = False
        self.rescale_range = rescale_range

        #setup reader for images
        self.orientation = orientation
        self.reader_type = reader_type

        #workflow setup
        self.plot_QC = plot_QC
        self.overwrite = overwrite
        self.channel_order = channel_order
        self.cache = cache

        self.initialize_outdir()

    def initialize_outdir(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print("Output directory created at: ", self.outdir)
        else:
            if self.overwrite:
                print(f"Output directory  at {self.outdir} already exists, overwriting.")
                shutil.rmtree(self.outdir)
                os.makedirs(self.outdir)
            else:
                raise FileExistsError(f"Output directory at {self.outdir} already exists. Set overwrite to True to overwrite the directory.")
    
    def get_channel_info(self):

        #get channel names
        self.channel_lookup = self.reader.metadata.channel_map
        self.channel_names = list(self.reader.metadata.channel_map.values())
        self.channels = list(self.reader.metadata.channel_map.keys())
        self.stitching_channel_id = list(self.channel_lookup.values()).index(self.stitching_channel)

    def setup_rescaling(self):
        #set up rescaling
        if self.do_intensity_rescale:
            
            self.reader.no_rescale_channel = []

            #if all channels should be rescaled the same initialize dictionary with all channels
            if type(self.rescale_range) is tuple:
                self.rescale_range = {k:self.rescale_range for k in self.channel_names}

            #check if all channels are in dictionary for rescaling
            rescale_channels = list(self.rescale_range.keys())

            #make sure all channels provided in lookup dictionary are in the experiment
            if not set(rescale_channels).issubset(set(self.channel_names)):
                raise ValueError("The rescale_range dictionary contains a channel not found in the experiment.")
            
            #check if we have any channels missing in the rescale_range dictionary
            missing_channels = set.difference(set(self.channel_names), set(rescale_channels))
            
            if len(missing_channels) > 0:
                Warning("The rescale_range dictionary does not contain all channels in the experiment. This may lead to unexpected results. For the missing channels rescaling will be turned off.")

                missing_channels = set.difference(self.channel_names, rescale_channels)
                for missing_channel in missing_channels:
                    self.rescale_range[missing_channel] = (0, 1)

                self.reader.no_rescale_channel = [list(self.channel_names).index(missing_channel) for missing_channel in missing_channels]

            #lookup channel names and match them with channel ids to return a new dict whose keys are the channel ids
            rescale_range_ids = {list(self.channel_names).index(k):v for k,v in self.rescale_range.items()}
            self.reader.do_rescale = True
            self.reader.rescale_range = rescale_range_ids #update so that the lookup can occur correctly

        else:
            self.reader.no_rescale_channel = []
            self.reader.do_rescale = False
            self.reader.rescale_range = None

        print(self.rescale_range)
        print(self.reader.rescale_range)
             
    def reorder_channels(self):
        if self.channel_order is None:
            self.channels = self.channels
        else:
            print("current channel order: ", self.channels)

            channels = []
            for channel in self.channel_order:
                self.channels.append(self.channels.index(channel))
            print("new channel order", channels)
            
            self.channels = channels

    def initialize_reader(self):
        
        if self.reader_type == FilePatternReaderRescale:
            self.reader = self.reader_type(self.input_dir, self.pattern, self.overlap, rescale_range = self.rescale_range)
        elif self.reader_type == BioformatsReaderRescale:
            print("THIS NEEDS TO BE IMPLEMENTED HERE")
        #setup correct orientation of slide (this depends on microscope used to generate the data)
        process_axis_flip(self.reader, flip_x = self.orientation['flip_x'], flip_y = self.orientation['flip_y'])

        #setup rescaling
        self.get_channel_info()
        self.setup_rescaling()

    def save_positions(self):
        positions = self.aligner.positions
        np.savetxt(os.path.join(self.outdir, self.slidename + "_tile_positions.tsv"), positions, delimiter="\t")
    
    def generate_thumbnail(self, scale = 0.05):
        self.initialize_reader()
        self.thumbnail = thumbnail.make_thumbnail(self.reader, channel=self.stitching_channel_id, scale=scale)

        #rescale thumbnail to 0-1 range
        self.thumbnail = rescale_image(self.thumbnail, self.rescale_range[self.stitching_channel])

    def initialize_aligner(self):

        aligner = EdgeAligner(self.reader, 
                                channel=self.stitching_channel_id, 
                                filter_sigma=self.filter_sigma, 
                                verbose=True, 
                                do_make_thumbnail=False, 
                                max_shift = self.max_shift)
        return(aligner)
    
    def perform_alignment(self):
        
        #intitialize reader for getting individual image tiles
        self.initialize_reader()
        
        print(f"performing stitching on channel {self.stitching_channel} with id number {self.stitching_channel_id}")
        self.aligner = self.initialize_aligner()
        self.aligner.run()  

        if self.plot_QC:
            fig = plot_edge_scatter(self.aligner, self.outdir)
            fig = plot_edge_quality(self.aligner, self.outdir)
        
        if self.save_positions:
            self.save_positions()
        
        self.aligner.reader._cache = {} #need to empty cache for some reason

        print("Alignment complete.")
    
    def initialize_mosaic(self):
        mosaic = Mosaic(self.aligner, 
                                 self.aligner.mosaic_shape, 
                                 verbose = True,
                                 channels = self.channels
                                 )
        return(mosaic)
    
    def assemble_mosaic(self):
        
        #get dimensions of assembled final mosaic
        n_channels = len(self.mosaic.channels)
        x, y = self.mosaic.shape
        
        # initialize tempmmap array to save assemled mosaic to
        # if no cache is specified the tempmmap will be created in the outdir

        if self.cache is None:
            TEMP_DIR_NAME = redefine_temp_location(self.outdir)
        else:
            TEMP_DIR_NAME = redefine_temp_location(self.cache)
            
        self.assembled_mosaic = tempmmap.array((n_channels, x, y), dtype=np.uint16)

        #assemble each of the channels
        for i, channel in tqdm(enumerate(self.channels), total = n_channels):
            self.assembled_mosaic[i, :, :] = self.mosaic.assemble_channel(channel = channel, out = self.assembled_mosaic[i, :, :])

            if self.rescale_full_image: 
                #warning this has not been tested for memory efficiency
                print("Rescaling entire input image to 0-1 range using percentiles specified in rescale_range.")
                self.assembled_mosaic[i, :, :] = rescale_image(self.assembled_mosaic[i, :, :], self.rescale_range[channel])

    def generate_mosaic(self):
        
        #reorder channels
        self.reorder_channels()

        self.mosaic = self.initialize_mosaic()

        #ensure dtype is set correctly
        self.mosaic.dtype = np.uint16
        self.assemble_mosaic()

    def stitch(self):
        self.perform_alignment()
        self.generate_mosaic()
    
    def write_tif(self, export_xml = True):
        filenames = []
        for i, channel in enumerate(self.channel_names):
            filename = os.path.join(self.outdir, f"{self.slidename}_{channel}.tif")
            filenames.append(filename)
            write_tif(filename, self.assembled_mosaic[i, :, :])
        
        if export_xml:
            write_xml(filenames, self.channel_names, self.slidename)

    def write_ome_zarr(self, downscaling_size = 4, n_downscaling_layers = 4, chunk_size = (1, 1024, 1024)):

        filepath = os.path.join(self.outdir, f"{self.slidename}.ome.zarr")
        write_ome_zarr(filepath, 
                       self.assembled_mosaic, 
                       self.channels, 
                       self.slidename,
                       overwrite = self.overwrite, 
                       downscaling_size = downscaling_size,
                       n_downscaling_layers = n_downscaling_layers,
                       chunk_size = chunk_size)
        
    def write_thumbnail(self):
        #calculate thumbnail if this has not already been done
        if "thumbnail" not in self.__dict__:
            self.generate_thumbnail()

        filename = os.path.join(self.outdir, self.slidename + '_thumbnail_'+self.stitching_channel+'.tif')
        write_tif(filename, self.thumbnail)

class ParallelStitcher(Stitcher):
    
    def __init__(self, 
                input_dir: str, 
                slidename: str, 
                outdir: str,
                stitching_channel: str,
                pattern: str,
                overlap: float = 0.1,
                max_shift: float = 30,
                filter_sigma: int = 0,
                do_intensity_rescale: bool = True,
                rescale_range: tuple = (1, 99),
                plot_QC: bool = True,
                WGAchannel: str = None,
                channel_order: [str] = None,
                overwrite: bool = False,
                reader_type = FilePatternReaderRescale, 
                orientation = {'flip_x': False, 'flip_y': True},
                cache: str = None,
                threads: int = 20
                ) -> None:
        
        super().__init__(input_dir, 
                        slidename, 
                        outdir,
                        stitching_channel,
                        pattern,
                        overlap,
                        max_shift,
                        filter_sigma,
                        do_intensity_rescale,
                        rescale_range,
                        channel_order,
                        reader_type, 
                        orientation,
                        plot_QC,
                        overwrite,
                        cache,
                        )
        
        self.threads = threads
    
    def initialize_aligner(self):
        aligner = ParallelEdgeAligner(self.reader, 
                                    channel=self.stitching_channel_id, 
                                    filter_sigma=self.filter_sigma, 
                                    verbose=True, 
                                    do_make_thumbnail=False, 
                                    max_shift = self.max_shift,
                                    n_threads = self.threads)
        return(aligner)
    
    def initialize_mosaic(self):
        mosaic = ParallelMosaic(self.aligner, 
                                self.aligner.mosaic_shape, 
                                verbose = True,
                                channels = self.channels,
                                n_threads = self.threads
                                )
        return(mosaic)
    
    def assemble_channel(self, args):
        channel, out, i, hdf5_path = args
        self.mosaic.assemble_channel_parallel(channel = channel, out = out, ch_index = i, hdf5_path = hdf5_path)

        if self.rescale_full_image: 
            
            #reconnect to memory mapped array
            image = mmap_array_from_path(hdf5_path)
            
            #warning this has not been tested for memory efficiency
            print("Rescaling entire input image to 0-1 range using percentiles specified in rescale_range.")
            image[i, :, :] = rescale_image(image[i, :, :], self.rescale_range[channel])
            
            del image

    def assemble_mosaic(self):
        
        #get dimensions of assembled final mosaic
        n_channels = len(self.mosaic.channels)
        x, y = self.mosaic.shape

        hdf5_path = create_empty_mmap((n_channels, x, y), dtype=np.uint16)
        self.assembled_mosaic = mmap_array_from_path(hdf5_path)

        #assemble each of the channels
        args = []
        for i, channel in enumerate(self.channels):
            args.append((channel, self.assembled_mosaic[i, :, :], i, hdf5_path))  
        tqdm_args = dict(
            file=sys.stdout,
            desc=f"assembling mosaic",
            total=len(self.channels),
        )
        #threading over channels is safe as the channels are written to different postions in the hdf5 file and do not interact with one another
        #threading over the writing of a single channel is not safe and leads to inconsistent results
        workers = np.min([self.threads, len(self.channels)])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(self.assemble_channel, args), **tqdm_args))
    
    def write_tif_parallel(self, export_xml = True):
        
        filenames = []
        args = []
        for i, channel in enumerate(self.channel_names):
            filename = os.path.join(self.outdir, f"{self.slidename}_{channel}.tif")
            filenames.append(filename)
            args.append((filename, i))
        
        tqdm_args = dict(
            file=sys.stdout,
            desc=f"writing tif files",
            total=len(self.channels),
        )

        #define helper function to execute in threadpooler
        def _write_tif(args):
            filename, ix = args
            write_tif(filename, self.assembled_mosaic[ix, :, :])

        #threading over channels is safe as the channels are written to different files
        workers = np.min([self.threads, len(self.channels)])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(_write_tif, args), **tqdm_args))
        
        #write_tif(filename, self.assembled_mosaic[i, :, :])
        
        if export_xml:
            write_xml(filenames, self.channel_names, self.slidename)