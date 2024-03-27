"""
parse
====================================

Contains functions to parse imaging data into a usable formats for downstream pipelines.
"""

import time
import sys
import os
import subprocess
import random
import numpy as np
from tqdm import tqdm
import shutil
import glob

import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import re
import numpy as np

from tifffile import imread, imwrite

def _get_child_name(elem):
    return(elem.split("}")[1])

class PhenixParser:

    def __init__(self, experiment_dir, flatfield_exported = True, export_symlinks = True, compress_rows = False, compress_cols = False) -> None:
        self.experiment_dir = experiment_dir
        self.export_symlinks = export_symlinks
        self.flatfield_status = flatfield_exported
        self.compress_rows = compress_rows
        self.compress_cols = compress_cols

        if self.compress_rows:
            print("The rows found in the phenix layout will be compressed into one row after parsing the images, r and c indicators will be adjusted accordingly.")
        if self.compress_cols:
            print("The wells found in the phenix layout will be compressed into one column after parsing the images, r and c indicators will be adjusted accordingly.")

        self.xml_path = self.get_xml_path()

        self.image_dir = self.get_input_dir()
        self.channel_lookup = self.get_channel_metadata(self.xml_path)
    
    def get_xml_path(self):
        if self.flatfield_status:
            index_file = os.path.join(self.experiment_dir, "Images", 'Index.ref.xml')
        else:
            index_file = os.path.join(self.experiment_dir, 'Index.idx.xml')
        
        #perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return(index_file)
    
    def get_input_dir(self):
        if self.flatfield_status:
            input_dir = os.path.join(self.experiment_dir, 'Images', "flex")   
        else:
            input_dir = os.path.join(self.experiment_dir, 'Images')   
        
        #perform sanity check if file exists else exit
        if not os.path.isdir(input_dir):
            sys.exit(f"Can not find directory containing images to parse: {input_dir}")

        return(input_dir)

    def define_outdir(self):

        self.outdir = f"{self.experiment_dir}/parsed_images"   

        # if output directory did not exist create it
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)  
       
    def get_channel_metadata(self, xml_path) ->  pd.DataFrame:

        index_file = xml_path
        
        #get channel names and ids and generate a lookup table
        cmd = """grep -E -m 20 '<ChannelName>|<ChannelID>' '""" + index_file + """'"""
        results = subprocess.check_output(
            cmd, shell=True).decode("utf-8").strip().split('\r\n')

        results = [x.strip() for x in results]
        channel_ids = [x.split('>')[1].split('<')[0] for x in results if x.startswith('<ChannelID')]
        channel_names = [x.split('>')[1].split('<')[0].replace(' ', '') for x in results if x.startswith('<ChannelName')]

        channel_ids = list(set(channel_ids))
        channel_ids.sort()
        channel_names = channel_names[0:len(channel_ids)]

        lookup = pd.DataFrame({'id': list(channel_ids),
                            'label': list(channel_names)})
        
        print(f"Experiment contains the following image channels: ")
        print(lookup, "\n")

        #save lookup file to csv
        lookup.to_csv(f"{self.experiment_dir}/channel_lookuptable.csv")
        print(f"Channel Lookup table saved to file at {self.experiment_dir}/channel_lookuptable.csv\n")

        self.channel_names = channel_names
        return(lookup)
        
    def read_phenix_xml(self, xml_path):
        
        #initialize lists to save results into
        rows = []
        cols = []
        fields = []
        planes = []
        channel_ids = []
        channel_names = []
        flim_ids = []
        timepoints = []
        x_positions = []
        y_positions = []
        times = []

        #extract information from XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for i, child in enumerate(root):        
            if _get_child_name(child.tag) == "Images":
                images = root[i]

        for i, image in enumerate(images):
            for ix, child in enumerate(image):
                tag = _get_child_name(child.tag)
                if  tag == "Row":
                    rows.append(child.text)
                if  tag == "Col":
                    cols.append(child.text)
                if  tag == "FieldID":
                    fields.append(child.text)    
                if  tag == "PlaneID":
                    planes.append(child.text)
                if  tag == "ChannelID":
                    channel_ids.append(child.text)
                if  tag == "ChannelName":
                    channel_names.append(child.text)
                if  tag == "FlimID":
                    flim_ids.append(child.text)
                if  tag == "TimepointID":
                    timepoints.append(child.text)
                if  tag == "PositionX":
                    x_positions.append(child.text)
                if  tag == "PositionY":
                    y_positions.append(child.text)
                if  tag == "AbsTime":
                    times.append(child.text)

        rows = [str(x).zfill(2) for x in rows]
        cols = [str(x).zfill(2) for x in cols]
        fields = [str(x).zfill(2) for x in fields]
        planes = [str(x).zfill(2) for x in planes]
        timepoints = [int(x) + 1 for x in timepoints]

        image_names = []
        for row, col, field, plane, channel_id, timepoint, flim_id in zip(rows, cols, fields, planes, channel_ids, timepoints, flim_ids):
            image_names.append(f"r{row}c{col}f{field}p{plane}-ch{channel_id}sk{timepoint}fk1fl{flim_id}.tiff")
        
        #remove extra spaces from channel names
        channel_names   = [x.replace(" ", "") for x in channel_names]

        #convert date/time into useful format   
        dates = [x.split("T")[0] for x in times]
        _times = [x.split("T")[1] for x in times]
        _times = [(x.split("+")[0].split(".")[0] + "+" + x.split("+")[1].replace(":", "")) for x in _times]
        time_final = [ x + " " + y for x, y in zip(dates, _times)]

        datetime_format = "%Y-%m-%d %H:%M:%S%z"
        time_unix = [datetime.strptime(x, datetime_format) for x in time_final]
        time_unix = [datetime.timestamp(x) for x in time_unix]
        
        #update file name if flatfield exported images are to be used
        if self.flatfield_status:
            image_names = [f"flex_{x}" for x in image_names]

        df = pd.DataFrame({"filename": image_names,
                        "Row": rows,
                        "Well": cols,
                        "Zstack": planes,
                        "Timepoint": timepoints,
                        "X": x_positions,
                        "Y": y_positions,
                        "date": dates,
                        "time": _times,
                        "unix_time": time_unix,
                        "Channel": channel_names})
        
        #define path where to find raw image files
        df["source"] = self.image_dir
        
        return(df)

    def get_phenix_metadata(self):
        return(self.read_phenix_xml(self.xml_path))
    
    def generate_new_filenames(self, metadata):

        #convert position values to numeric to ensure proper sorting
        metadata["X"] = [float(x) for x in metadata.X]
        metadata["Y"] = [float(x) for x in metadata.Y]

        #convert X_positions into row and col values
        metadata["X_pos"] = None
        X_values = metadata.X.value_counts().index.to_list()
        X_values = np.sort(X_values)
        for i, x in enumerate(X_values):
            metadata.loc[metadata.X == x, 'X_pos'] = i

        #get y positions
        metadata["Y_pos"] = None
        Y_values = metadata.Y.value_counts().index.to_list()
        Y_values = np.sort(Y_values) #ensure that the values are numeric and not string
        for i, y in enumerate(Y_values):
            metadata.loc[metadata.Y == y, 'Y_pos'] = i

        #get number of rows and wells and adjust labelling if specific entries need to be compressed
        wells = metadata.Well.value_counts().index.to_list()
        rows = metadata.Row.value_counts().index.to_list()

        wells.sort()
        rows.sort(reverse = True) #invert because the image quadrant beginns in the bottom left

        if self.compress_rows:
            for well in wells:
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    else:
                        max_y = metadata.loc[((metadata.Well == well) & (metadata.Row == rows[0]))].Y_pos.max()
                        metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Y_pos"] = metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Y_pos"] + int(max_y) + int(1)
                        metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Row"] = rows[0]

        if self.compress_cols:
            for i, well in enumerate(wells):
                if i == 0:
                    continue
                else:
                    max_x = metadata.loc[((metadata.Well == wells[0]))].X_pos.max()
                    metadata.loc[(metadata.Well == well), "X_pos"] = metadata.loc[(metadata.Well == well), "X_pos"] + int(max_x) + int(1)
                    metadata.loc[(metadata.Well == well), "Well"] = wells[0]

        metadata.X_pos = [str(int(x)).zfill(3) for x in metadata.X_pos]
        metadata.Y_pos = [str(int(x)).zfill(3) for x in metadata.Y_pos]
        metadata.Timepoint = [str(x).zfill(3) for x in metadata.Timepoint]
        metadata.Zstack = [str(x).zfill(2) for x in metadata.Zstack]

        #generate new file names
        for i in range(metadata.shape[0]):
            _row = metadata.loc[i, :]
            name = "Timepoint{}_Row{}_Well{}_{}_zstack{}_r{}_c{}.tif".format(_row.Timepoint,
                _row.Row, _row.Well, _row.Channel,_row.Zstack, _row.Y_pos, _row.X_pos)
            name = name
            metadata.loc[i, 'new_file_name'] = name

        return(metadata)

    def check_for_missing_files(self, metadata = None, return_values = False):

        if metadata is None:
            print("No metadata passed, so reading from file. This can take a moment...")
            metadata = self.get_phenix_metadata()
            metadata = self.generate_new_filenames(metadata)

        channels = np.unique(metadata.Channel)
        zstacks = np.unique(metadata.Zstack)

        rows = np.unique(metadata.Row)
        wells = np.unique(metadata.Well)
        timepoints = np.unique(metadata.Timepoint)

        #all X and Y pos values need to be there
        y_range = np.unique(metadata.Y_pos) 
        x_range = np.unique(metadata.X_pos)

        missing_images = []
        print("Checking for missing images...")
        for timepoint in timepoints:
            _df = metadata[metadata.Timepoint == timepoint]
            for row in rows:
                __df = _df[_df.Row == row]
                for well in wells:
                    ___df = __df[__df.Well == well]
                    
                    #define X_pos as missing if we expect them but they dont exist
                    missing_x = [x for x in x_range if x not in np.unique(___df.X_pos)]
                    
                    max_tiles_x = ___df.X_pos.value_counts().max()
                    min_tiles_x = ___df.X_pos.value_counts().min()
                    
                    y_range = ___df.Y_pos.max()
                    
                    #also define them as missing if there are some positions that have less images than other
                    if max_tiles_x != min_tiles_x:
                        missing_x = missing_x + ___df.X_pos.value_counts()[___df.X_pos.value_counts() < max_tiles_x].index.to_list()
                
                    if len(missing_x) > 0:
                        for _x in missing_x:
                            ____df = ___df[___df.X_pos == _x]
                            
                            missing_y = [y for y in y_range if y not in np.unique(____df.Y_pos)]
                            for _y in missing_y:
                                print(f"Missing tile at position: Timepoint{timepoint}_Row{row}_Well{well}_CHANNELS_zstackXX_r{str(_x).zfill(3)}_c{str(_y).zfill(3)}.tif")
                                for channel in channels:
                                    for zstack in zstacks:
                                        missing_images.append(f"Timepoint{timepoint}_Row{row}_Well{well}_{channel}_zstack{zstack}_r{str(_x).zfill(3)}_c{str(_y).zfill(3)}.tif")
        if len(missing_images) == 0:
            print("No missing tiles found.")
        else:
            
            #get size of missing images that need to be replaced
            image = imread(os.path.join(metadata["source"][0], metadata["filename"][0]))
            image[:] = int(0)
            self.black_image = image

            print(f"The found missing tiles need to be replaced with black images of the size {image.shape}. You can do this be executing replace_missing_images().")

        self.missing_images = missing_images
        
        if return_values:
            return(missing_images)
    
    def replace_missing_images(self):
        #get output directory
        self.define_outdir()

        if self.missing_images is None:
            self.check_for_missing_files()
        
        if len(self.missing_images) > 0:
            for missing_image in self.missing_images:
                print(f"Creating black image with name: {missing_image}")
                imwrite(os.path.join(self.outdir, missing_image), self.black_image)
            
            print(f"All missing images successfully replaced with black images of the dimension {self.black_image.shape}")
        
    def copy_files(self, metadata):

        print("Starting copy process...")
        #define function for copying depending on if symlinks should be used or not
        if self.export_symlinks:
            def copyfunction(input, output):
                try:
                    os.symlink(input, output)
                except:
                    return()
        else:
            def copyfunction(input, output):
                shutil.copyfile(input, output)

        #actually perform the copy process  
        for old, new, source in tqdm(zip(metadata.filename.tolist(), metadata.new_file_name.tolist(), metadata.source.tolist()), 
                            total = len(metadata.new_file_name.tolist())):
            
            #define old and new paths for copy process
            old_path = os.path.join(source, old)
            new_path = os.path.join(self.outdir, new)
            
            #check if old path exists
            if os.path.exists(old_path):
                copyfunction(old_path, new_path)
            else:
                print("Error: ", old_path, "not found.")
        print("Copy process completed.")
    
    def save_metadata(self, metadata):

        #save to csv file
        metadata.to_csv(f"{self.experiment_dir}/metadata_image_parsing.csv")
        print(f"Metadata used to parse images saved to file {self.experiment_dir}/metadata_image_parsing.csv")

    def parse(self):
        
        #create output directory
        self.define_outdir()

        #get metadata for the images we want to parse
        metadata = self.get_phenix_metadata()
        metadata_new = self.generate_new_filenames(metadata)

        #copy/link the images to their new names
        self.copy_files(metadata=metadata_new)

        #check for missing images and replace them
        self.check_for_missing_files(metadata = metadata_new)
        self.replace_missing_images()
        self.save_metadata(metadata_new)
        
class CombinedPhenixParser(PhenixParser):
    directory_combined_measurements = "experiments_to_combine" 

    def __init__(self, experiment_dir, flatfield_exported=True, export_symlinks=True, compress_rows=False, compress_cols=False) -> None:
        
        self.experiment_dir = experiment_dir
        self.get_datasets_to_combine()

        super().__init__(experiment_dir, flatfield_exported, export_symlinks, compress_rows, compress_cols)   

    def get_xml_path(self):
        #get index file of the first phenix dir(this is our main experiment!)
        index_file = f"{self.phenix_dirs[0]}/Images/Index.ref.xml"
        
        #perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return(index_file)
    
    def get_input_dir(self):
        input_dir = f"{self.phenix_dirs[0]}/Images/flex"

        #perform sanity check if file exists else exit
        if not os.path.isdir(input_dir):
            sys.exit(f"Can not find directory containing images to parse: {input_dir}")

        return(input_dir)
    
    def get_datasets_to_combine(self):
        input_path = f"{self.experiment_dir}/{self.directory_combined_measurements}"
        
        #get phenix directories that need to be comined together
        phenix_dirs = os.listdir(input_path)

        #order phenix directories accoding to creation data
        # Define a regular expression pattern to match the date and time
        pattern = r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}' 

        # Extract the date and time information from each file name
        dates_times = [re.search(pattern, file_name).group() for file_name in phenix_dirs]

        # Sort the file names based on the extracted date and time information
        sorted_phenix_dirs = [file_name for _, file_name in sorted(zip(dates_times, phenix_dirs))]

        self.phenix_dirs = [f"{input_path}/{phenix_dir}" for phenix_dir in sorted_phenix_dirs]
    
    def get_phenix_metadata(self):

        ###
        #read metadata from all experiments and merge into one file
        #note: if more than one image exists at a specific position then the first image aquired will be preserved based on the timestamps in the exported phenix measurement names
        ####

        #define under what path the actual exported images will be found
        #this is hard coded through phenix export script, do not change
        if self.flatfield_status:
            append_string = "Images/flex"
        else:
            append_string = "Images"

        #read all metadata
        metadata = {}
        for phenix_dir in self.phenix_dirs:
            df = self.read_phenix_xml(f"{phenix_dir}/Images/Index.ref.xml")
            df = df.set_index(["Row", "Well", "Zstack", "Timepoint", "X", "Y", "Channel"])
            df.loc[:, "source"] = f"{phenix_dir}/{append_string}" #update source with the correct strings
            metadata[phenix_dir] = df

        #merge generated metadata files together (order of what is preserved is according to calcualted creation times above)
        for i, key in enumerate(metadata.keys()):
            if i == 0:
                metadata_merged = metadata[key]
            else:
                metadata_merged = metadata_merged.combine_first(metadata[key]) 
        
        metadata_merged = metadata_merged.reset_index()
        
        #return generated dataframe
        print("merged metadata generated from all passed phenix experiments.")
        return(metadata_merged)
    
    def parse(self):

        #create output directory
        self.define_outdir()

        metadata = self.get_phenix_metadata()
        metadata_new = self.generate_new_filenames(metadata=metadata)

        self.copy_files(metadata=metadata_new)

        #check for missing images and replace them
        self.check_for_missing_files(metadata = metadata_new)
        self.replace_missing_images()

        #generate a log report of the parsing process and write out the metadatafile used to parse the images
        self.save_metadata(metadata_new)


def sort_timepoints(parsed_dir, use_symlink = False):
    """
    Additionally sort generated timecourse images according to well and tile position. Function 
    generates a new folder called timecourse_sorted which contains a unqiue folder for each unique tile
    position containing all imaging data (i.e. zstacks, timepoints, channels) of that tile.
    This function is meant for quick sorting of generated images for simple import of e.g. timecourse 
    experiments into FIJI. 
    
    Parameters
    ----------
    parsed_dir
        filepath to parsed images folder generated with the function parse_phenix.
    use_symlonks : bool
        boolean value indicating if the images should be copied as symlinks or as regular files. Symlinks can potentially cause issues if using the data on
        different OS but is signficiantly faster and does not produce as much data overhead.
    """

    outdir = parsed_dir.replace(os.path.basename(parsed_dir), "timecourse_sorted")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    images = os.listdir(parsed_dir)
    images = [x for x in images if x.endswith((".tiff", ".tif"))]

    rows = list(set([x.split("_")[1] for x in images]))
    wells = list(set([x.split("_")[2] for x in images]))
    tiles = list(set([x.split(".tif")[0].split("_zstack")[1][4:] for x in images]))

    print("Found the following image specs: ")
    print("\t Rows: ", rows)
    print("\t Wells: ", wells)
    print("\t Tiles: ", tiles)
    
    for row in rows:
        for well in wells:
            _outdir = os.path.join(outdir, row + "_" + well)
            for tile in tiles:
                
                __outdir = os.path.join(_outdir + "_" + tile)
                
                #create outdir if not already existing
                if not os.path.exists(__outdir):
                    os.mkdir(__outdir)

                #copy all files that match this folder

                expression = f"*_{row}_{well}_*_{tile}.tif"

                if use_symlink:
                    def copyfunction(input, output):
                        try:
                            os.symlink(input, output)
                        except OSError as e:
                            if e.errno == errno.EEXIST:
                                os.remove(output)
                                os.symlink(input, output)
                else:
                    def copyfunction(input, output):
                        shutil.copyfile(input, output)

                for file in glob.glob(os.path.join(parsed_dir, expression)):
                    copyfunction(file, os.path.join(__outdir, os.path.basename(file)))
                print("completed export for " + row + "_" + well + "_" + tile)

def sort_wells(parsed_dir, use_symlink = False, assign_random_id = False):
    """
    Sort acquired phenix images into unique folders for each well. 

    Parameters
    ----------
    parsed_dir
        filepath to parsed images folder generated with the function parse_phenix.
    use_symlink
        boolean value indicating if the images should be copied as symlinks to their new destination
    assign_random_id
        boolean value indicating if the images in the sorted wells folder should be prepended with a random id.
    """

    outdir = parsed_dir.replace("parsed_images", "well_sorted")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    images = os.listdir(parsed_dir)
    images = [x for x in images if x.endswith((".tiff", ".tif"))]

    rows = list(set([x.split("_")[1] for x in images]))
    wells = list(set([x.split("_")[2] for x in images]))
    tiles = list(set([x.split(".tif")[0].split("_zstack")[1][4:] for x in images]))

    print("Found the following image specs: ")
    print("\t Rows: ", rows)
    print("\t Wells: ", wells)
    print("\t Tiles: ", tiles)
    
    for row in tqdm(rows):
        for well in tqdm(wells):
            _outdir = os.path.join(outdir, row + "_" + well)
            
            #create outdir if not already existing
            if not os.path.exists(_outdir):
                os.mkdir(_outdir)

            #copy all files that match this folder

            expression = f"*_{row}_{well}_*.tif"

            if use_symlink:
                def copyfunction(input, output):
                    try:
                        os.symlink(input, output)
                    except OSError as e:
                        if e.errno == erro.EEXIST:
                            os.remove(output)
                            os.symlink(input, output)
            else:
                def copyfunction(input, output):
                    shutil.copyfile(input, output)
            files = glob.glob(os.path.join(parsed_dir, expression))
            files = np.sort(files)

            if assign_random_id:
                random.seed(16)
                random.shuffle(files)
                for i, file in enumerate(files):
                    outfile_name = str(i) + "_" + os.path.basename(file)
                    copyfunction(file, os.path.join(_outdir, outfile_name))
            else:
                for file in files:
                    copyfunction(file, os.path.join(_outdir, os.path.basename(file)))