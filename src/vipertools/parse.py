"""
parse
====================================

Contains functions to parse imaging data into a usable formats for downstream pipelines.
"""

import time
import os
import subprocess
import pandas as pd
from tqdm import tqdm
import shutil
import glob
from datetime import datetime

def parse_phenix(phenix_dir,
                 flatfield_exported = True,
                 parallel = False,
                 WGAbackground = False,
                 export_meta = True,
                 export_as_symlink = False):

    """
    Function to automatically rename TIFS exported from Harmony into a format where row and well ID as well as Tile position are indicated in the file name.
    Example of an exported file name: "Timepoint{#}_Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif"

    Parameters
    ----------
    phenix_dir
        Path indicating the exported harmony files to parse.
    flatfield_exported
        boolean indicating if the data was exported from harmony with or without flatfield correction.
    parallel
        boolean value indicating if the data parsing should be performed with parallelization or without (CURRENTLY NOT FUNCTIONAL ONLY USE AS FALSE)
    WGAbackground
        export second copy of WGA stains for background correction to improve segmentation. If set to False not performed. Else enter value of the channel
        that should be copied and contains the WGA stain.
    export_meta
        boolean value indicating if a metadata file containing, tile positions, exact time of measurement etc. should be written out.
    export_as_symlink
        boolean value indicating if the parsed files should be copied or symlinked. If set to true can lead to issues when accessing remote filesystems 
        from differentoperating systems
    """

    #start timer
    start_time = time.time()

    #generate directories to save output values
    if flatfield_exported:
        input_dir = os.path.join(phenix_dir, 'Images', "flex")
        index_file = os.path.join(input_dir, 'index.flex.xml')
    else:
        input_dir = os.path.join(phenix_dir, 'Images')
        index_file = os.path.join(input_dir, 'Index.idx.xml')

    lookuppath = os.path.join(input_dir, 'lookup.csv')
    outfile = os.path.join(input_dir, 'parsed_index.txt')
    outdir = os.path.join(phenix_dir, 'parsed_images')
    metadata_file = os.path.join(outdir, "metadata.csv")

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
    cmd = """grep -E '<URL>|<PositionX Unit="m">|<PositionY Unit="m">|<AbsTime>' '""" + \
        index_file + """' > '""" + outfile + """'"""
    subprocess.check_output(cmd, shell=True)

    images = []
    x_positions = []
    y_positions = []
    times = []

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
            elif line.strip().startswith("<AbsTime"): #relevant for time course experiments
                times.append(_line)
            else:
                print('error')

    #convert date/time into useful format   
    print(times)
    dates = [x.split("T")[0] for x in times]
    _times = [x.split("T")[1] for x in times]
    _times = [(x.split("+")[0].split(".")[0] + "+" + x.split("+")[1].replace(":", "")) for x in _times]
    time_final = [ x + " " + y for x, y in zip(dates, _times)]

    datetime_format = "%Y-%m-%d %H:%M:%S%z"
    time_unix = [datetime.strptime(x, datetime_format) for x in time_final]
    time_unix = [datetime.timestamp(x) for x in time_unix]

    #get plate and well Ids as well as channel information
    rows = [int(x[0:3].replace('r', '')) for x in images]
    wells = [int(x[3:6].replace('c', '')) for x in images]
    channels = [x.split('-')[1][2:3] for x in images]
    zstack = [int(x.split("p")[1][0:2]) for x in images]
    timepoint = [int(x.split("sk")[1].split("fk")[0]) for x in images]

    #fill up time point and zstack with leading 0s so that alphabetical sorting works correctly
    rows = [str(x).zfill(2) for x in rows] #need to add this since otherwise we wont be able to correctly match to wells with more than 1 digit id
    wells = [str(x).zfill(2) for x in wells]
    zstack = [str(x).zfill(3) for x in zstack]
    timepoint = [str(x).zfill(3) for x in timepoint]

    df = pd.DataFrame({"Image_files": images,
                       "Row": rows,
                       "Well": wells,
                       "Zstack":zstack,
                       "Timepoint":timepoint,
                       "X": x_positions,
                       "Y": y_positions,
                       "X_pos":None,
                       "Y_pos":None,
                       "date": dates,
                       "time": _times,
                       "unix_time": time_unix,
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
        name = "Timepoint{}_Row{}_Well{}_{}_zstack{}_r{}_c{}.tif".format(_row.Timepoint,
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
        print('parallel processing currently not implemented please rerun with parallel = False')
       # Parallel(n_jobs = 10)(delayed(shutil.copyfile(os.path.join(input_dir, old), os.path.join(outdir, new)))) for old, new in zip(df.Image_files.tolist(), df.new_file_name.tolist())
    else:
        #define copy function (i.e. if it should generate symlinks or not)
        
        if export_as_symlink:
            def copyfunction(input, output):
                os.symlink(input, output)
        else:
            def copyfunction(input, output):
                shutil.copyfile(input, output)

        for old, new in tqdm(zip(df.Image_files.tolist(), df.new_file_name.tolist()), 
                         total = len(df.new_file_name.tolist())):
            old_path = os.path.join(input_dir, old)
            new_path = os.path.join(outdir, new)
            #check if old path exists
            if os.path.exists(old_path):
                copyfunction(old_path, new_path)
            else:
                print("Error: ", old_path, "not found.")
                
    #export meta data if requested
    if export_meta:
        print("Metadata file was exported.")
        df.to_csv(metadata_file)

    endtime = time.time() - start_time
    print("Parsing Phenix data completed, total time was ", str(endtime/60), "minutes.")

    if WGAbackground != False:

        print("starting WGAbackground export.")

        #start timer
        start_time = time.time()

        for file_name in os.listdir(outdir):
            # construct full file path
            if WGAbackground in file_name:
                source = os.path.join(outdir, file_name)
                destination = os.path.join(outdir, file_name.replace(WGAbackground, "WGAbackground"))
                # copy only files
                if os.path.isfile(source):
                    shutil.copy(source, destination)

        print("Parsing Alexa488 data completed, total time was ", str(endtime/60), "minutes.")

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
    """

    outdir = parsed_dir.replace("parsed_images", "timecourse_sorted")
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
                        os.symlink(input, output)
                else:
                    def copyfunction(input, output):
                        shutil.copyfile(input, output)

                for file in glob.glob(os.path.join(parsed_dir, expression)):
                    copyfunction(file, os.path.join(__outdir, os.path.basename(file)))
                print("completed export for " + row + "_" + well + "_" + tile)

