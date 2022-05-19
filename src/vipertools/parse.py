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

def parse_phenix(phenix_dir,
                 flatfield_exported = True,
                 parallel = False):

    """
    Function to automatically rename TIFS exported from Harmony into a format where row and well ID as well as Tile position are indicated in the file name.
    Example of an exported file name: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif"

    Parameters
    ----------
    phenix_dir
        Path indicating the exported harmony files to parse.
    flatfield_exported
        boolean indicating if the data was exported from harmony with or without flatfield correction.
    parallel
        boolean value indicating if the data parsing should be performed with parallelization or without (CURRENTLY NOT FUNCTIONAL ONLY USE AS FALSE)
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
        print('parallel processing currently not implemented please rerun with parallel = False')
       # Parallel(n_jobs = 10)(delayed(shutil.copyfile(os.path.join(input_dir, old), os.path.join(outdir, new)))) for old, new in zip(df.Image_files.tolist(), df.new_file_name.tolist())
    else:
        for old, new in tqdm(zip(df.Image_files.tolist(), df.new_file_name.tolist()), 
                         total = len(df.new_file_name.tolist())):
            old_path = os.path.join(input_dir, old)
            new_path = os.path.join(outdir, new)
            shutil.copyfile(old_path, new_path)

    endtime = time.time() - start_time
    print("Parsing Phenix data completed, total time was ", str(endtime/60), "minutes.")