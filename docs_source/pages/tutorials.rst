*******************
Tutorials
*******************

Parsing and Stitching Data from Opera Phenix
============================================

First you need to export your data from Harmony and rename the path to eliminate any spaces in the name.
Then you can run the following script to parses and stitch your data.

.. code-block:: python
    :caption: example script for parsing and stitching phenix data

    #import relevant libraries
    import os
    from sparcstools.parse import parse_phenix
    from sparcstools.stitch import generate_stitched

    #parse image data
    path = "path to exported harmony project without any spaces"
    parse_phenix(path, flatfield_exported = True, export_as_symlink = True) #export as symlink true enabled for better speed and to not duplicate data, set to False if you want to work with hardcopies or plan on accessing the data from multiple OS

    #define important information for your slide that you want to stitch

    # the code below needs to be run for each slide contained in the imaging experiment! 
    # Can be put into a loop for example to automate this or also can be subset to seperate 
    # jobs when running on a HPC

    input_dir = os.path.join(path, "parsed_images")
    slidename = "Slide1"
    outdir = os.path.join(path, "stitched", slidename)
    overlap = 0.1 #adjust in case your data was aquired with another overlap

    #define parameters to find correct slide in experiment folder
    row = 1
    well = 1
    zstack_value = 1
    timepoint = 1

    #define on which channel should be stitched
    stitching_channel = "Alexa647"
    output_filetype = [".tif", "ome.zarr"] #one of .tif, .ome.tif, .ome.zarr (can pass several if you want to generate all filetypes)

    #adjust cropping parameter
    crop = {'top':0, 'bottom':0, 'left':0, 'right':0}  #this does no cropping
    #crop = {'top':72, 'bottom':52, 'left':48, 'right':58} #this is good default values for an entire PPS slide with cell culture samples imaged with the SPARCSpy protocol

    #create output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #define pattern to recognize which slide should be stitched
    #remember to adjust the zstack value if you aquired zstacks and want to stitch a speciifc one in the parameters above 

    pattern = "Timepoint"+str(timepoint.zfill(3) +"_Row"+ str(row).zfill(2) + "_" + "Well" + str(well).zfill(2) + "_{channel}_"+"zstack"+str(zstack_value).zfill(3)+"_r{row:03}_c{col:03}.tif"
    generate_stitched(input_dir, 
                        slidename,
                        pattern,
                        outdir,
                        overlap,
                        crop = crop ,
                        stitching_channel = stitching_channel, 
                        filetype = output_filetype)

