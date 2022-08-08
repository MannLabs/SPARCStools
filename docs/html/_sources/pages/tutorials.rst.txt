*******************
Tutorials
*******************

Parsing and Stitching Data from Opera Phenix
============================================

First you need to export your data from harmony and rename the path to eliminate any spaces in the name.
Then you can run the following script to parses and stitch your data.

.. code-block:: python
    :caption: example script for parsing and stitching phenix data

    #import relevant libraries
    import os
    from vipertools.parse import parse_phenix
    from vipertools.stitch import generate_stitched

    #parse image data
    path = "path to exported harmony project without any spaces"
    parse_phenix(path, flatfield_exported = True, parallel = False)

    #define important information for your slide that you want to stitch

    # the code below needs to be run for each slide contained in the imaging experiment! 
    # Can be put into a loop for example to automate this or also can be subset to seperate 
    # jobs to run on for example the hpc

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
    #crop = {'top':72, 'bottom':52, 'left':48, 'right':58} #this is good default values for an entire PPS slide with cell culture samples imaged with my protocol

    #create output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #define patter to recognize which slide should be stitched
    #remember to adjust the zstack value if you aquired zstacks and want to stitch a speciifc one in the parameters above 

    pattern = "Timepoint"+str(timepoint)+"_Row"+ str(row) + "_" + "Well" + str(well) + "_{channel}_"+"zstack"+str(zstack_value)+"_r{row:03}_c{col:03}.tif"
    generate_stitched(input_dir, 
                        slidename,
                        pattern,
                        outdir,
                        overlap,
                        crop = crop ,
                        stitching_channel = stitching_channel, 
                        filetype = output_filetype)

