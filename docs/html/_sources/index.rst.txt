.. sparcstools documentation master file, created by
   sphinx-quickstart on Tue May 17 16:34:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPARCSTools documentation!
=======================================

This python module contains wrapper functions to perform stitching with the `Ashlar API <https://labsyspharm.github.io/ashlar/>`_ directly in python. In addition it contains
data parsing functions to make imaging data aquired with the `Perkinelmer Opera Phenix Microscope <https://www.perkinelmer.com/uk/product/opera-phenix-plus-system-hh14001000>`_ accessible to the Ashlar API to perform stitching 
or also to other downstream applications. 

A stitching workflow which stitches imaging data aquired from one well on the OperaPhenix would e.g. look like this: 

.. code:: python

   from sparcstools.parse import parse_phenix 

   #after exporting data from Harmony perform image parsing
   phenix_dir = "path/to/exported/data"
   parse_phenix(phenix_dir, WGAbackground = "Alexa488", export_as_symlink = True)

   #the parsed results are written to "path/to/exported/data/parsed_images"
   # after this has completed stitching can be performed

   input_stitching = os.path.join(phenix_dir, "parsed_images")
   outdir_sample = os.path.join(outdir, slidename)
   outdir_merged = os.path.join(outdir_sample, 'merged_files')
   
   #create output directory if it does not already exist
   if not os.path.exists(outdir_sample):
      os.makedirs(outdir_sample)
   if not os.path.exists(outdir_merged):
      os.makedirs(outdir_merged)

   #define parameters specific to experiment
   RowID = 1
   WellID = 1 
   zstack_value = 1 #since we only took 1 zstack!
   overlap = 0.1 #fraction indicating with how much overlap the image tiles were acquired
      
   #define file pattern for reading
   pattern = "Timepoint001_Row"+ str(RowID).zfill(2) + "_" + "Well" + str(WellID).zfill(2) + "_{channel}_"+"zstack"+str(zstack_value).zfill(3)+"_r{row:03}_c{col:03}.tif"

   from sparcstools.stitch import generate_stitched
   
   generate_stitched(input_stitching, 
                  slidename,
                  pattern,
                  outdir_merged,
                  overlap,
                  stitching_channel = "Alexa488",
                  crop = eval(crop),
                  plot_QC = True,
                  filetype = [".tif", "ome.zarr"],
                  WGAchannel = "Alexa488",
                  do_intensity_rescale = True,
                  export_XML = True)

The generated stitched images can then be used for downstream processing for example using the `SPARCSpy <link URL>`_ pipeline or also using `BIAS <https://single-cell-technologies.com/bias-2/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Modules:
   
   pages/modules.rst
   
.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Tutorials:
   
   pages/tutorials

