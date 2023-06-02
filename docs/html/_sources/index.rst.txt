.. sparcstools documentation master file

Welcome to SPARCStools documentation!
=======================================

This python module contains wrapper functions to perform stitching with the `Ashlar API <https://labsyspharm.github.io/ashlar/>`_. In addition it contains
data parsing functions to make imaging data aquired with the `Perkinelmer Opera Phenix Microscope <https://www.perkinelmer.com/uk/product/opera-phenix-plus-system-hh14001000>`_ accessible Ashlar 
or also to other downstream applications. 

The generated stitched images can then be used for downstream applications, for example the `SPARCSpy <link URL>`_ workflow or other analysis frameworks.

Installation
------------

SPARCStools has been tested using python >= 3.8 on Linux and MacOS. Currently to run on Windows please utilize a Linux Virtual Machine.

Clone the github repository and navigate to the main directory:

.. code::

   git clone https://github.com/MannLabs/SPARCStools.git
   cd SPARCStools

Create a conda environment and activate it

.. code::

   conda create -n stitching python=3.10
   conda activate stitching

Install Java using conda

.. code::

   conda install -c conda-forge openjdk

Install package via pip. This should install all dependencies as well.

.. code::
   
   pip install .

Citing our Work 
----------------

This code was developed by Sophia Maedler and Niklas Schmacke in the labs of Matthias Mann and Veit Hornung. 

If you use our work please cite the [following manuscript](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1):

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416


.. toctree::
   :maxdepth: 3
   :caption: Modules:
   
   pages/modules.rst
   
.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Tutorials:
   
   pages/tutorials

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Example Notebooks:
   
   pages/notebooks/example_stitching_notebook