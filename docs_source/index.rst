.. sparcstools documentation master file, created by
   sphinx-quickstart on Tue May 17 16:34:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPARCSTools documentation!
=======================================

This python module contains wrapper functions to perform stitching with the `Ashlar API <https://labsyspharm.github.io/ashlar/>`_ directly in python. In addition it contains
data parsing functions to make imaging data aquired with the `Perkinelmer Opera Phenix Microscope <https://www.perkinelmer.com/uk/product/opera-phenix-plus-system-hh14001000>`_ accessible to the Ashlar API to perform stitching 
or also to other downstream applications. 

The generated stitched images can then be used for downstream processing for example using the `SPARCSpy <link URL>`_ pipeline or also using `BIAS <https://single-cell-technologies.com/bias-2/>`_.

Installation
------------

SPARCStools has been tested using python >= 3.8 on Linux and MacOS. Currently to run on Windows please utilize a Linux Virtual Machine.

Clone the github repository and navigate to the main directory:

.. code::

   git clone https://github.com/sophiamaedler/SPARCStools.git
   cd SPARCStools

Create a conda environment and activate it

.. code::

   conda create -n stitching python=3.10
   conda activate stitching

Install Java using conda

.. code::
   conda install -c conda-forge openjdk

Install package via pip. This should install all dependencies as well.

   pip install .

.. toctree::
   :maxdepth: 3
   :caption: Modules:
   
   pages/modules.rst
   
.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Tutorials:
   
   pages/tutorials

