[![Python package](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/MannLabs/py-lmd/actions/workflows/python-package.yml) 
[![website](https://img.shields.io/website?url=https%3A%2F%2Fmannlabs.github.io/SPARCStools/html/index.html)](https://mannlabs.github.io/SPARCStools/html/index.html)

# SPARCStools

This python module contains wrapper functions to perform stitching with the [Ashlar API](https://github.com/labsyspharm/ashlar). In addition it contains data parsing functions to make imaging data aquired with the Perkinelmer Opera Phenix Microscope accessible Ashlar or also to other downstream applications.

The generated stitched images can then be used for downstream applications, for example the [SPARCSpy](https://github.com/MannLabs/SPARCSpy) workflow or other analysis frameworks.

The documentation can be found at: https://mannlabs.github.io/SPARCStools/html/index.html

## Installation

Clone the github repository and navigate to the main directory:

    git clone https://github.com/MannLabs/SPARCStools.git
    cd SPARCStools

Create a conda environment and activate it

    conda create -n stitching python=3.10
    conda activate stitching

Install Java using conda

    conda install -c conda-forge openjdk

Install package via pip. This should install all dependencies as well.

    pip install .

## Citing our Work

py-lmd was developed by Georg Wallmann, Sophia Mädler and Niklas Schmacke in the labs of Veit Hornung and Matthias Mann. If you use our code please cite the [following manuscript](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1):

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416
