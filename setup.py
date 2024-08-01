import os
from gettext import install
from setuptools import setup, find_packages

# get path for requirements file
parent_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = os.path.join(parent_folder, 'requirements.txt')

# load requirements
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setup(name='sparcstools',
      version="1.0.0",
      description="Set of tools to complete tasks associated with SPARCSpy pipeline including stitching and data parsing.",
      long_description="",
      classifiers=[  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Visualization'
      ],
      platforms=['Platform Independent'],
      keywords='',
      author='Sophia Maedler',
      author_email='maedler@biochem.mpg.de',
      url='',
      license='MIT',
      include_package_data=True,
      zip_safe=True,
      install_requires=install_requires,
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      )
