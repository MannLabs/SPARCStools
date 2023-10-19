import os
import numpy as np
import h5py
import pandas as pd
from shapely.geometry import Polygon
from rasterio.features import rasterize
import numpy.ma as ma
from tqdm.auto import tqdm

def get_indexes(project_location, cell_ids, return_annotation = False):
    
    #load hdf5 to get indexes
    hf_path = os.path.join(project_location, "extraction", "data", "single_cells.h5")
    hf = h5py.File(hf_path)
    indexes = hf.get("single_cell_index")[:]
    hf.close()
    
    index_values, ids = indexes.T
    lookup = dict(zip(ids, index_values))

    index_locs = []
    for cell_id in tqdm(cell_ids, desc = "getting indexes"):
        index_locs.append(lookup[cell_id])
    
    if return_annotation:
        annotation = pd.DataFrame({"index_hdf5": index_locs, "cell_id":cell_ids})
        annotation = annotation.sort_values("index_hdf5")
        return(np.array(annotation.index_hdf5.tolist()), np.array(annotation.cell_id.tolist()))
    else:
        return(np.sort(index_locs))    
    
def save_cells_to_new_hdf5(project_location, name, cell_ids, annotation = "selected_cells", append = False):
    
    #get output directory
    outdir = f"{project_location}/extraction/filtered_data/{name}/"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    #generate outfile
    outfile = f"{project_location}/extraction/filtered_data/{name}/single_cells.h5"
    
    #get indexes of cells we want to write
    indexes, cell_ids = get_indexes(project_location, cell_ids, return_annotation = True)
    
    #generate annotation
    annotation_df = pd.DataFrame({"cell_id":cell_ids, "index_hdf5":indexes, "label":annotation}) 
    annotation_df.cell_id = annotation_df.cell_id.astype("str") #explicitly conver to string so that we can save to HDF5
    annotation_df.index_hdf5= annotation_df.index_hdf5.astype("str")
    
    #get cell images we want to write to new location
    print("getting cell images for selected cells...")
    with h5py.File(f"{project_location}/extraction/data/single_cells.h5", "r") as hf_in:
        cell_images = hf_in.get("single_cell_data")[indexes]
    
    #delete file if append is False and it already exists so that we generate a new one
    if not append:
        if os.path.isfile(outfile):
            os.remove(outfile)
          
    if os.path.isfile(outfile):
        print(f"appending to existing dataset at {outfile}")
        with h5py.File(outdir, "a") as hf_out:
            #get handels on datasets
            single_cell_data = hf_out.get("single_cell_data")
            single_cell_index = hf_out.get("single_cell_index")
            annotation_hf = hf_out.get("annotation")

            #calculate how we need to adjust the HDF5 dataset size
            single_cell_data_shape = single_cell_data.shape
            hf_length_old = single_cell_data_shape[0]
            hf_length_new = single_cell_data_shape[0] + len(indexes)

            single_cell_data.resize((hf_length_new, single_cell_data_shape[1], single_cell_data_shape[2], single_cell_data_shape[3]))
            single_cell_data[hf_length_old : hf_length_new] = cell_images

            single_cell_index.resize((hf_length_new, 2))
            single_cell_index[hf_length_old : hf_length_new] = np.array((list(range(hf_length_old, hf_length_new)), indexes)).T

            annotation_hf.resize((hf_length_new, 3))
            annotation_hf[hf_length_old : hf_length_new] = annotation_df

        print("results saved")
    else:
        print(f"creating new dataset at path {outfile}") 
        with h5py.File(outfile, "w") as hf_out: 
            #get size of images
            n_cells, n_channels, x, y = cell_images.shape
            
            #create index file
            hf_out.create_dataset('single_cell_index', (n_cells, 2), maxshape = (None, 2), dtype="uint64")
            

            #create dataset
            hf_out.create_dataset('single_cell_data', (n_cells, n_channels, x, y), 
                                  maxshape=(None, n_channels, x, y),  
                                  chunks=(1, 1, x, y),
                                  compression = True,
                                  dtype="float16")
            dt = h5py.special_dtype(vlen=str)
            hf_out.create_dataset("annotation", annotation_df.shape, maxshape = (None, 3), dtype = dt, chunks = None)

            #actually save our data
            hf_out.get("single_cell_data")[:] = cell_images
            hf_out.get("single_cell_index")[:] = np.array((list(range(len(indexes))), indexes)).T
            hf_out.get("annotation")[:] = annotation_df
        print("results saved.")

def _read_napari_csv(path):
    # read csv table
    shapes = pd.read_csv(path, sep = ",")
    shapes.columns = ['index_shape', 'shape-type', 'vertex-index', 'axis-0', 'axis-1']
    
    #get unqiue shapes
    shape_ids = shapes.index_shape.value_counts().index.tolist()
    
    polygons = []

    for shape_id in shape_ids:
        _shapes = shapes.loc[shapes.index_shape == shape_id]
        x = _shapes["axis-0"].tolist()
        y = _shapes["axis-1"].tolist()

        polygon = Polygon(zip(x, y))
        polygons.append(polygon)
    
    return polygons

def _generate_mask_polygon(poly, outshape):
    x, y = outshape
    img = rasterize(poly, out_shape = (x, y))
    return(img.astype("bool"))

def extract_single_cells_napari_area(napari_path, project_location):

    #get name from naparipath
    name = os.path.basename(napari_path).split(".")[0]

    # read napari csv
    polygons = _read_napari_csv(napari_path)
    
    #get segmentation results
    hf = h5py.File(f"{project_location}/segmentation/segmentation.h5", "r")
    labels = hf.get("labels")
    
    #determine size that mask needs to be generated for
    _, x, y = labels.shape
    
    #generate mask indicating which areas of the image to use
    mask = _generate_mask_polygon(polygons, outshape = (x, y))
    indexes_to_get = mask.nonzero()
    
    #to minimize the amount of data that needs to be loaded select square region containing the area we are interested in
    region1 = slice(indexes_to_get[0].min(), indexes_to_get[0].max())
    region2 = slice(indexes_to_get[1].min(), indexes_to_get[1].max())
    
    #load data
    print("getting labels to extract unique cell ids")
    segmentation_labels = labels[0, region1, region2]
    hf.close()
    mask_cropped = mask[region1, region2]
    
    #get all unique cell ids in this area
    masked = ma.masked_array(segmentation_labels, mask=~mask_cropped)
    cell_ids = list(np.unique(masked.compressed()))
    cell_ids.remove(0)

    #filter cell ids for those not on shard edges
    cell_ids_all = set(pd.read_csv(f"{project_location}/segmentation/classes.csv", header = None)[0].astype("int").tolist())
    cell_ids = set(cell_ids)
    cell_ids = cell_ids.intersection(cell_ids_all)
    cell_ids = list(cell_ids) #need list type for lookup of indexes

    print(f"found {len(cell_ids)} unique cell_ids in selected area. Will now export to new HDF5 single cell dataset.")
    save_cells_to_new_hdf5(project_location, cell_ids = cell_ids, name = name, annotation = name, append = False)