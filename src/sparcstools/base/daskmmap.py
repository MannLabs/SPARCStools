# Import necessary modules
import numpy as np
import dask
import dask.array as da
import h5py

def dask_array_from_path(file_path):
    """create a memory mapped dask file from a memory mapped HDF5 compatible file.

    Parameters
    ----------
    file_path : str
        path pointing to the HDF5 file comatible with memory mapping using the alphabase.io.tempmmap module.

    Returns
    -------
    dask.array
        dask.array that is memory mapped to disk
    """
    # Load the memory-mapped array from the file
    with h5py.File(file_path, "r") as hdf_file:
        array = hdf_file["array"]
        shape = array.shape
        dtype = array.dtype
        offset = array.id.get_offset()

    # Create a Dask array from the memory-mapped file
    dask_array = mmap_dask_array(file_path, shape, dtype, offset=offset)
    return dask_array

def mmap_dask_array(filename, shape, dtype, offset=0, blocksize=5):
    '''
    Create a Dask array from raw binary data in `filename`
    by memory mapping.
    '''
    load = dask.delayed(mmap_load_chunk)
    chunks = []
    for index in range(0, shape[0], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[0] - index)
        chunk = dask.array.from_delayed(
            load(
                filename,
                shape=shape,
                dtype=dtype,
                offset=offset,
                sl=slice(index, index + chunk_size)
            ),
            shape=(chunk_size, ) + shape[1:],
            dtype=dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)

def mmap_load_chunk(filename, shape, dtype, offset, sl):
    '''
    Memory map the given file with overall shape and dtype and return a slice
    specified by `sl`.
    '''
    data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset)
    return data[sl]