import numpy as np
import dask
import dask.array as da
import h5py

def dask_array_from_path(file_path):
    """Create a memory-mapped Dask array from a HDF5 file.

    Parameters
    ----------
    file_path : str
        Path pointing to the HDF5 file compatible with memory mapping.

    Returns
    -------
    dask.array.Array
        Dask array that is memory-mapped to disk.
    """
    # Load the memory-mapped array from the file
    with h5py.File(file_path, "r") as hdf_file:
        array = hdf_file["array"]
        shape = array.shape
        dtype = array.dtype
        offset = array.id.get_offset()

    # Calculate the chunk sizes
    chunks = calculate_chunk_sizes(shape, dtype)

    # Create a Dask array from the memory-mapped file
    dask_array = mmap_dask_array(file_path, shape, dtype, offset=offset, chunks=chunks)
    return dask_array

def calculate_chunk_sizes(shape, dtype, target_size_gb=5):
    """
    Calculate chunk sizes that result in chunks of approximately the target size in GB. 

    XY dimensions are divided by 2 until the target size is reached to keep chunks as square as possible. The chunk sizes for all higher dimensions are set to 1.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : np.dtype
        Data type of the array.
    target_size_gb : int
        Target size of each chunk in gigabytes.

    Returns
    -------
    tuple
        Calculated chunk sizes for the Dask array.
    """
    # Size of each element in bytes
    element_size = np.dtype(dtype).itemsize
    # Target number of bytes per chunk
    target_size_bytes = target_size_gb * 1024 ** 3
    # Calculate the total number of elements that fit into the target chunk size
    total_elements_per_chunk = target_size_bytes // element_size

    # Initialize chunk sizes to 1
    chunk_sizes = [1] * len(shape)
    chunk_sizes[-1] = shape[-1]
    chunk_sizes[-2] = shape[-2]
    
    while np.product(chunk_sizes) > total_elements_per_chunk:
        chunk_sizes[-1] = chunk_sizes[-1] // 2
        chunk_sizes[-2] = chunk_sizes[-2] // 2

    return tuple(chunk_sizes)

def mmap_dask_array(filename, shape, dtype, offset=0, chunks=(5,)):
    """
    Create a Dask array from raw binary data in `filename` by memory mapping.

    Parameters
    ----------
    filename : str
        Path to the raw binary data file.
    shape : tuple
        Shape of the array.
    dtype : np.dtype
        Data type of the array.
    offset : int, optional
        Offset in bytes from the beginning of the file.
    chunks : tuple, optional
        Chunk sizes for the Dask array.

    Returns
    -------
    dask.array.Array
        Dask array that is memory-mapped to disk.
    """
    load = dask.delayed(mmap_load_chunk)
    chunk_arrays = []

    for i in range(0, shape[0], chunks[0]):
        channel_chunks = []
        for j in range(0, shape[1], chunks[1]):
            row_chunks = []
            for k in range(0, shape[2], chunks[2]):
                chunk_shape = (
                    min(chunks[0], shape[0] - i),
                    min(chunks[1], shape[1] - j),
                    min(chunks[2], shape[2] - k),
                )
                slices = (slice(i, i + chunk_shape[0]), slice(j, j + chunk_shape[1]), slice(k, k + chunk_shape[2]))
                chunk = da.from_delayed(
                    load(filename, shape, dtype, offset, slices),
                    shape=chunk_shape,
                    dtype=dtype
                )
                row_chunks.append(chunk)
            channel_chunks.append(da.concatenate(row_chunks, axis=2))
        chunk_arrays.append(da.concatenate(channel_chunks, axis=1))
    return da.concatenate(chunk_arrays, axis=0)

def mmap_load_chunk(filename, shape, dtype, offset, slices):
    """
    Memory map the given file with overall shape and dtype and return a slice specified by `slices`.

    Parameters
    ----------
    filename : str
        Path to the raw binary data file.
    shape : tuple
        Shape of the array.
    dtype : np.dtype
        Data type of the array.
    offset : int
        Offset in bytes from the beginning of the file.
    slices : tuple
        Tuple of slices specifying the chunk to load.

    Returns
    -------
    np.ndarray
        The sliced chunk from the memory-mapped array.
    """
    data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset)
    return data[slices]
