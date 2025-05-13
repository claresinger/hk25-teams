import numpy as np
import xarray as xr
import zarr
import numcodecs
import gc

def get_dtype(da):
    if np.issubdtype(da.dtype, np.floating):
        return "float32"
    else:
        return da.dtype

def get_chunks(dimensions):
    chunks = {
            "time": 24,
            "cell": 4**9,
        }
    return tuple(chunks.get(d, -1) for d in dimensions)

def get_compressor():
    return numcodecs.Blosc("zstd", shuffle=2)

def get_encoding(dataset):
    return {
        var: {
            "compressor": get_compressor(),
            "dtype": get_dtype(dataset[var]),
            "chunks": get_chunks(dataset[var].dims),
        }
        for var in dataset.variables
        if var not in dataset.dims
    }

starting_zoom = 6
ceres_type = "CERES_Cru23"
# ceres_type = "CERES_EBAF-TOA_Ed4.2.1"
native_zoom = "/home/cs3554/"+ceres_type+"_hp"+str(starting_zoom)+".zarr"
dn=xr.open_dataset(native_zoom,chunks={"time":24,"cell": 4**9})

for x in range(starting_zoom-1,-1,-1):
    new_zoom = "/home/cs3554/"+ceres_type+"_hp"+str(x)+".zarr"
    dx=dn.coarsen(cell=4).mean()
    dx['crs'].attrs['healpix_nside'] = 2**x   # Update HEALPix level metadata
    if 'cell' in dx.data_vars:
        dx = dx.drop_vars('cell')
    store = zarr.storage.DirectoryStore(new_zoom, dimension_separator="/")
    dx.chunk({"time": 24, "cell": 4**8}).to_zarr(store, mode='w', encoding=get_encoding(dx))
    dn=dx
    del dx,store
    gc.collect()
