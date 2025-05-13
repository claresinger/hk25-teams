import healpix as hp
import numpy as np
import xarray as xr
import numcodecs
import zarr
import easygems.remap as egr

localpath = "/home/cs3554/"
# netcdfpath = "CERES_Cru23_ds.nc"
# weightsfile = "CERES_cru23_healpix_weights"
# zarrfile = "CERES_Cru23_hp"
netcdfpath = "CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202502.nc"
weightsfile = "CERES_4.2.1_healpix_weights"
zarrfile = "CERES_EBAF-TOA_Ed4.2.1_hp"

ds = xr.open_dataset(localpath+netcdfpath)

s='6'
order = zoom = int(s)
nside = hp.order2nside(order)
npix = hp.nside2npix(nside)

write = True

if write:

    hp_lon, hp_lat = hp.pix2ang(
        nside=nside, ipix=np.arange(npix), lonlat=True, nest=True)
    hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
    hp_lon += 360 / (4 * nside) / 4  # shift quarter-width

# Get lat/lon from CERES data
    ds = ds.stack(xy=("lon","lat"))
    slon=ds.lon
    slat=ds.lat
    
# Periodically extend longitude to the east and west
    lon_periodic = np.hstack((slon - 360, slon, slon + 360))
    lat_periodic = np.hstack((slat, slat, slat))

# Compute weights
    eweights = egr.compute_weights_delaunay(
        points=(lon_periodic, lat_periodic),
        xi=(hp_lon, hp_lat)
    )

# Remap the source indices back to their valid range
    eweights = eweights.assign(src_idx=eweights.src_idx % slat.size)

# Save the calculated weights for future use
    eweights.to_netcdf(localpath+weightsfile+s+".nc")

else: 
    eweights=xr.open_dataset(localpath+weightsfile+s+".nc") 


### Rewrite from Jill Zhang
def custom_indexing(ds_var, src_idx, weights, valid):
    # Perform the advanced indexing in a NumPy context
    return np.where(
        valid,
        (ds_var[src_idx] * weights).sum(axis=-1),
        np.nan
    )


### Fast way from Jill Zhang

valid = eweights["valid"]
src_idx = eweights["src_idx"]
weights = eweights["weights"]


## Remap the full data set

# Consolidate xy dimension into a single chunk for remapping
ds = ds.chunk(dict(xy=-1))

# if the dataset contains coordinates, then remove them 
if 'lat' in ds.variables:
   ds_sm=ds.drop_vars(["lat","lon"])
    
ceres_remap = xr.apply_ufunc(
    egr.apply_weights,
    ds_sm,
    kwargs=eweights,
    keep_attrs=True,
    input_core_dims=[["xy"]],
    output_core_dims=[["cell"]],
    output_dtypes=["f4"],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={
        "output_sizes": {"cell": npix},
    },
)


ceres_remap["crs"] = xr.DataArray(
    name="crs",
    data=[],
    dims="crs",
    attrs={
        "grid_mapping_name": "healpix",
        "healpix_nside": 2**zoom,
        "healpix_order": "nest",
    },
)


## Write to Zarr

# Single Precision for Floats
def get_dtype(da):
    if np.issubdtype(da.dtype, np.floating):
        return "float32"
    else:
        return da.dtype

# Chunking (note, 'cell' has to match name of column dimension in input)
def get_chunks(dimensions):
    if "level" in dimensions:
        chunks = {
            "time": 24,
            "cell": 4**5,
            "level": 4,
        }
    else:
        chunks = {
            "time": 24,
            "cell": 4**9,
        }

    return tuple((chunks[d] for d in dimensions))

# Compression
def get_compressor():
    return numcodecs.Blosc("zstd", shuffle=2)

##
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


#all variables
ofn=localpath+zarrfile+s+".zarr"
store = zarr.storage.DirectoryStore(ofn, dimension_separator="/")
ceres_remap.chunk(
    {"time": 24, "cell": 4**9}).to_zarr(
        store, encoding=get_encoding(ceres_remap))


