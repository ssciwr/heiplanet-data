"""Map 0.5 deg ISIMIP population density onto the 0.1 deg ERA5-Land silver grid.

Each 0.1 deg target point takes the density of the 0.5 deg source cell that
contains it (piecewise-constant, i.e. area-conservative for a density).
"""

import sys

import numpy as np
import xarray as xr

pop_file, era5_file, out_file = sys.argv[1:4]

pop = xr.open_dataset(pop_file)
era5 = xr.open_dataset(era5_file)
dens = pop["total-population-density"].load()

src_res = 0.5
src_lat = dens.latitude.values
src_lon = dens.longitude.values
# source cells: lat edges 90 -> -90 (descending), lon edges -180 -> 180
assert np.isclose(src_lat[0], 90 - src_res / 2) and src_lat[1] < src_lat[0]
assert np.isclose(src_lon[0], -180 + src_res / 2) and src_lon[1] > src_lon[0]

tgt_lat = era5.latitude.values
tgt_lon = era5.longitude.values

# index of the containing source cell (round to avoid float noise at edges)
ilat = np.floor(np.round((90.0 - tgt_lat) / src_res, 6)).astype(int)
ilat = np.clip(ilat, 0, len(src_lat) - 1)  # lat = -90 -> last row
ilon = np.floor(np.round((tgt_lon + 180.0) / src_res, 6)).astype(int)
ilon = ilon % len(src_lon)  # lon = 180 wraps to -180

out = dens.isel(
    latitude=xr.DataArray(ilat, dims="latitude"),
    longitude=xr.DataArray(ilon, dims="longitude"),
)
out = out.assign_coords(latitude=tgt_lat, longitude=tgt_lon)
out.attrs = dens.attrs.copy()
out.attrs["regrid_note"] = (
    "Mapped from 0.5 deg ISIMIP density to 0.1 deg ERA5-Land grid; each point "
    "takes the density of the containing 0.5 deg cell."
)
for c in ("latitude", "longitude"):
    out[c].attrs = era5[c].attrs.copy()

ds_out = out.to_dataset(name="total-population-density")
ds_out.attrs = pop.attrs.copy()
ds_out.to_netcdf(
    out_file,
    format="NETCDF4",
    encoding={"total-population-density": {"zlib": True, "complevel": 4}},
)

# --- sanity check: global population from density x cell area ---------------
R = 6371.0088  # km


def cell_area(lat, res_lat, res_lon):
    lat = np.asarray(lat, dtype=float)
    top = np.radians(np.clip(lat + res_lat / 2, -90, 90))
    bot = np.radians(np.clip(lat - res_lat / 2, -90, 90))
    return R**2 * np.radians(res_lon) * np.abs(np.sin(top) - np.sin(bot))


a05 = xr.DataArray(cell_area(src_lat, 0.5, 0.5), dims="latitude")
a01 = xr.DataArray(cell_area(tgt_lat, 0.1, 0.1), dims="latitude")
tot_src = pop["total-population"].sum(("latitude", "longitude")).values
tot_05 = (dens * a05).sum(("latitude", "longitude")).values
tot_01 = (out * a01).sum(("latitude", "longitude")).values
print(ds_out)
for t, a, b, c in zip(dens.time.values, tot_src, tot_05, tot_01):
    print(
        f"{str(t)[:4]}: counts {a:.4e}  dens*A(0.5) {b:.4e}  dens*A(0.1) {c:.4e}"
        f"  rel.diff {(c - a) / a:+.2e}"
    )
