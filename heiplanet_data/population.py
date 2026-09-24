"""Population-derived quantities on regular latitude-longitude grids.

This module turns population counts per grid cell (e.g. ISIMIP
``total-population``) into population density:

* grid-cell area in km^2 computed from the grid coordinates, assuming a
  spherical Earth as CDO's ``gridarea`` operator does
  (`calculate_grid_cell_area`),
* grid-cell area loaded from a precomputed NetCDF file
  (`load_grid_cell_area`),
* population density as count divided by cell area
  (`calculate_population_density`).
"""

from pathlib import Path

import numpy as np
import xarray as xr

# CDO's default Earth radius (``gridarea``), in km
EARTH_RADIUS_KM = 6371.0

DENSITY_SUFFIX = "-density"


def _grid_spacing(coord: xr.DataArray) -> float:
    """Return the constant spacing of a regular 1D coordinate in degrees.

    Args:
        coord (xr.DataArray): Latitude or longitude coordinate.

    Returns:
        float: Absolute spacing between neighbouring points.
    """
    if coord.size < 2:
        raise ValueError(
            f"Coordinate '{coord.name}' needs at least two points to infer the grid spacing."
        )
    diffs = np.abs(np.diff(coord.values.astype(np.float64)))
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(f"Coordinate '{coord.name}' is not regularly spaced.")
    return float(diffs[0])


def calculate_grid_cell_area(
    dataset: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    radius_km: float = EARTH_RADIUS_KM,
) -> xr.DataArray:
    """Calculate the area of each grid cell of a regular lat-lon grid in km^2.

    Coordinates are taken as cell centers. The area of a cell between
    latitudes phi1 and phi2 and with longitude width dlambda (radians) is
    R^2 * dlambda * (sin(phi2) - sin(phi1)). Cell edges are clipped to
    +-90 degrees.

    Args:
        dataset (xr.Dataset): Dataset providing the grid coordinates.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        radius_km (float): Earth radius in km. Default is 6371.0 (as in CDO).

    Returns:
        xr.DataArray: Cell area in km^2 with dimensions (lat_name, lon_name).
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    lat = dataset[lat_name]
    lon = dataset[lon_name]
    half_dlat = _grid_spacing(lat) / 2
    dlon = np.deg2rad(_grid_spacing(lon))

    lat_values = lat.values.astype(np.float64)
    upper = np.deg2rad(np.clip(lat_values + half_dlat, -90.0, 90.0))
    lower = np.deg2rad(np.clip(lat_values - half_dlat, -90.0, 90.0))
    band_area = radius_km**2 * dlon * (np.sin(upper) - np.sin(lower))

    return xr.DataArray(
        np.broadcast_to(band_area[:, np.newaxis], (lat.size, lon.size)).copy(),
        dims=[lat_name, lon_name],
        coords={lat_name: lat.values, lon_name: lon.values},
        name="cell_area",
        attrs={
            "standard_name": "area",
            "long_name": "area of grid cell",
            "units": "km2",
        },
    )


def load_grid_cell_area(
    area_file: Path | str,
    dataset: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    var_name: str = "cell_area",
) -> xr.DataArray:
    """Load grid-cell areas in km^2 from a NetCDF file and match them to a dataset.

    The file may use ``lat``/``lon`` or the dataset's coordinate names. Its
    grid must match the dataset's grid; areas in m2 are converted to km2.

    Args:
        area_file (Path | str): Path to the NetCDF file with cell areas.
        dataset (xr.Dataset): Dataset whose grid the areas must match.
        lat_name (str): Name of the latitude coordinate in the dataset.
            Default is "latitude".
        lon_name (str): Name of the longitude coordinate in the dataset.
            Default is "longitude".
        var_name (str): Name of the area variable in the file.
            Default is "cell_area".

    Returns:
        xr.DataArray: Cell area in km^2 on the dataset's coordinates.
    """
    with xr.open_dataset(area_file) as area_ds:
        if var_name not in area_ds.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in '{area_file}'.")
        area = area_ds[var_name].load()

    area = area.rename(
        {
            old: new
            for old, new in (("lat", lat_name), ("lon", lon_name))
            if old in area.dims and old != new
        }
    ).squeeze(drop=True)
    if set(area.dims) != {lat_name, lon_name}:
        raise ValueError(
            f"Area variable must have dimensions ('{lat_name}', '{lon_name}'), "
            f"got {area.dims}."
        )

    # align to the dataset grid; fail instead of silently producing NaNs
    area = area.reindex(
        {lat_name: dataset[lat_name].values, lon_name: dataset[lon_name].values},
        method="nearest",
        tolerance=1e-6,
    )
    if area.isnull().any():
        raise ValueError(f"Grid of '{area_file}' does not match the dataset grid.")

    units = area.attrs.get("units", "km2")
    if units == "m2":
        area = area / 1e6
    elif units not in ("km2", "km^2", "km**2"):
        raise ValueError(f"Unsupported area units '{units}'; expected km2 or m2.")
    area.attrs["units"] = "km2"
    return area.transpose(lat_name, lon_name)


def calculate_population_density(
    dataset: xr.Dataset,
    var_names: list[str],
    area: xr.DataArray,
) -> xr.Dataset:
    """Add population density (persons per km^2) for population count variables.

    For each count variable ``<name>`` a new variable
    ``<name>-density`` = count / cell area is added; the counts are kept.
    Cells without data (e.g. ocean) stay NaN.

    Args:
        dataset (xr.Dataset): Dataset with population counts per grid cell.
        var_names (list[str]): Names of the population count variables.
        area (xr.DataArray): Cell area in km^2 on the dataset's grid,
            e.g. from `calculate_grid_cell_area`.

    Returns:
        xr.Dataset: Dataset with the added density variables.
    """
    missing = [name for name in var_names if name not in dataset.data_vars]
    if missing:
        raise ValueError(f"Variables {missing} not found in the dataset.")

    for name in var_names:
        count = dataset[name]
        density = (count / area).astype(count.dtype)
        density.attrs = {
            "standard_name": f"{count.attrs.get('standard_name', name)} density",
            "long_name": f"{count.attrs.get('long_name', name)} density",
            "units": "km-2",
        }
        dataset[f"{name}{DENSITY_SUFFIX}"] = density.transpose(*count.dims)

    return dataset
