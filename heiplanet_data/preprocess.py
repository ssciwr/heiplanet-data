from typing import TypeVar, Union, Dict, Any, Tuple, Literal
import xarray as xr
import numpy as np
import warnings
from pathlib import Path
from heiplanet_data import utils
import geopandas as gpd
import pandas as pd
import re
import xesmf as xe
import tempfile
import textwrap
from cdo import Cdo
from dataclasses import dataclass
import logging
import exactextract as ee
from functools import reduce


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Union[np.float64, xr.DataArray])
warn_positive_resolution = "New resolution must be a positive number."
CRS = 4326  # EPSG code for WGS 84


def convert_360_to_180(longitude: T) -> T:
    """Convert longitude from 0-360 to -180-180.

    Args:
        longitude (T): Longitude in 0-360 range.

    Returns:
        T: Longitude in -180-180 range.
    """
    return (longitude + 180) % 360 - 180


def adjust_longitude_360_to_180(
    dataset: xr.Dataset,
    limited_area: bool = False,
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Adjust longitude from 0-360 to -180-180.

    Args:
        dataset (xr.Dataset): Dataset with longitude in 0-360 range.
        limited_area (bool): Flag indicating if the dataset is a limited area.
            Default is False.
        lon_name (str): Name of the longitude variable in the dataset.
            Default is "longitude".

    Returns:
        xr.Dataset: Dataset with longitude adjusted to -180-180 range.
    """
    if lon_name not in dataset.coords:
        raise ValueError(f"Longitude coordinate '{lon_name}' not found in the dataset.")
    # record attributes
    lon_attrs = dataset[lon_name].attrs.copy()

    # adjust longitude
    dataset = dataset.assign_coords(
        {lon_name: convert_360_to_180(dataset[lon_name])}
    ).sortby(lon_name)
    dataset[lon_name].attrs = lon_attrs

    # update attributes of data variables
    for var in dataset.data_vars.keys():
        if limited_area:
            # get old attribute values
            old_lon_first_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfFirstGridPointInDegrees"
            )
            old_lon_last_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfLastGridPointInDegrees"
            )
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": convert_360_to_180(
                        old_lon_first_grid
                    ),
                    "GRIB_longitudeOfLastGridPointInDegrees": convert_360_to_180(
                        old_lon_last_grid
                    ),
                }
            )
        else:
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-179.9),
                    "GRIB_longitudeOfLastGridPointInDegrees": np.float64(180.0),
                }
            )

    return dataset


def convert_to_celsius(temperature_kelvin: T) -> T:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin (T): Temperature in Kelvin,
            accessed through t2m variable in the dataset.

    Returns:
        T: Temperature in Celsius.
    """
    return temperature_kelvin - 273.15


def convert_to_celsius_with_attributes(
    dataset: xr.Dataset,
    inplace: bool = True,
    var_name: str = "t2m",
) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing temperature in Kelvin.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is True.
        var_name (str): Name of the temperature variable in the dataset.
            Default is "t2m".

    Returns:
        xr.Dataset: Dataset with temperature converted to Celsius.
    """
    if var_name not in dataset.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in the dataset.")
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert temperature variable
    dataset[var_name] = convert_to_celsius(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "C",
            "units": "C",
        }
    )

    return dataset


def rename_coords(dataset: xr.Dataset, coords_mapping: dict) -> xr.Dataset:
    """Rename coordinates in the dataset based on a mapping.

    Args:
        dataset (xr.Dataset): Dataset with coordinates to rename.
        coords_mapping (dict): Mapping of old coordinate names to new names.

    Returns:
        xr.Dataset: A new dataset with renamed coordinates.
    """
    coords_mapping_check = (
        isinstance(coords_mapping, dict)
        and bool(coords_mapping)
        and all(
            isinstance(old_name, str) and isinstance(new_name, str)
            for old_name, new_name in coords_mapping.items()
        )
    )
    if not coords_mapping_check:
        raise ValueError(
            "coords_mapping must be a non-empty dictionary of {old_name: new_name} pairs."
        )

    for old_name, new_name in coords_mapping.items():
        if old_name in dataset.coords:
            dataset = dataset.rename({old_name: new_name})
        else:
            warnings.warn(
                f"Coordinate '{old_name}' not found in the dataset and will be skipped.",
                UserWarning,
            )

    return dataset


def convert_m_to_mm(precipitation: T) -> T:
    """Convert precipitation from meters to millimeters.

    Args:
        precipitation (T): Precipitation in meters.

    Returns:
        T: Precipitation in millimeters.
    """
    return precipitation * 1000.0


def convert_m_to_mm_with_attributes(
    dataset: xr.Dataset, inplace: bool = True, var_name: str = "tp"
) -> xr.Dataset:
    """Convert precipitation from meters to millimeters and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing precipitation in meters.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is True.
        var_name (str): Name of the precipitation variable in the dataset.
            Default is "tp".

    Returns:
        xr.Dataset: Dataset with precipitation converted to millimeters.
    """
    if var_name not in dataset.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in the dataset.")
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert precipitation variable
    dataset[var_name] = convert_m_to_mm(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "mm",
            "units": "mm",
        }
    )

    return dataset


def check_downsample_condition(
    dataset: xr.Dataset,
    new_resolution: float,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
) -> float:
    """Check if downsampling conditions are met.

    Args:
        dataset (xr.Dataset): Dataset to check downsampling conditions.
        new_resolution (float): Desired new resolution in degrees.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.

    Raises:
        ValueError: If coordinate names are incorrect, new resolution is non-positive,
            new resolution is not greater than old resolution,
            or agg_funcs is not None and not a dictionary.

    Returns:
        float: Old resolution in degrees.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )
    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution <= old_resolution:
        raise ValueError(
            f"To downsample, degree of new resolution {new_resolution} "
            "should be greater than {old_resolution}."
        )

    if agg_funcs is not None and not isinstance(agg_funcs, dict):
        raise ValueError(
            "agg_funcs must be a dictionary of variable names and aggregation functions."
        )

    return old_resolution


def check_agg_funcs(agg_funcs: Dict[str, str], valid_agg_funcs: set) -> None:
    """Check if aggregation functions are valid.

    Args:
        agg_funcs (Dict[str, str]): Aggregation functions for each variable.
        valid_agg_funcs (set): Set of valid aggregation function names.

    Raises:
        ValueError: If any aggregation function is not valid or agg_funcs is not a dictionary.
    """
    if agg_funcs is None:
        return

    if not agg_funcs or not isinstance(agg_funcs, dict):
        raise ValueError(
            "agg_funcs must be a dictionary of variable names and aggregation function names."
        )

    invalid_funcs = set(agg_funcs.values()) - valid_agg_funcs
    if invalid_funcs:
        raise ValueError(
            f"Aggregation functions '{invalid_funcs}' are not valid. "
            f"Valid options are: {valid_agg_funcs}."
        )


def downsample_resolution_with_xarray(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
            Possible keys are:
                * `mean`
                * `sum`
                * `max`
                * `min`

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    # check aggregation functions
    valid_agg_funcs = {"mean", "sum", "max", "min"}
    check_agg_funcs(agg_funcs, valid_agg_funcs)

    old_resolution = check_downsample_condition(
        dataset,
        new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
        agg_funcs=agg_funcs,
    )

    weight = int(np.ceil(new_resolution / old_resolution))
    dim_kwargs = {
        lon_name: weight,
        lat_name: weight,
    }

    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "mean")

    result = {}
    for var in dataset.data_vars:
        func_str = agg_funcs.get(var, None)

        if func_str is None:
            warnings.warn(
                f"No aggregation function found for variable '{var}'. Using mean.",
                UserWarning,
            )
            func_str = "mean"

        # apply coarsening and reduction per variable
        result[var] = (
            dataset[var]
            .coarsen(**dim_kwargs, boundary="trim")
            .reduce(getattr(np, func_str))  # np.mean, np.sum, etc.
        )
        result[var].attrs = dataset[var].attrs.copy()

    # copy attributes of the dataset
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def align_lon_lat_with_popu_data(
    dataset: xr.Dataset,
    expected_longitude_max: np.float64 = np.float64(179.75),
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Align longitude and latitude coordinates with population data\
    of the same resolution.
    This function is specifically designed to ensure that the
    longitude and latitude coordinates in the dataset match the expected
    values used in population data, which are:
    - Longitude: -179.75 to 179.75, 720 points
    - Latitude: 89.75 to -89.75, 360 points

    Args:
        dataset (xr.Dataset): Dataset with longitude and latitude coordinates.
        expected_longitude_max (np.float64): Expected maximum longitude
            after adjustment. Default is np.float64(179.75).
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".

    Returns:
        xr.Dataset: Dataset with adjusted longitude and latitude coordinates.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    old_longitude_min = dataset[lon_name].min().values
    old_longitude_max = dataset[lon_name].max().values

    # TODO: find a more general solution
    special_case = (
        np.isclose(expected_longitude_max, np.float64(179.75))
        and np.isclose(old_longitude_min, np.float64(-179.7))
        and np.isclose(old_longitude_max, np.float64(179.8))
    )
    if special_case:
        offset = expected_longitude_max - old_longitude_max

        # adjust coord values
        dataset = dataset.assign_coords(
            {
                lon_name: (dataset[lon_name] + offset).round(2),
                lat_name: (dataset[lat_name] + offset).round(2),
            }
        )

    return dataset


def downsample_resolution_with_xesmf(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    new_min_lat: float | None = None,
    new_max_lat: float | None = None,
    new_min_lon: float | None = None,
    new_max_lon: float | None = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset using xESMF.
    Ref: https://xesmf.readthedocs.io/en/stable/notebooks/Rectilinear_grid.html

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        new_min_lat (float): Minimum latitude of the new grid. Default is None.
        new_max_lat (float): Maximum latitude of the new grid. Default is None.
        new_min_lon (float): Minimum longitude of the new grid. Default is None.
        new_max_lon (float): Maximum longitude of the new grid. Default is None.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation is used, i.e. `bilinear` for all variables.
            Possible keys are:
                * `bilinear`
                * `conservative`, need grid corner information
                * `conservative_normed`, need grid corner information
                * `patch`
                * `nearest_s2d`
                * `nearest_d2s`

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """

    def _get_default_values(val: float | None, arr: xr.DataArray, func: str) -> float:
        # using item() instead of values is also possible,
        # but only works if the result of func is a single value
        return getattr(arr, func)().values if val is None else val

    # check aggregation functions
    valid_agg_funcs = {
        "bilinear",
        "conservative",
        "conservative_normed",
        "patch",
        "nearest_s2d",
        "nearest_d2s",
    }
    check_agg_funcs(agg_funcs, valid_agg_funcs)

    old_res = check_downsample_condition(
        dataset,
        new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    new_min_lat = _get_default_values(new_min_lat, dataset[lat_name], "min")
    new_max_lat = _get_default_values(new_max_lat, dataset[lat_name], "max")
    new_min_lon = _get_default_values(new_min_lon, dataset[lon_name], "min")
    new_max_lon = _get_default_values(new_max_lon, dataset[lon_name], "max")

    # prepare the new dataset
    min_num = 0.001
    new_lats = np.arange(new_max_lat, new_min_lat - min_num, -new_resolution)
    new_lons = np.arange(new_min_lon, new_max_lon + min_num, new_resolution)
    new_grid = xr.Dataset(
        {
            lat_name: ([lat_name], new_lats, dataset[lat_name].attrs),
            lon_name: ([lon_name], new_lons, dataset[lon_name].attrs),
        }
    )

    # define regridders
    # each regridder for each function defined in agg_funcs
    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "bilinear")

    # TODO: check this again!
    # create grid corners for conservative regridding
    if {"conservative", "conservative_normed"} & set(agg_funcs.values()):
        if "lat_b" not in dataset.coords or "lon_b" not in dataset.coords:
            old_lat = dataset[lat_name].values
            old_lon = dataset[lon_name].values

            old_lat_b = np.arange(
                max(old_lat) + old_res, min(old_lat) - old_res, -old_res
            )
            old_lon_b = np.arange(
                min(old_lon) - old_res, max(old_lon) + old_res, old_res
            )
            dataset = dataset.assign_coords(
                {
                    "lat_b": (
                        ["lat_b"],
                        old_lat_b,
                        dataset[lat_name].attrs,
                    ),
                    "lon_b": (
                        ["lon_b"],
                        old_lon_b,
                        dataset[lon_name].attrs,
                    ),
                }
            )
        if "lat_b" not in new_grid.coords or "lon_b" not in new_grid.coords:
            new_lat_b = np.arange(
                max(new_lats) + new_resolution,
                min(new_lats) - new_resolution,
                -new_resolution,
            )
            new_lon_b = np.arange(
                min(new_lons) - new_resolution,
                max(new_lons) + new_resolution,
                new_resolution,
            )
            new_grid = new_grid.assign_coords(
                {
                    "lat_b": (
                        ["lat_b"],
                        new_lat_b,
                        dataset[lat_name].attrs,
                    ),
                    "lon_b": (
                        ["lon_b"],
                        new_lon_b,
                        dataset[lon_name].attrs,
                    ),
                }
            )

    # avoid creating duplicate regridders
    unique_funcs = set(agg_funcs.values()).union({"bilinear"})  # default aggregation
    regridder_dict = {}
    regridder_var_dict = {}
    for func in unique_funcs:
        regridder_dict[func] = xe.Regridder(dataset, new_grid, func, periodic=True)

    for var in agg_funcs.keys():
        regridder_var_dict[var] = regridder_dict[agg_funcs[var]]

    # apply regridders to data variables
    result = {}
    for var in dataset.data_vars:
        regridder_func = regridder_var_dict.get(var, None)
        if regridder_func is None:
            warnings.warn(
                f"No aggregation function found for variable '{var}'. Using bilinear.",
                UserWarning,
            )
            regridder_func = regridder_dict["bilinear"]  # default aggregation

        result[var] = regridder_func(dataset[var], keep_attrs=True)
    # create a new dataset with the regridded variables
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def downsample_resolution_with_cdo(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    new_min_lat: float | None = None,
    new_lat_size: int | None = None,
    new_min_lon: float | None = None,
    new_lon_size: int | None = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
    gridtype: Literal["gaussian", "lonlat", "curvilinear", "unstructured"] = "lonlat",
) -> xr.Dataset:
    """Downsample the resolution of a dataset using CDO.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        new_min_lat (float): Minimum latitude of the new grid. Default is None.
        new_lat_size (int): Size of latitude of the new grid. Default is None.
        new_min_lon (float): Minimum longitude of the new grid. Default is None.
        new_lon_size (int): Size of longitude of the new grid. Default is None.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation is used, i.e. `bil` (bilinear). Default is None.
            Possible keys are:
                * `nn` (nearest neighbor),
                * `bil` (bilinear),
                * `bic` (bicubic),
                * `con` (conservative),
                * `con2` (conservative 2nd order).
        gridtype (Literal["gaussian", "lonlat", "curvilinear", "unstructured"]):
            Type of the grid. Default is "lonlat".

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """

    # helper functions
    def _get_min_value(val: float | None, arr: xr.DataArray) -> float:
        return arr.min().item() if val is None else val

    def _get_size_value(val: float | None, arr: xr.DataArray, res: float) -> float:
        size = int(np.round((arr.max() - arr.min()).item() / res, 0)) + 1
        return size if val is None else val

    # check downsampling condition
    _ = check_downsample_condition(
        dataset,
        new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    # check aggregation functions
    valid_agg_funcs = {"nn", "bil", "bic", "con", "con2"}
    check_agg_funcs(agg_funcs, valid_agg_funcs)

    # prepare new grid parameters
    new_min_lat = _get_min_value(new_min_lat, dataset[lat_name])
    new_lat_size = _get_size_value(new_lat_size, dataset[lat_name], new_resolution)
    new_min_lon = _get_min_value(new_min_lon, dataset[lon_name])
    new_lon_size = _get_size_value(new_lon_size, dataset[lon_name], new_resolution)

    # prepare aggregation functions
    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "bil")

    # make sure the dataset works with CDO
    # i.e. having "lat" and "lon" as coordinate names
    # and Conventions attribute set to "CF-1.7"
    old_lat_name = lat_name
    old_lon_name = lon_name
    dataset = dataset.rename({old_lat_name: "lat", old_lon_name: "lon"})
    dataset.attrs.update({"Conventions": "CF-1.7"})

    # split dataset into individual data variables and save to temporary files
    ds_tmp_files = {}
    for var in dataset.data_vars:
        tmp_file = tempfile.NamedTemporaryFile(suffix=f"_{var}.nc", delete=False)
        tmp_file_name = tmp_file.name
        tmp_file.close()  # so xarray can write to it
        dataset[[var]].to_netcdf(
            tmp_file_name
        )  # use [[var]] to keep as dataset with coords
        ds_tmp_files[var] = tmp_file_name

    # prepare gridspec file
    gridspec = f"""
        gridtype = {gridtype}
        xfirst = {new_min_lon}
        xinc = {new_resolution}
        xsize = {new_lon_size}
        yfirst = {new_min_lat}
        yinc = {new_resolution}
        ysize = {new_lat_size}
    """
    gridspec = textwrap.dedent(gridspec).strip()
    gridspec_file = tempfile.NamedTemporaryFile(suffix="_gridspec.txt", delete=False)
    gridspec_file_name = gridspec_file.name
    gridspec_file.write(gridspec.encode())
    gridspec_file.close()

    # apply cdo remap to each variable file
    tmp_dss = {}
    cdo = Cdo()
    for var, tmp_file_name in ds_tmp_files.items():
        agg_func = agg_funcs.get(var, None)
        if agg_func is None:
            warnings.warn(
                f"No aggregation function found for variable '{var}'. Using bilinear.",
                UserWarning,
            )
            agg_func = "bil"

        try:
            tmp_ds = getattr(cdo, f"remap{agg_func}")(
                gridspec_file_name,
                input=tmp_file_name,
                returnXDataset=True,
            )
            tmp_dss[var] = tmp_ds
        except Exception as e:
            raise RuntimeError(
                f"CDO remapping failed for variable '{var}' with error: {e}"
            ) from e
        finally:
            # remove temporary variable file
            Path(tmp_file_name).unlink()

    # remove temporary gridspec file
    Path(gridspec_file_name).unlink()

    # create a new dataset with the regridded data
    result_dataset = xr.merge(tmp_dss.values())

    # restore original coordinate names
    result_dataset = result_dataset.rename({"lat": old_lat_name, "lon": old_lon_name})

    return result_dataset


def upsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.1,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Upsample the resolution of a dataset using `xarray.interp`.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.1.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used.
            Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )
    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution >= old_resolution:
        raise ValueError(
            f"To upsample, degree of new resolution {new_resolution} "
            "should be smaller than {old_resolution}."
        )

    lat_min, lat_max = (
        dataset[lat_name].min().values,
        dataset[lat_name].max().values,
    )
    lon_min, lon_max = (
        dataset[lon_name].min().values,
        dataset[lon_name].max().values,
    )
    updated_lat = np.arange(lat_min, lat_max + new_resolution, new_resolution)
    updated_lon = np.arange(lon_min, lon_max + new_resolution, new_resolution)
    updated_coords = {
        lat_name: updated_lat,
        lon_name: updated_lon,
    }

    if method_map is None:
        method_map = dict.fromkeys(dataset.data_vars, "linear")
    elif not isinstance(method_map, dict):
        raise ValueError(
            "method_map must be a dictionary of variable names and interpolation methods."
        )

    # interpolate each variable
    result = {}
    for var in dataset.data_vars:
        method = method_map.get(var, "linear")
        result[var] = dataset[var].interp(**updated_coords, method=method)
        result[var].attrs = dataset[var].attrs.copy()

    # create a new dataset with the interpolated variables
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


@dataclass
class ResolutionConfig:
    """Configuration for resolution resampling.

    Attributes:
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        downsample_lib (Literal["xarray", "xesmf", "cdo"]): Library to use for downsampling.
            Options are "xarray", "xesmf", or "cdo". Default is "xesmf".
        downsample_agg_funcs (Dict[str, str] | None): Aggregation function for each variable.
            If None, default aggregation of corresponding library is used. Default is None.
        upsample_method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used. Default is None.
    """

    new_resolution: float = 0.5
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    downsample_lib: Literal["xarray", "xesmf", "cdo"] = "xesmf"
    downsample_agg_funcs: Dict[str, str] | None = None
    upsample_method_map: Dict[str, str] | None = None


@dataclass
class GridConfig:
    """Configuration for grid specification for resampling.

    Attributes:
        expected_longitude_max_xarray (np.float64): Expected maximum longitude.
            Default is np.float64(179.75).
            This is used to adjust the grid after resampling with xarray,
            e.g. to align with population data.
        new_min_lat (float | None): Minimum latitude of the new grid. Default is None.
            This is used for resampling with xESMF and CDO.
        new_max_lat (float | None): Maximum latitude of the new grid. Default is None.
            This is used for resampling with xESMF.
        new_min_lon (float | None): Minimum longitude of the new grid. Default is None.
            This is used for resampling with xESMF and CDO.
        new_max_lon (float | None): Maximum longitude of the new grid. Default is None.
            This is used for resampling with xESMF.
        new_lat_size (int | None): Size of latitude of the new grid. Default is None.
            This is used for resampling with CDO.
        new_lon_size (int | None): Size of longitude of the new grid. Default is None.
            This is used for resampling with CDO.
        gridtype (Literal["gaussian", "lonlat", "curvilinear", "unstructured"]):
            Type of the grid. Default is "lonlat".
            This is used for resampling with CDO.
    """

    expected_longitude_max_xarray: np.float64 = np.float64(179.75)
    new_min_lat: float | None = None
    new_max_lat: float | None = None
    new_min_lon: float | None = None
    new_max_lon: float | None = None
    new_lat_size: int | None = None
    new_lon_size: int | None = None
    gridtype: Literal["gaussian", "lonlat", "curvilinear", "unstructured"] = "lonlat"


def resample_resolution(
    dataset: xr.Dataset,
    resolution_config: ResolutionConfig = ResolutionConfig(),
    grid_config: GridConfig = GridConfig(),
) -> xr.Dataset:
    """Resample the grid of a dataset to a new resolution.

    Args:
        dataset (xr.Dataset): Dataset to resample.
        resolution_config (ResolutionConfig): Configuration for resolution resampling.
        grid_config (GridConfig): Configuration for grid specification.

    Returns:
        xr.Dataset: Resampled dataset with changed resolution.
    """
    new_resolution = resolution_config.new_resolution
    lat_name = resolution_config.lat_name
    lon_name = resolution_config.lon_name
    downsample_lib = resolution_config.downsample_lib
    downsample_agg_funcs = resolution_config.downsample_agg_funcs
    upsample_method_map = resolution_config.upsample_method_map

    expected_longitude_max = grid_config.expected_longitude_max_xarray
    new_min_lat = grid_config.new_min_lat
    new_max_lat = grid_config.new_max_lat
    new_min_lon = grid_config.new_min_lon
    new_max_lon = grid_config.new_max_lon
    new_lat_size = grid_config.new_lat_size
    new_lon_size = grid_config.new_lon_size
    gridtype = grid_config.gridtype

    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution > old_resolution:
        if downsample_lib == "xarray":
            dataset = downsample_resolution_with_xarray(
                dataset,
                new_resolution=new_resolution,
                lat_name=lat_name,
                lon_name=lon_name,
                agg_funcs=downsample_agg_funcs,
            )
            return align_lon_lat_with_popu_data(
                dataset,
                expected_longitude_max=expected_longitude_max,
                lat_name=lat_name,
                lon_name=lon_name,
            )
        elif downsample_lib == "xesmf":
            return downsample_resolution_with_xesmf(
                dataset,
                new_resolution=new_resolution,
                new_min_lat=new_min_lat,
                new_max_lat=new_max_lat,
                new_min_lon=new_min_lon,
                new_max_lon=new_max_lon,
                lat_name=lat_name,
                lon_name=lon_name,
                agg_funcs=downsample_agg_funcs,
            )
        elif downsample_lib == "cdo":
            return downsample_resolution_with_cdo(
                dataset,
                new_resolution=new_resolution,
                new_min_lat=new_min_lat,
                new_lat_size=new_lat_size,
                new_min_lon=new_min_lon,
                new_lon_size=new_lon_size,
                lat_name=lat_name,
                lon_name=lon_name,
                agg_funcs=downsample_agg_funcs,
                gridtype=gridtype,
            )
        else:
            raise ValueError("lib must be one of 'xarray', 'xesmf', or 'cdo'.")

    return upsample_resolution(
        dataset,
        new_resolution=new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
        method_map=upsample_method_map,
    )


def shift_time(
    dataset: xr.Dataset,
    offset: int = -1,
    time_unit: Literal["W", "D", "h", "m", "s", "ms", "ns"] = "D",
    var_name: str = "time",
):
    """Shift the time coordinate of a dataset by a specified timedelta.
    The dataset is overwritten with the shifted time values.

    Args:
        dataset (xr.Dataset): Dataset to shift.
        offset (int): Amount to shift the time coordinate. Default is -1.
        time_unit (Literal["W", "D", "h", "m", "s", "ms", "ns"]):
            Time unit for the shift. Default is "D".
        var_name (str): Name of the time variable in the dataset. Default is "time".
    """
    if var_name not in dataset.coords:
        raise ValueError(f"Coordinate '{var_name}' not found in dataset.")

    if not isinstance(offset, int):
        raise ValueError("Offset value must be an int.")

    if time_unit not in ["W", "D", "h", "m", "s", "ms", "ns"]:
        raise ValueError(
            "time_unit must be one of 'W', 'D', 'h', 'm', 's', 'ms', or 'ns'."
        )

    dataset[var_name] = dataset[var_name] + np.timedelta64(offset, time_unit).astype(
        "timedelta64[ns]"
    )
    return dataset


def _parse_date(date: str | np.datetime64 | None) -> np.datetime64 | None:
    """Parse a date from string or numpy datetime64 to numpy datetime64.
    If the input is None, return None.

    Args:
        date (str | np.datetime64 | None): Date to parse.
            The string should be in the format "YYYY-MM-DD".

    Returns:
        np.datetime64 | None: Parsed date as numpy datetime64 or None.
    """
    if date is None:
        return None

    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if isinstance(date, str):
        if not re.match(date_pattern, date):
            raise ValueError("Date string must be in the format 'YYYY-MM-DD'.")
        try:
            date = np.datetime64(date, "ns")
        except ValueError as e:
            raise ValueError(f"Invalid date value. Error: {e}")

    if not isinstance(date, np.datetime64):
        raise ValueError("Date must be of type string, np.datetime64, or None.")

    return date


def truncate_data_by_time(
    dataset: xr.Dataset,
    start_date: Union[str, np.datetime64],
    end_date: Union[str, np.datetime64, None] = None,
    var_name: str = "time",
) -> xr.Dataset:
    """Truncate data from a specific start date to an end date. Both dates are inclusive.

    Args:
        dataset (xr.Dataset): Dataset to truncate.
        start_date (Union[str, np.datetime64]): Start date for truncation.
            Format as "YYYY-MM-DD" or as a numpy datetime64 object.
        end_date (Union[str, np.datetime64, None]): End date for truncation.
            Format as "YYYY-MM-DD" or as a numpy datetime64 object.
            If None, truncate until the last date in the dataset. Default is None.
        var_name (str): Name of the time variable in the dataset. Default is "time".

    Returns:
        xr.Dataset: Dataset truncated from the specified start date.
    """
    start_date = _parse_date(start_date)
    end_date = _parse_date(end_date)

    if start_date is None:
        raise ValueError("Start date must be provided and cannot be None.")

    if var_name not in dataset.data_vars and var_name not in dataset.coords:
        raise ValueError(f"The variable '{var_name}' not found in the dataset.")

    if end_date is None:
        end_date = dataset[var_name].max().values

    if start_date > end_date:
        raise ValueError(
            "The start date must be earlier than or equal to the end date."
        )

    return dataset.sel({var_name: slice(start_date, end_date)})


def _check_month_start_data(times: xr.DataArray) -> bool:
    """Check if all time points are at the start of the month.
    E.g. 2016-01-01, 2016-02-01, ..., 2017-01-01, 2018-01-01 ...

    Args:
        times (xr.DataArray): Time coordinate to check.

    Returns:
        bool: True if all time points are at the start of the month, False otherwise.
    """
    days = times.dt.day.values

    # check if all days are 1
    if not np.all(days == 1):
        return False

    return True


def calculate_monthly_precipitation(
    dataset: xr.Dataset, var_name: str = "tp", time_coord: str = "time"
) -> xr.Dataset:
    """Calculate monthly total precipitation from data downloaded from ERA5-Land monthly data.
    The real precipitation of the month = downloaded value * number of days in the month.

    Args:
        dataset (xr.Dataset): Dataset with total precipitation data.
        var_name (str): Name of the precipitation variable in the dataset. Default is "tp".
        time_coord (str): Name of the time coordinate in the dataset. Default is "time".

    Returns:
        xr.Dataset: Dataset with monthly total precipitation values.
    """
    # check inputs
    if time_coord not in dataset.coords:
        raise ValueError(f"Time coordinate '{time_coord}' not found in dataset.")

    if var_name not in dataset.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    times = dataset[time_coord]

    if not _check_month_start_data(times):
        raise ValueError("The dataset does not have month start data.")

    # calculate number of days in each month
    days_in_month = times.dt.days_in_month

    # calculate monthly total precipitation
    org_attrs = dataset[var_name].attrs.copy()
    dataset[var_name] = dataset[var_name] * days_in_month
    dataset[var_name].attrs = org_attrs

    return dataset


def _replace_decimal_point(degree: float) -> str:
    """Replace the decimal point in a degree string with 'p'
    if the degree is greater than or equal to 1.0,
    or remove it if the degree is less than 1.0.

    Args:
        degree (float): Degree value to convert.

    Returns:
        str: String representation of the degree without decimal point.
    """
    if not isinstance(degree, (float)):
        raise ValueError("Resolution degree must be a float.")
    if degree < 1.0:
        return str(degree).replace(".", "")
    else:
        return str(degree).replace(".", "p")


def _apply_preprocessing(
    dataset: xr.Dataset,
    file_name_base: str,
    settings: Dict[str, Any],
) -> Tuple[xr.Dataset, str]:
    """Apply preprocessing steps to the dataset based on settings.

    Args:
        dataset (xr.Dataset): Dataset to preprocess.
        file_name_base (str): Base name for the output file.
        settings (Dict[str, Any]): Settings for preprocessing.

    Returns:
        Tuple[xr.Dataset, str]: Preprocessed dataset and updated file name.
    """
    # get settings
    unify_coords = settings.get("unify_coords", False)
    unify_coords_fname = settings.get("unify_coords_fname")
    uni_coords = settings.get("uni_coords")

    adjust_longitude = settings.get("adjust_longitude", False)
    adjust_longitude_vname = settings.get("adjust_longitude_vname")
    adjust_longitude_fname = settings.get("adjust_longitude_fname")

    convert_kelvin_to_celsius = settings.get("convert_kelvin_to_celsius", False)
    convert_kelvin_to_celsius_vname = settings.get("convert_kelvin_to_celsius_vname")
    convert_kelvin_to_celsius_fname = settings.get("convert_kelvin_to_celsius_fname")

    convert_m_to_mm_precipitation = settings.get("convert_m_to_mm_precipitation", False)
    convert_m_to_mm_precipitation_vname = settings.get(
        "convert_m_to_mm_precipitation_vname"
    )
    convert_m_to_mm_precipitation_fname = settings.get(
        "convert_m_to_mm_precipitation_fname"
    )

    resample_grid = settings.get("resample_grid", False)
    resample_grid_vname = settings.get("resample_grid_vname")
    lat_name, lon_name = resample_grid_vname if resample_grid_vname else (None, None)
    resample_grid_fname = settings.get("resample_grid_fname")
    resample_degree = settings.get("resample_degree")
    downsample_agg_funcs = settings.get("downsample_agg_funcs", None)
    upsample_method_map = settings.get("upsample_method_map", None)
    resample_expected_longitude_max = settings.get(
        "downsample_max_lon_xarray", np.float64(179.75)
    )
    downsample_lib = settings.get("downsample_lib", "xesmf")
    new_min_lat = settings.get("downsample_new_min_lat", None)
    new_max_lat = settings.get("downsample_new_max_lat", None)
    new_min_lon = settings.get("downsample_new_min_lon", None)
    new_max_lon = settings.get("downsample_new_max_lon", None)
    new_lat_size = settings.get("downsample_new_lat_size", None)
    new_lon_size = settings.get("downsample_new_lon_size", None)
    gridtype = settings.get("downsample_gridtype", "lonlat")

    truncate_date = settings.get("truncate_date", False)
    truncate_date_from = settings.get("truncate_date_from")
    truncate_date_to = settings.get("truncate_date_to")
    truncate_date_vname = settings.get("truncate_date_vname")

    cal_monthly_tp = settings.get("cal_monthly_tp", False)
    cal_monthly_tp_vname = settings.get("cal_monthly_tp_vname")
    cal_monthly_tp_tcoord = settings.get("cal_monthly_tp_tcoord")
    cal_monthly_tp_fname = settings.get("cal_monthly_tp_fname")

    # define helper function
    def apply_step(
        ds: xr.Dataset,
        fname_base: str,
        step: Dict[str, Any],
        logger: logging.Logger,
    ) -> Tuple[xr.Dataset, str]:
        """Apply a preprocessing step to the dataset and update the file name."""
        if not step["condition"](ds):
            return ds, fname_base

        logger.info(step["message"])
        ds = step["transform"](ds)

        suffix = step.get("suffix")

        if suffix:
            fname_base += f"_{suffix}"

        return ds, fname_base

    # define steps with common structure
    pp_common_steps = [
        {
            "condition": lambda ds: unify_coords,
            "message": "Renaming coordinates to unify them across datasets...",
            "transform": lambda ds: rename_coords(ds, uni_coords),
            "suffix": unify_coords_fname,
        },
        {
            "condition": lambda ds: adjust_longitude
            and adjust_longitude_vname in ds.coords,
            "message": "Adjusting longitude from 0-360 to -180-180...",
            "transform": lambda ds: adjust_longitude_360_to_180(
                ds, lon_name=adjust_longitude_vname
            ),  # only consider full map for now, i.e. limited_area=False
            "suffix": adjust_longitude_fname,
        },
        {
            "condition": lambda ds: convert_kelvin_to_celsius
            and convert_kelvin_to_celsius_vname in ds.data_vars,
            "message": "Converting temperature from Kelvin to Celsius...",
            "transform": lambda ds: convert_to_celsius_with_attributes(
                ds, var_name=convert_kelvin_to_celsius_vname
            ),
            "suffix": convert_kelvin_to_celsius_fname,
        },
        {
            "condition": lambda ds: convert_m_to_mm_precipitation
            and convert_m_to_mm_precipitation_vname in ds.data_vars,
            "message": "Converting precipitation from meters to millimeters...",
            "transform": lambda ds: convert_m_to_mm_with_attributes(
                ds, var_name=convert_m_to_mm_precipitation_vname
            ),
            "suffix": convert_m_to_mm_precipitation_fname,
        },
        {
            "condition": lambda ds: cal_monthly_tp
            and all(
                (
                    cal_monthly_tp_vname in ds.data_vars,
                    cal_monthly_tp_tcoord in ds.coords,
                )
            ),
            "message": (
                "Calculating monthly total precipitation = "
                "downloaded data * number of days in month..."
            ),
            "transform": lambda ds: calculate_monthly_precipitation(
                ds,
                var_name=cal_monthly_tp_vname,
                time_coord=cal_monthly_tp_tcoord,
            ),
            "suffix": cal_monthly_tp_fname,
        },
    ]

    # apply common steps
    for step in pp_common_steps:
        dataset, file_name_base = apply_step(dataset, file_name_base, step, logger)

    # handle complex steps separately
    if resample_grid and all((lat_name in dataset.coords, lon_name in dataset.coords)):
        logger.info("Resampling grid to a new resolution...")
        dataset = resample_resolution(
            dataset,
            resolution_config=ResolutionConfig(
                new_resolution=resample_degree,
                lat_name=lat_name,
                lon_name=lon_name,
                downsample_lib=downsample_lib,
                downsample_agg_funcs=downsample_agg_funcs,
                upsample_method_map=upsample_method_map,
            ),
            grid_config=GridConfig(
                expected_longitude_max_xarray=resample_expected_longitude_max,
                new_min_lat=new_min_lat,
                new_max_lat=new_max_lat,
                new_min_lon=new_min_lon,
                new_max_lon=new_max_lon,
                new_lat_size=new_lat_size,
                new_lon_size=new_lon_size,
                gridtype=gridtype,
            ),
        )
        degree_str = _replace_decimal_point(resample_degree)
        file_name_base += f"_{degree_str}{resample_grid_fname}"

    if truncate_date and truncate_date_vname in dataset.coords:
        logger.info("Truncating data from a specific start date...")
        dataset = truncate_data_by_time(
            dataset,
            start_date=truncate_date_from,
            end_date=truncate_date_to,
            var_name=truncate_date_vname,
        )

        min_year = truncate_date_from[:4]
        max_time = dataset[truncate_date_vname].max().values
        end_date = truncate_date_to or max_time
        max_year = np.datetime64(end_date, "Y")
        file_name_base += f"_{min_year}-{max_year}"

    return dataset, file_name_base


def preprocess_data_file(
    netcdf_file: Path,
    source: Literal["era5", "isimip"] = "era5",
    settings: Path | str = "default",
    new_settings: Dict[str, Any] | None = None,
    unique_tag: str | None = None,
) -> Tuple[xr.Dataset, str]:
    """Preprocess the dataset based on provided settings.
    If the settings path is "default", use the default settings of the source.
    The settings and preprocessed files are saved in the directory,
    which is specified by the settings file and unique number.

    Args:
        netcdf_file (Path): Path to the NetCDF file to preprocess.
        source (Literal["era5", "isimip"]): Source of the data.
            Defaults to "era5".
        settings (Path | str): Path to the settings file or "default" for default settings.
        new_settings (Dict[str, Any] | None): Additional settings to overwrite defaults.
            Defaults to None.
        unique_tag (str | None): Unique tag to append to the output file name
            and settings file.
            Defaults to None.
    Returns:
        Tuple[xr.Dataset, str]: Preprocessed dataset and
            the name of the preprocessed file.
    """
    if not utils.is_non_empty_file(netcdf_file):
        raise ValueError(f"NetCDF file {netcdf_file} does not exist or is empty.")

    # generate unique tag for the settings file and output file
    if unique_tag is None or not unique_tag:
        unique_tag = utils.generate_unique_tag()

    # load settings
    settings, settings_fname = utils.load_settings(
        source=source, setting_path=settings, new_settings=new_settings
    )

    folder_path = Path(settings.get("output_dir", "data/processed"))
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    # save settings to a file
    settings_fname_w_tag = f"{settings_fname}_{unique_tag}.json"
    utils.save_settings_to_file(settings, folder_path, settings_fname_w_tag)

    # prepare to preprocess NetCDF file
    file_name = netcdf_file.stem
    file_name = file_name[: -len("_raw")] if file_name.endswith("_raw") else file_name
    file_ext = netcdf_file.suffix

    with xr.open_dataset(netcdf_file, chunks={}) as dataset:
        dataset, file_name_base = _apply_preprocessing(dataset, file_name, settings)
        # save the processed dataset
        output_file = folder_path / f"{file_name_base}_{unique_tag}{file_ext}"
        dataset.to_netcdf(output_file, mode="w", format="NETCDF4")
        logger.info(f"Processed dataset saved to: {output_file}")
        return dataset, str(output_file.name)


def _prepare_for_aggregation(
    dataset: xr.Dataset,
    normalize_time: bool = True,
    agg_dict: Dict[str, str] | None = None,
) -> Tuple[xr.Dataset, Dict[str, str]]:
    """
    Prepare the dataset for aggregation by:
        * normalizing time if needed,
        * preparing aggregation dictionary.

    Args:
        dataset (xr.Dataset): Dataset to prepare.
        normalize_time (bool): If True, normalize time to the beginning of the day.
            e.g. 2025-10-01T12:00:00 becomes 2025-10-01T00:00:00.
            Default is True.
        agg_dict (Dict[str, str] | None): Dictionary of aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used.

    Returns:
        Tuple[xr.Dataset, Dict[str, str]]: First item is the prepared dataset,
            with time is normalized if specified.
            The second item is the aggregation dictionary.
    """
    # normalize time if specified
    if normalize_time:
        dataset["time"] = dataset["time"].dt.floor("D")

    # prepare aggregation dictionary
    # get list of data variable names
    var_names = list(dataset.data_vars.keys())

    invalid_agg_dict = agg_dict is not None and (
        not isinstance(agg_dict, dict)
        or not all(
            isinstance(var, str) and isinstance(func, str)
            for var, func in agg_dict.items()
        )
        or (isinstance(agg_dict, dict) and len(agg_dict) == 0)
        or not all(var in var_names for var in agg_dict.keys())
    )
    if invalid_agg_dict or agg_dict is None:
        if invalid_agg_dict:
            warnings.warn(
                "Invalid agg_dict provided. Using default aggregation (mean) for all variables.",
                UserWarning,
            )
        # default aggregation is mean for each variable
        agg_dict = dict.fromkeys(var_names, "mean")

    return dataset, agg_dict


def _aggregate_netcdf_nuts_gpd(
    nuts_data: gpd.GeoDataFrame,
    nc_file: Path,
    agg_dict: Dict[str, str] | None,
    normalize_time: bool = True,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Aggregate NetCDF data by NUTS regions using GeoPandas (i.e. `sjoin`).

    Notes:
        * `sjoin` does not consider weights based on area overlap.
        * It is not recommended for very large datasets as it may consume a lot of memory,
            e.g. global datasets with fine resolution (0.1 deg)
            and long time series (24 months).

    Args:
        nuts_data (gpd.GeoDataFrame): GeoDataFrame containing NUTS data from shape file.
        nc_file (Path): Path to the NetCDF file.
        agg_dict (Dict[str, str] | None): Dictionary of aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used.
        normalize_time (bool): If True, normalize time to the beginning of the day.
            e.g. 2025-10-01T12:00:00 becomes 2025-10-01T00:00:00.
            Default is True.

    Returns:
        Tuple[pd.DataFrame, list[str]]: First item is aggregated DataFrame,
            with coordinates "NUTS_ID", "time", and
            data variables include aggregated data variables.
            The second item in the tuple is list of data variable names.
    """
    with xr.open_dataset(nc_file, chunks={"time": "auto"}) as dataset:
        # ensure the dataset has the required coordinates
        if not all(
            coord in dataset.coords for coord in ["latitude", "longitude", "time"]
        ):
            raise ValueError(
                f"NetCDF file '{nc_file}' must contain "
                f"'latitude', 'longitude', and 'time' coordinates."
            )

        # Raise error if the dataset is too large
        num_lat = dataset.sizes.get("latitude", 0)
        num_lon = dataset.sizes.get("longitude", 0)
        num_time = dataset.sizes.get("time", 0)
        num_vars = len(dataset.data_vars)

        if (
            num_lat * num_lon * num_time * num_vars
            > 360
            * 720
            * 24
            * 2  # global data for 2 years, 2 variables, 0.5 degree resolution
        ):  # e.g. global 0.1 deg for 1 year
            raise ValueError(
                f"The NetCDF file '{nc_file}' may be too large for "
                f"GeoPandas spatial join aggregation. "
                f"Consider using 'exactextract' library for aggregation. "
                f"Dataset size: {num_lat} lat x {num_lon} lon x {num_time} time points.",
            )

        # prepare dataset for aggregation
        dataset, agg_dict = _prepare_for_aggregation(dataset, normalize_time, agg_dict)
        r_var_names = list(agg_dict.keys())

        # Convert the NetCDF dataset to a GeoDataFrame
        nc_data = dataset.to_dataframe().reset_index()
        gpd_nc_data = gpd.GeoDataFrame(
            nc_data,
            geometry=gpd.points_from_xy(nc_data["longitude"], nc_data["latitude"]),
            crs=f"EPSG:{CRS}",
        )

        # merge nc data with NUTS data
        nc_data_merged = gpd.sjoin(
            gpd_nc_data, nuts_data, how="inner", predicate="intersects"
        )

        # drop NaN before grouping
        nc_data_merged = nc_data_merged[~nc_data_merged["NUTS_ID"].isna()]

        # group by NUTS_ID and time, aggregate using agg_dict
        nc_data_agg = nc_data_merged.groupby(["NUTS_ID", "time"], as_index=False).agg(
            agg_dict
        )

    return nc_data_agg, r_var_names


def _aggregate_netcdf_nuts_ee(
    nuts_data: gpd.GeoDataFrame,
    nc_file: Path,
    agg_dict: Dict[str, str] | None,
    normalize_time: bool = True,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Aggregate NetCDF data by NUTS regions using `exactextract`.

    Notes:
        * `exactextract` only consider non-`NaN` values during calculation
            * mean of all NaN values is NaN.
            * sum of all NaN values is 0.

    Args:
        nuts_data (gpd.GeoDataFrame): GeoDataFrame containing NUTS data from shape file.
        nc_file (Path): Path to the NetCDF file.
        agg_dict (Dict[str, str] | None): Dictionary of aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used.
        normalize_time (bool): If True, normalize time to the beginning of the day.
            e.g. 2025-10-01T12:00:00 becomes 2025-10-01T00:00:00.
            Default is True.

    Returns:
        Tuple[pd.DataFrame, list[str]]: First item is aggregated DataFrame,
            with coordinates "NUTS_ID", "time", and
            data variables include aggregated data variables.
            The second item in the tuple is list of data variable names.
    """
    with xr.open_dataset(nc_file, chunks={"time": "auto"}) as dataset:
        # ensure the dataset has the required coordinates
        if not all(
            coord in dataset.coords for coord in ["latitude", "longitude", "time"]
        ):
            raise ValueError(
                f"NetCDF file '{nc_file}' must contain "
                f"'latitude', 'longitude', and 'time' coordinates."
            )

        # prepare dataset for aggregation
        dataset, agg_dict = _prepare_for_aggregation(dataset, normalize_time, agg_dict)
        r_var_names = list(agg_dict.keys())

        # aggregate data for each time step
        data_agg_list = []
        for time_val in dataset["time"].values:
            data_agg_t = []
            for data_var in r_var_names:
                data_var_time = dataset[[data_var]].sel(time=time_val)

                convert_minus = False
                if "-" in data_var:
                    # rename data_var to discard `-` in the name
                    # exactextract does not support `-` in variable names
                    updated_data_var = data_var.replace("-", "_")
                    convert_minus = True
                else:
                    updated_data_var = data_var

                data_var_time_agg = ee.exact_extract(
                    data_var_time,
                    nuts_data,
                    f"{updated_data_var}={agg_dict[data_var]}",  # rename the agg column by data_var name
                    include_cols=["NUTS_ID"],
                    output="pandas",
                )
                data_var_time_agg["time"] = time_val  # add time column

                if convert_minus:
                    # change back the column name to original data_var name
                    data_var_time_agg = data_var_time_agg.rename(
                        columns={updated_data_var: data_var}
                    )

                data_agg_t.append(data_var_time_agg)
            data_agg_list.append(data_agg_t)

        # merge all aggregated dataframes
        # by NUTS_ID and time, along all data variables
        merged_dfs = [
            reduce(
                lambda left, right: pd.merge(
                    left, right, on=["NUTS_ID", "time"], how="outer", validate="1:1"
                ),
                data_t,
            )
            for data_t in data_agg_list
        ]
        # concatenate all merged dataframes
        # as they have different time steps
        assert len(merged_dfs) == len(dataset["time"].values)
        nc_data_agg = pd.concat(merged_dfs, ignore_index=True)

    return nc_data_agg, r_var_names


def _check_aggregation_inputs(
    netcdf_files: dict[str, tuple[Path, Dict[str, str] | None]],
    nuts_file: Path,
):
    """Check the inputs for aggregation function.

    Args:
        netcdf_files (dict[str, tuple[Path, Dict[str, str] | None]]): Dictionary of NetCDF files.
            Keys are dataset names and values are tuples of (file path, agg_dict).
            The agg_dict can contain aggregation options for each data variable.
            For example, {"t2m": "mean", "tp": "sum"}.
            If agg_dict is None, default aggregation (i.e. mean) is used.
            NetCDF files must contain "latitude", "longitude", and "time" coordinates.
        nuts_file (Path): Path to the NUTS regions shape file.
            The shape file has columns such as "NUTS_ID" and "geometry".
    """
    if not isinstance(netcdf_files, dict) or not netcdf_files:
        raise ValueError("netcdf_files must be a non-empty dictionary.")

    for netcdf_file in netcdf_files.values():
        if not utils.is_non_empty_file(netcdf_file[0]):
            raise ValueError(
                f"NetCDF file '{netcdf_file[0]}' is not valid path or empty."
            )
    if not utils.is_non_empty_file(nuts_file):
        raise ValueError("nuts_file must be a valid file path.")


def aggregate_data_by_nuts(
    netcdf_files: dict[str, tuple[Path, Dict | None]],
    nuts_file: Path,
    normalize_time: bool = True,
    output_dir: Path | None = None,
    agg_lib: Literal["geopandas", "exactextract"] = "exactextract",
) -> Path:
    """Aggregate data from NetCDF files by NUTS regions, data variable names, and time.
    The aggregated data is saved to a NetCDF file with coordinates "NUTS_ID", "time",
    and data variables include aggregated data variables.

    Args:
        netcdf_files (dict[str, tuple[Path, Dict | None]]): Dictionary of NetCDF files.
            Keys are dataset names and values are tuples of (file path, agg_dict).
            The agg_dict can contain aggregation options for each data variable.
            For example, {"t2m": "mean", "tp": "sum"}.
            If agg_dict is None, default aggregation (i.e. mean) is used.
            NetCDF files must contain "latitude", "longitude", and "time" coordinates.
        nuts_file (Path): Path to the NUTS regions shape file.
            The shape file has columns such as "NUTS_ID" and "geometry".
        normalize_time (bool): If True, normalize time to the beginning of the day.
            e.g. 2025-10-01T12:00:00 becomes 2025-10-01T00:00:00.
            Default is True.
        output_dir (Path | None): Directory to save the aggregated NetCDF file.
            If None, the output file is saved in the same directory as the NUTS file.
            Default is None.
        agg_lib (Literal["geopandas", "exactextract"]): Library to use for aggregation.
            Options are "geopandas" or "exactextract". Default is "exactextract".

    Returns:
        Path: Path to the aggregated NetCDF file.
    """
    # check inputs
    _check_aggregation_inputs(netcdf_files, nuts_file)

    # load data from the nuts shape file
    nuts_data = gpd.read_file(nuts_file)

    if "NUTS_ID" not in nuts_data.columns or "geometry" not in nuts_data.columns:
        raise ValueError(
            "NUTS_ID and geometry columns must be present in the nuts shape file."
        )

    # set the base name for the output file
    out_file_name = nuts_file.stem.replace(
        ".shp", ""
    )  # replace .shp (if any) with empty string
    out_file_name = out_file_name + "_agg"

    # load data from the NetCDF file
    # merge nuts data with aggregated NetCDF data
    out_data = nuts_data["NUTS_ID"].to_frame()  # start with all NUTS_IDs
    agg_var_names = []
    first_merge = True
    for ds_name, file_info in netcdf_files.items():
        file_path, agg_dict = file_info
        logger.info(f"Processing NetCDF file: {file_path}")

        if agg_lib == "geopandas":
            nc_data_agg, r_var_names = _aggregate_netcdf_nuts_gpd(
                nuts_data,
                file_path,
                agg_dict,
                normalize_time=normalize_time,
            )
        elif agg_lib == "exactextract":
            nc_data_agg, r_var_names = _aggregate_netcdf_nuts_ee(
                nuts_data,
                file_path,
                agg_dict,
                normalize_time=normalize_time,
            )
        else:
            raise ValueError("agg_lib must be one of 'geopandas' or 'exactextract'.")

        # merge nuts data with aggregated NetCDF data
        out_columns = set(out_data.columns) - {"NUTS_ID", "time"}
        nc_columns = set(nc_data_agg.columns) - {"NUTS_ID", "time"}

        if first_merge:
            out_data = out_data.merge(nc_data_agg, on=["NUTS_ID"], how="outer")
            first_merge = False

        elif out_columns.isdisjoint(nc_columns):
            # if the next NetCDF file has different data variable names,
            # merge the data
            out_data = out_data.merge(
                nc_data_agg, on=["NUTS_ID", "time"], how="outer", validate="1:1"
            )

        else:
            # if there are overlapping data variable names,
            # merge first with new values suffixed by '_new'
            out_data = out_data.merge(
                nc_data_agg,
                on=["NUTS_ID", "time"],
                how="outer",
                validate="1:1",
                suffixes=("", "_new"),
            )
            # update old data variable values with new values
            for var in out_columns.intersection(nc_columns):
                out_data[var] = out_data[var + "_new"].combine_first(out_data[var])

            # drop the new suffixed columns
            out_data = out_data.drop(
                columns=[var + "_new" for var in out_columns.intersection(nc_columns)]
            )

        # update the output file name
        out_file_name += f"_{ds_name}"

        agg_var_names = agg_var_names + [
            name for name in r_var_names if name not in agg_var_names
        ]

    # filter the merged data to keep only
    # NUTS_ID, time, and aggregated data variables
    out_data_filtered = out_data[["NUTS_ID", "time"] + agg_var_names]

    # convert the GeoDataFrame to a NetCDF file
    ds_out = out_data_filtered.set_index(["NUTS_ID", "time"]).to_xarray()

    # update out put file name
    min_time = str(ds_out.time.min().values)[:7]
    max_time = str(ds_out.time.max().values)[:7]
    min_max_time = f"{min_time}-{max_time}"
    out_file_name += f"_{min_max_time}.nc"

    # save the aggregated dataset to a NetCDF file
    if output_dir is None:
        output_dir = nuts_file.parent
    output_file = output_dir / out_file_name
    ds_out.to_netcdf(output_file, mode="w")
    logger.info(f"Aggregated data saved to: {output_file}")

    return output_file
