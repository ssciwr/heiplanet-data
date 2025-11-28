from typing import TypeVar, Union, Callable, Dict, Any, Tuple, Literal
import xarray as xr
import numpy as np
import warnings
from pathlib import Path
from heiplanet_data import utils
import geopandas as gpd
import pandas as pd
import re
import xesmf as xe


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
    inplace: bool = False,
    var_name: str = "t2m",
) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing temperature in Kelvin.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
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
    dataset: xr.Dataset, inplace: bool = False, var_name: str = "tp"
) -> xr.Dataset:
    """Convert precipitation from meters to millimeters and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing precipitation in meters.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
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


def downsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
    agg_map: Dict[str, Callable[[Any], float]] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
        agg_map (Dict[str, Callable[[Any], float]] | None): Mapping of string
            to aggregation functions.
            If None, default mapping is used. Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
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

    if agg_map is None:
        agg_map = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
        }
    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "mean")

    result = {}
    for var in dataset.data_vars:
        func_str = agg_funcs.get(var, "mean")
        func = agg_map.get(func_str, np.mean)

        # apply coarsening and reduction per variable
        result[var] = dataset[var].coarsen(**dim_kwargs, boundary="trim").reduce(func)
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
    min_lat: float = -89.75,
    max_lat: float = 89.75,
    min_lon: float = -179.75,
    max_lon: float = 179.75,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset using xESMF.
    Ref: https://xesmf.readthedocs.io/en/stable/notebooks/Rectilinear_grid.html

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        min_lat (float): Minimum latitude of the new grid. Default is -89.75.
        min_lon (float): Minimum longitude of the new grid. Default is -179.75.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation is used, i.e. `bilinear` for all variables.
    """
    old_res = check_downsample_condition(
        dataset,
        new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    # prepare the new dataset
    min_num = 0.001
    new_lats = np.arange(max_lat, min_lat - min_num, -new_resolution)
    new_lons = np.arange(min_lon, max_lon + min_num, new_resolution)
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
    if "conservative" in agg_funcs.values():
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
    unique_funcs = set(agg_funcs.values())
    regridder_dict = {}
    regridder_var_dict = {}
    for func in unique_funcs:
        regridder_dict[func] = xe.Regridder(dataset, new_grid, func, periodic=True)

    for var in agg_funcs.keys():
        regridder_var_dict[var] = regridder_dict[agg_funcs[var]]

    # apply regridders to data variables
    result = {}
    for var in dataset.data_vars:
        if var in regridder_var_dict:
            result[var] = regridder_var_dict[var](dataset[var], keep_attrs=True)

    # create a new dataset with the regridded variables
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def upsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.1,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Upsample the resolution of a dataset.

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


def resample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
    agg_map: Dict[str, Callable[[Any], float]] | None = None,
    expected_longitude_max: np.float64 = np.float64(179.75),
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Resample the grid of a dataset to a new resolution.

    Args:
        dataset (xr.Dataset): Dataset to resample.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
        agg_map (Dict[str, Callable[[Any], float]] | None): Mapping of string
            to aggregation functions. If None, default mapping is used. Default is None.
        expected_longitude_max (np.float64): Expected maximum longitude
            after adjustment. Default is np.float64(179.75).
        method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used. Default is None.

    Returns:
        xr.Dataset: Resampled dataset with changed resolution.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution > old_resolution:
        dataset = downsample_resolution(
            dataset,
            new_resolution=new_resolution,
            lat_name=lat_name,
            lon_name=lon_name,
            agg_funcs=agg_funcs,
            agg_map=agg_map,
        )
        return align_lon_lat_with_popu_data(
            dataset,
            expected_longitude_max=expected_longitude_max,
            lat_name=lat_name,
            lon_name=lon_name,
        )

    return upsample_resolution(
        dataset,
        new_resolution=new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
        method_map=method_map,
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
    lat_name = resample_grid_vname[0] if resample_grid_vname else None
    lon_name = resample_grid_vname[1] if resample_grid_vname else None
    resample_grid_fname = settings.get("resample_grid_fname")
    resample_degree = settings.get("resample_degree")

    truncate_date = settings.get("truncate_date", False)
    truncate_date_from = settings.get("truncate_date_from")
    truncate_date_to = settings.get("truncate_date_to")
    truncate_date_vname = settings.get("truncate_date_vname")

    if unify_coords:
        print("Renaming coordinates to unify them across datasets...")
        dataset = rename_coords(dataset, uni_coords)
        file_name_base += f"_{unify_coords_fname}"

    if adjust_longitude and adjust_longitude_vname in dataset.coords:
        print("Adjusting longitude from 0-360 to -180-180...")
        dataset = adjust_longitude_360_to_180(
            dataset, lon_name=adjust_longitude_vname
        )  # only consider full map for now, i.e. limited_area=False
        file_name_base += f"_{adjust_longitude_fname}"

    if (
        convert_kelvin_to_celsius
        and convert_kelvin_to_celsius_vname in dataset.data_vars
    ):
        print("Converting temperature from Kelvin to Celsius...")
        dataset = convert_to_celsius_with_attributes(
            dataset, var_name=convert_kelvin_to_celsius_vname
        )
        file_name_base += f"_{convert_kelvin_to_celsius_fname}"

    if (
        convert_m_to_mm_precipitation
        and convert_m_to_mm_precipitation_vname in dataset.data_vars
    ):
        print("Converting precipitation from meters to millimeters...")
        dataset = convert_m_to_mm_with_attributes(
            dataset, var_name=convert_m_to_mm_precipitation_vname
        )
        file_name_base += f"_{convert_m_to_mm_precipitation_fname}"

    if resample_grid and lat_name in dataset.coords and lon_name in dataset.coords:
        print("Resampling grid to a new resolution...")
        dataset = resample_resolution(
            dataset,
            new_resolution=resample_degree,
            lat_name=lat_name,
            lon_name=lon_name,
        )  # agg_funcs, agg_map, and method_map are omitted for simplicity
        degree_str = _replace_decimal_point(resample_degree)
        file_name_base += f"_{degree_str}{resample_grid_fname}"

    if truncate_date and truncate_date_vname in dataset.coords:
        print("Truncating data from a specific start date...")
        dataset = truncate_data_by_time(
            dataset,
            start_date=truncate_date_from,
            end_date=truncate_date_to,
            var_name=truncate_date_vname,
        )

        min_year = truncate_date_from[:4]
        max_time = dataset[truncate_date_vname].max().values
        max_year = (
            truncate_date_to[:4] if truncate_date_to else np.datetime64(max_time, "Y")
        )
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
        print(f"Processed dataset saved to: {output_file}")
        return dataset, str(output_file.name)


def _aggregate_netcdf_nuts(
    nuts_data: gpd.GeoDataFrame,
    nc_file: Path,
    agg_dict: dict | None,
    normalize_time: bool = True,
) -> Tuple[gpd.GeoDataFrame, list[str]]:
    """
    Aggregate NetCDF data by NUTS regions.
    Left join is used to ensure that all NUTS regions are included,
    even if some regions do not have data in the NetCDF file.

    Args:
        nuts_data (gpd.GeoDataFrame): GeoDataFrame containing NUTS data from shape file.
        nc_file (Path): Path to the NetCDF file.
        agg_dict (dict | None): Dictionary of aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used.
        normalize_time (bool): If True, normalize time to the beginning of the day.
            e.g. 2025-10-01T12:00:00 becomes 2025-10-01T00:00:00.
            Default is True.

    Returns:
        Tuple[gpd.GeoDataFrame, list[str]]: First item is aggregated GeoDataFrame,
            with coordinates "NUTS_ID", "time", and
            data variables include aggregated data variables.
            The second item in the tuple is list of data variable names.
    """
    with xr.open_dataset(nc_file, chunks={"time": "auto"}) as dataset:
        # Ensure the dataset has the required coordinates
        if not all(
            coord in dataset.coords for coord in ["latitude", "longitude", "time"]
        ):
            raise ValueError(
                f"NetCDF file '{nc_file}' must contain "
                f"'latitude', 'longitude', and 'time' coordinates."
            )

        if normalize_time:
            dataset["time"] = dataset["time"].dt.floor("D")

        # get list of data variable names
        var_names = list(dataset.data_vars.keys())

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
            r_var_names = var_names
        else:
            # use provided aggregation functions
            r_var_names = list(agg_dict.keys())

        nc_data_agg = nc_data_merged.groupby(["NUTS_ID", "time"], as_index=False).agg(
            agg_dict
        )

    return nc_data_agg, r_var_names


def aggregate_data_by_nuts(
    netcdf_files: dict[str, tuple[Path, dict | None]],
    nuts_file: Path,
    normalize_time: bool = True,
    output_dir: Path | None = None,
) -> Path:
    """Aggregate data from a NetCDF file by NUTS regions, data variable names, and time.
    The aggregated data is saved to a NetCDF file with coordinates "NUTS_ID", "time",
    and data variables include aggregated data variables.

    Args:
        netcdf_files (dict[str, tuple[Path, dict | None]]): Dictionary of NetCDF files.
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

    Returns:
        Path: Path to the aggregated NetCDF file.
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
    out_data = nuts_data
    agg_var_names = []
    first_merge = True
    for ds_name, file_info in netcdf_files.items():
        file_path, agg_dict = file_info
        print(f"Processing NetCDF file: {file_path}")

        nc_data_agg, r_var_names = _aggregate_netcdf_nuts(
            nuts_data,
            file_path,
            agg_dict,
            normalize_time=normalize_time,
        )

        # merge nuts data with aggregated NetCDF data
        if first_merge:
            out_data = out_data.merge(nc_data_agg, on=["NUTS_ID"], how="outer")
            first_merge = False
        elif set(nc_data_agg.columns).issubset(set(out_data.columns)):
            # if the next NetCDF file has the same data variable names,
            # concat the data and drop duplicates
            out_data = gpd.GeoDataFrame(
                pd.concat([out_data, nc_data_agg])
                .drop_duplicates(subset=["NUTS_ID", "time"], keep="last")
                .sort_values("NUTS_ID", ignore_index=True),
                crs=out_data.crs,
            )
        else:
            out_data = out_data.merge(nc_data_agg, on=["NUTS_ID", "time"], how="outer")

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
    print(f"Aggregated data saved to: {output_file}")

    return output_file
