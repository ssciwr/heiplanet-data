"""Unit and coordinate conversions for climate datasets.

This module holds the elementary, dependency-light transformations applied to
raw datasets:

* longitude conversion from the 0-360 range to the -180-180 range
  (`convert_360_to_180`, `adjust_longitude_360_to_180`),
* temperature conversion from Kelvin to Celsius
  (`convert_to_celsius`, `convert_to_celsius_with_attributes`),
* precipitation conversion from meters to millimeters
  (`convert_m_to_mm`, `convert_m_to_mm_with_attributes`),
* renaming of coordinates to unified names across data sources
  (`rename_coords`).

The ``*_with_attributes`` variants operate on a whole dataset and keep the
variable attributes (e.g. GRIB metadata) intact.
"""

from typing import TypeVar, Union
import xarray as xr
import numpy as np
import warnings


T = TypeVar("T", bound=Union[np.float64, xr.DataArray])


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
