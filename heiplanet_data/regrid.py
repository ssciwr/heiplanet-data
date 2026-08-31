"""Spatial resolution resampling (downsampling and upsampling) of datasets.

This module changes the grid resolution of a dataset:

* downsampling (coarser grid) with one of three backends:
  `downsample_resolution_with_xarray` (coarsen/reduce),
  `downsample_resolution_with_xesmf` (regridding via xESMF), and
  `downsample_resolution_with_cdo` (remapping via the CDO binary),
* upsampling (finer grid) via interpolation (`upsample_resolution`),
* the dispatching entry point `resample_resolution`, configured through the
  `ResolutionConfig` and `GridConfig` dataclasses, which picks the backend
  and direction based on the requested resolution.

This is the only module that imports the heavy optional dependencies
``xesmf``/``esmpy`` and ``cdo``; both require the conda environment to be
activated so that the underlying binaries are on ``PATH``.
"""

import tempfile
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import xesmf as xe
from cdo import Cdo

warn_positive_resolution = "New resolution must be a positive number."

# module-level singleton so it isn't constructed at every def/dataclass evaluation
_DEFAULT_EXPECTED_LONGITUDE_MAX = np.float64(179.75)


def check_downsample_condition(
    dataset: xr.Dataset,
    new_resolution: float,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: dict[str, str] | None = None,
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
            f"should be greater than {old_resolution}."
        )

    if agg_funcs is not None and not isinstance(agg_funcs, dict):
        raise ValueError(
            "agg_funcs must be a dictionary of variable names and aggregation functions."
        )

    return old_resolution


def check_agg_funcs(agg_funcs: dict[str, str], valid_agg_funcs: set) -> None:
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
    agg_funcs: dict[str, str] | None = None,
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
    expected_longitude_max: np.float64 = _DEFAULT_EXPECTED_LONGITUDE_MAX,
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
    agg_funcs: dict[str, str] | None = None,
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

    for var in agg_funcs:
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
    agg_funcs: dict[str, str] | None = None,
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
        with tempfile.NamedTemporaryFile(suffix=f"_{var}.nc", delete=False) as tmp_file:
            tmp_file_name = tmp_file.name
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
    with tempfile.NamedTemporaryFile(
        suffix="_gridspec.txt", delete=False
    ) as gridspec_file:
        gridspec_file_name = gridspec_file.name
        gridspec_file.write(gridspec.encode())

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
    method_map: dict[str, str] | None = None,
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
            f"should be smaller than {old_resolution}."
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
    downsample_agg_funcs: dict[str, str] | None = None
    upsample_method_map: dict[str, str] | None = None


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

    expected_longitude_max_xarray: np.float64 = _DEFAULT_EXPECTED_LONGITUDE_MAX
    new_min_lat: float | None = None
    new_max_lat: float | None = None
    new_min_lon: float | None = None
    new_max_lon: float | None = None
    new_lat_size: int | None = None
    new_lon_size: int | None = None
    gridtype: Literal["gaussian", "lonlat", "curvilinear", "unstructured"] = "lonlat"


def resample_resolution(
    dataset: xr.Dataset,
    resolution_config: ResolutionConfig | None = None,
    grid_config: GridConfig | None = None,
) -> xr.Dataset:
    """Resample the grid of a dataset to a new resolution.

    Args:
        dataset (xr.Dataset): Dataset to resample.
        resolution_config (ResolutionConfig): Configuration for resolution resampling.
            Default is None, which uses the default `ResolutionConfig`.
        grid_config (GridConfig): Configuration for grid specification.
            Default is None, which uses the default `GridConfig`.

    Returns:
        xr.Dataset: Resampled dataset with changed resolution.
    """
    if resolution_config is None:
        resolution_config = ResolutionConfig()
    if grid_config is None:
        grid_config = GridConfig()

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
