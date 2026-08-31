"""Aggregation of gridded NetCDF data over NUTS regions.

This module reduces preprocessed gridded data to one value per NUTS region
and time step:

* `aggregate_data_by_nuts` is the public entry point: it reads one or more
  NetCDF files, aggregates each data variable over the NUTS polygons from a
  shape file, merges the results, and writes them to a NetCDF file indexed
  by ``NUTS_ID`` and ``time``,
* two aggregation backends are available: ``exactextract`` (default,
  area-weighted, suited for large datasets) and ``geopandas`` (point-in-
  polygon spatial join, memory-hungry on fine global grids).

This is the only module that imports ``exactextract``.
"""

import logging
import warnings
from functools import reduce
from pathlib import Path
from typing import Literal

import exactextract as ee
import geopandas as gpd
import pandas as pd
import xarray as xr

from heiplanet_data import utils

logger = logging.getLogger(__name__)

CRS = 4326  # EPSG code for WGS 84


def _prepare_for_aggregation(
    dataset: xr.Dataset,
    normalize_time: bool = True,
    agg_dict: dict[str, str] | None = None,
) -> tuple[xr.Dataset, dict[str, str]]:
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
        or not all(var in var_names for var in agg_dict)
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
    agg_dict: dict[str, str] | None,
    normalize_time: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
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
    agg_dict: dict[str, str] | None,
    normalize_time: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
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
    netcdf_files: dict[str, tuple[Path, dict[str, str] | None]],
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
    netcdf_files: dict[str, tuple[Path, dict | None]],
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
