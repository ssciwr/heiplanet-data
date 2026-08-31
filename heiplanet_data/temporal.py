"""Time-axis operations: shifting, truncation, and monthly aggregation.

This module manipulates the time coordinate of a dataset:

* shifting all time points by a fixed offset (`shift_time`),
* truncating a dataset to a date range (`truncate_data_by_time`),
* converting ERA5-Land monthly mean precipitation to monthly totals by
  multiplying with the number of days per month
  (`calculate_monthly_precipitation`).
"""

import re
from typing import Literal

import numpy as np
import xarray as xr


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
        raise TypeError("Offset value must be an int.")

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
        raise TypeError("Date must be of type string, np.datetime64, or None.")

    return date


def truncate_data_by_time(
    dataset: xr.Dataset,
    start_date: str | np.datetime64,
    end_date: str | np.datetime64 | None = None,
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
    return bool(np.all(days == 1))


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
