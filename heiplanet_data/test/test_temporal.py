import numpy as np
import pytest
import xarray as xr
from heiplanet_data import temporal


def test_shift_time_invalid(get_dataset):
    with pytest.raises(ValueError):
        temporal.shift_time(
            get_dataset, offset="invalid", time_unit="D", var_name="time"
        )
    with pytest.raises(ValueError):
        temporal.shift_time(get_dataset, offset=2.5, time_unit="D", var_name="invalid")
    with pytest.raises(ValueError):
        temporal.shift_time(get_dataset, offset=2, time_unit="Y", var_name="time")
    with pytest.raises(ValueError):
        temporal.shift_time(get_dataset, offset=2, time_unit="M", var_name="time")


def test_shift_time_forward(get_dataset):
    original_time = get_dataset["time"].copy()
    # shift time by 2 days
    offset = 2
    time_unit = "D"
    time_shift = np.timedelta64(2, "D")
    temporal.shift_time(
        get_dataset, offset=offset, time_unit=time_unit, var_name="time"
    )

    # check if the time dimension is preserved
    assert len(get_dataset["time"]) == 2

    # check if the time is shifted correctly
    expected_time = original_time + time_shift.astype("timedelta64[ns]")
    assert np.array_equal(
        np.sort(get_dataset["time"].values), np.sort(expected_time.values)
    )

    # check if time is at midnight after shifting
    assert all(get_dataset["time"].dt.hour.values == 0)


def test_shift_time_backward(get_dataset):
    original_time = get_dataset["time"].copy()

    # shift time by -2 hours
    offset = -2
    time_unit = "h"
    time_shift = np.timedelta64(offset, time_unit)
    temporal.shift_time(
        get_dataset, offset=offset, time_unit=time_unit, var_name="time"
    )
    expected_time = original_time + time_shift.astype("timedelta64[ns]")
    assert np.array_equal(
        np.sort(get_dataset["time"].values), np.sort(expected_time.values)
    )
    assert all(get_dataset["time"].dt.hour.values == 22)


def test_parse_date_invalid():
    with pytest.raises(ValueError):
        temporal._parse_date(date="invalid_date")

    with pytest.raises(ValueError):
        temporal._parse_date(date="2024-13-01")

    with pytest.raises(ValueError):
        temporal._parse_date(date=12345)


def test_parse_date():
    date_str = "2024-07-15"
    parsed_date = temporal._parse_date(date_str)
    expected_date = np.datetime64("2024-07-15")
    assert parsed_date == expected_date

    date_np = np.datetime64("2025-12-31")
    parsed_date = temporal._parse_date(date_np)
    assert parsed_date == date_np


def test_truncate_data_by_time_invalid(get_dataset):
    with pytest.raises(ValueError):
        temporal.truncate_data_by_time(
            get_dataset, start_date=None, end_date=None, var_name="time"
        )
    with pytest.raises(ValueError):
        temporal.truncate_data_by_time(
            get_dataset, start_date="2025-01-01", end_date="2024-01-01", var_name="time"
        )
    with pytest.raises(ValueError):
        temporal.truncate_data_by_time(
            get_dataset, start_date="2025-01-01", end_date=None, var_name="invalid_var"
        )


def test_truncate_data_by_time(get_dataset):
    # truncate data by time
    truncated_dataset = temporal.truncate_data_by_time(
        get_dataset, start_date="2025-01-01", end_date="2025-01-01", var_name="time"
    )

    # check if the time dimension is reduced
    assert len(truncated_dataset["t2m"].time) == 1
    assert len(truncated_dataset["tp"].time) == 1

    # check if the data is truncated correctly
    assert np.allclose(
        truncated_dataset["t2m"].values, get_dataset["t2m"].isel(time=1).values
    )
    assert np.allclose(
        truncated_dataset["tp"].values, get_dataset["tp"].isel(time=1).values
    )

    # start date as np.datetime64
    truncated_dataset = temporal.truncate_data_by_time(
        get_dataset,
        start_date=np.datetime64("2025-01-01"),
        end_date=np.datetime64("2025-01-01"),
        var_name="time",
    )

    assert np.allclose(
        truncated_dataset["t2m"].values, get_dataset["t2m"].isel(time=1).values
    )
    assert np.allclose(
        truncated_dataset["tp"].values, get_dataset["tp"].isel(time=1).values
    )

    # random start date
    truncated_dataset = temporal.truncate_data_by_time(
        get_dataset,
        start_date=np.datetime64("2024-07-17"),
        end_date=np.datetime64("2025-01-01"),
        var_name="time",
    )
    assert len(truncated_dataset["t2m"].time) == 1
    assert truncated_dataset["t2m"].time.values[0] == np.datetime64("2025-01-01")

    # None end date
    truncated_dataset = temporal.truncate_data_by_time(
        get_dataset,
        start_date=np.datetime64("2025-01-01"),
        end_date=None,
        var_name="time",
    )
    assert len(truncated_dataset["t2m"].time) == 1
    assert truncated_dataset["t2m"].time.values[0] == np.datetime64("2025-01-01")


def test_check_month_start_data():
    months = ["2016-01-01", "2016-03-01"]
    data = xr.DataArray(
        data=np.array(months, dtype="datetime64[ns]"),
        dims=["time"],
    )
    assert temporal._check_month_start_data(data) is True

    # invalid case
    months = ["2016-01-15", "2016-03-01"]
    data = xr.DataArray(
        data=np.array(months, dtype="datetime64[ns]"),
        dims=["time"],
    )
    assert temporal._check_month_start_data(data) is False


def test_calculate_monthly_precipitation_invalid(get_dataset):
    with pytest.raises(ValueError):
        temporal.calculate_monthly_precipitation(
            get_dataset, var_name="error", time_coord="time"
        )
    with pytest.raises(ValueError):
        temporal.calculate_monthly_precipitation(
            get_dataset, var_name="tp", time_coord="error"
        )
    # modify time to non-monthly start dates
    get_dataset_invalid = get_dataset.copy()
    get_dataset_invalid = get_dataset_invalid.assign_coords(
        time=("time", [np.datetime64("2024-01-15"), np.datetime64("2025-02-15")])
    )
    with pytest.raises(ValueError):
        temporal.calculate_monthly_precipitation(
            get_dataset_invalid, var_name="tp", time_coord="time"
        )


def test_calculate_monthly_precipitation(get_dataset):
    org_ds = get_dataset.copy()
    # change time to get different days in month
    get_dataset = get_dataset.assign_coords(
        time=("time", [np.datetime64("2024-01-01"), np.datetime64("2024-02-01")])
    )
    # calculate monthly precipitation
    monthly_dataset = temporal.calculate_monthly_precipitation(
        get_dataset, var_name="tp", time_coord="time"
    )

    assert len(monthly_dataset["tp"].time) == 2
    assert monthly_dataset["tp"].time.values[0] == np.datetime64("2024-01-01")
    assert monthly_dataset["tp"].time.values[1] == np.datetime64("2024-02-01")

    expected_tp = org_ds["tp"].values * np.array([31, 29])[:, None, None]
    assert np.allclose(
        monthly_dataset["tp"].values,
        expected_tp,
    )
