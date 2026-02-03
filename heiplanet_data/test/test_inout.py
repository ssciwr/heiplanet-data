from heiplanet_data import inout
import pytest
import xarray as xr
import numpy as np
from datetime import datetime


def test_download_data_by_request_invalid():
    # empty output file path
    with pytest.raises(ValueError):
        inout.download_data_by_request(None, "test_dataset", {"param": "value"})

    # empty dataset name
    with pytest.raises(ValueError):
        inout.download_data_by_request("test_output.nc", "", {"param": "value"})

    # invalid dataset name
    with pytest.raises(ValueError):
        inout.download_data_by_request("test_output.nc", 123, {"param": "value"})

    # empty request information
    with pytest.raises(ValueError):
        inout.download_data_by_request("test_output.nc", "test_dataset", None)

    # invalid request information
    with pytest.raises(ValueError):
        inout.download_data_by_request(
            "test_output.nc", "test_dataset", "invalid_request"
        )


def test_download_data_by_request_valid(tmp_path):
    output_file = tmp_path / "test" / "test_output.nc"
    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2025"],
        "month": ["03"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [0, -1, 0, 1],  # [N, W, S, E]
    }
    inout.download_data_by_request(output_file, dataset, request)
    assert output_file.exists()
    assert output_file.parent.exists()
    # Clean up
    output_file.unlink()


@pytest.fixture()
def get_data():
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 3)) * 1000 + 273.15
    data_array = xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": [0, 1], "longitude": [0, 1, 2]},
    )
    return data_array


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_save_to_netcdf(get_data, tmp_path):
    with pytest.raises(ValueError):
        inout.save_to_netcdf(get_data, None, encoding=None)

    file_name = tmp_path / "test_output_celsius.nc"
    inout.save_to_netcdf(get_data, file_name)
    assert file_name.exists()
    # Clean up
    file_name.unlink()


def test_format_ymdt():
    numbers = None
    assert inout._format_ymdt(numbers) == ""

    numbers = []
    assert inout._format_ymdt(numbers) == ""

    numbers = [str(i).zfill(2) for i in range(1, 7)]
    assert inout._format_ymdt(numbers) == "01-06"

    numbers = [str(i).zfill(2) for i in range(1, 13, 2)]
    assert inout._format_ymdt(numbers) == "01_03_05_07_09_etc"

    numbers = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_ymdt(numbers) == "01_03"


def test_format_months():
    months = None
    assert inout._format_months(months) == ""

    months = []
    assert inout._format_months(months) == ""

    months = [str(i).zfill(2) for i in range(1, 13)]
    assert inout._format_months(months) == "allm"

    months = [str(i).zfill(2) for i in range(1, 7)]
    assert inout._format_months(months) == "01-06"

    months = [str(i).zfill(2) for i in range(1, 13, 2)]
    assert inout._format_months(months) == "01_03_05_07_09_etc"

    months = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_months(months) == "01_03"


def test_format_days():
    days = None
    assert inout._format_days(days) == ""

    days = []
    assert inout._format_days(days) == ""

    days = [str(i).zfill(2) for i in range(1, 32)]
    assert inout._format_days(days) == "alld"

    days = [str(i).zfill(2) for i in range(1, 16)]
    assert inout._format_days(days) == "01-15"

    days = [str(i).zfill(2) for i in range(1, 32, 2)]
    assert inout._format_days(days) == "01_03_05_07_09_etc"

    days = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_days(days) == "01_03"


def test_format_times():
    times = None
    assert inout._format_times(times) == ""

    times = []
    assert inout._format_times(times) == ""

    times = ["00:00"]
    assert inout._format_times(times) == "midnight"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 24)]
    assert inout._format_times(times) == "allt"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 12)]
    assert inout._format_times(times) == "00-11"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 24, 2)]
    assert inout._format_times(times) == "00_02_04_06_08_etc"

    times = ["00:00", "02:00", "04:00"]
    assert inout._format_times(times) == "00_02_04"


def test_format_variables():
    variables = []
    assert inout._format_variables(variables) == ""

    variables = ["2m_temperature"]
    assert inout._format_variables(variables) == "2t"

    variables = ["2m_temperature", "total_precipitation"]
    assert inout._format_variables(variables) == "2t_tp"

    variables = [
        "2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    ]
    assert inout._format_variables(variables) == "2t_etc"


def test_format_ds_type():
    ds_name = "reanalysis-era5-land-monthly-means"
    time_str = "_midnight"
    assert inout._format_ds_type(ds_name, time_str) == "monthly"

    ds_name = "reanalysis-era5-land"
    time_str = "_midnight"
    assert inout._format_ds_type(ds_name, time_str) == "daily"

    ds_name = "reanalysis-era5-land"
    time_str = "_00-11"
    assert inout._format_ds_type(ds_name, time_str) == ""


def test_file_extension():
    data_format = "netcdf"
    assert inout._file_extension(data_format) == "nc"

    data_format = "grib"
    assert inout._file_extension(data_format) == "grib"

    data_format = "unknown"
    with pytest.raises(ValueError):
        inout._file_extension(data_format)


def test_area_format():
    has_area = True
    assert inout._area_format(has_area) == "area"

    has_area = False
    assert inout._area_format(has_area) == ""


def test_add_prefix_if_not_empty():
    prefix = "_"
    string = "test"
    assert inout._add_prefix_if_not_empty(string, prefix) == "_test"

    prefix = "_"
    string = ""
    assert inout._add_prefix_if_not_empty(string, prefix) == ""


def test_truncate_string():
    string = "a" * 50
    max_length = 2
    assert inout._truncate_string(string, max_length) == "aa_etc"

    string = "short_string"
    max_length = 30
    assert inout._truncate_string(string, max_length) == "short_string"

    string = "a" * 100
    max_length = 100
    assert inout._truncate_string(string, max_length) == "a" * 100


def test_get_filename_var():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_monthly_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_allm_2t_monthly_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_2t_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        False,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "grib",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_area_raw.grib"


def test_get_filename_vars():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature", "total_precipitation"],
    )
    assert file_name == "era5_data_2025_01-02_2t_tp_monthly_area_raw.nc"


def test_get_filename_long():
    # long vars
    var_names = [
        "2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    ]

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2025_01-02_2t_etc_monthly_area_raw.nc"

    # long years and long vars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_1900-2029_01-02_2t_etc_monthly_area_raw.nc"

    # non-continuous years and long vars
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2020", "2023", "2021"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2020_2021_2023_01-02_2t_etc_monthly_area_raw.nc"

    # non-continuous years with more than 5 years
    years = [str(i) for i in range(2020, 2040, 2)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert (
        file_name
        == "era5_data_2020_2022_2024_2026_2028_etc_01-02_2t_etc_monthly_area_raw.nc"
    )

    # more than 100 chars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars",
        var_names,
    )
    assert (
        file_name
        == "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars_"
        "1900-2029_01-02_2t_etc_mon_etc_raw.nc"
    )


def test_get_filename_none_cases():
    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land-monthly-means",
        data_format="netcdf",
        years=None,
        months=None,
        days=None,
        times=None,
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2t_monthly_raw.nc"


def test_get_filename_days_times_dstype():
    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 32)],
        times=[f"{str(i).zfill(2)}:00" for i in range(0, 24)],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_alld_allt_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 11)],
        times=[f"{str(i).zfill(2)}:00" for i in range(0, 11)],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_01-10_00-10_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=["01", "05"],
        times=["00:00", "02:00"],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_01_05_00_02_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=["01", "05", "10", "15", "20", "25"],
        times=["00:00", "02:00", "04:00", "06:00", "08:00", "10:00"],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert (
        file_name == "era5_data_2025_01_01_05_10_15_20_etc_00_02_04_06_08_etc_2t_raw.nc"
    )

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 32)],
        times=["00:00"],
        has_area=True,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_alld_midnight_2t_daily_area_raw.nc"


def test_download_sub_tp_data(tmp_path):
    out_dir = tmp_path / "test_sub_tp"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2025-03-30"
    end_date = "2025-03-31"
    tmp_file_path = inout._download_sub_tp_data(
        date_range=(
            datetime.strptime(start_date, "%Y-%m-%d"),
            datetime.strptime(end_date, "%Y-%m-%d"),
        ),
        range_idx=0,
        area=[0, -1, 0, 1],
        out_dir=out_dir,
        file_name="era5_data",
        file_ext="nc",
        ds_name="reanalysis-era5-land",
        var_name="total_precipitation",
        coord_name="valid_time",
        data_format="netcdf",
    )
    assert tmp_file_path.exists()

    with xr.open_dataset(tmp_file_path) as tmp_ds:
        assert tmp_ds["valid_time"].values[0] == np.datetime64("2025-03-30")
        assert tmp_ds["valid_time"].values[1] == np.datetime64("2025-03-31")

    # Clean up
    tmp_file_path.unlink()


def test_download_sub_tp_data_existing_file(tmp_path):
    out_dir = tmp_path / "test_sub_tp_existing"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2025-03-30"
    end_date = "2025-03-31"

    expected_file = out_dir / "era5_data_tmp_2025-03-30-2025-03-31.nc"
    expected_file.touch()

    tmp_file_path = inout._download_sub_tp_data(
        date_range=(
            datetime.strptime(start_date, "%Y-%m-%d"),
            datetime.strptime(end_date, "%Y-%m-%d"),
        ),
        range_idx=0,
        area=[0, -1, 0, 1],
        out_dir=out_dir,
        file_name="era5_data",
        file_ext="nc",
        ds_name="reanalysis-era5-land",
        var_name="total_precipitation",
        coord_name="valid_time",
        data_format="netcdf",
    )
    assert tmp_file_path == expected_file
    assert tmp_file_path.stat().st_size == 0  # file is empty

    # Clean up
    tmp_file_path.unlink()


def test_download_total_precipitation_from_hourly_era5_land_invalid_dates(tmp_path):
    out_dir = tmp_path / "test_download_tp_invalid"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        inout.download_total_precipitation_from_hourly_era5_land(
            start_date="2025",
            end_date=1.0,
            area=[0, -1, 0, 1],
            out_dir=out_dir,
        )

    with pytest.raises(ValueError):
        inout.download_total_precipitation_from_hourly_era5_land(
            start_date="2025-01-01",
            end_date="2024-12-31",
            area=[0, -1, 0, 1],
            out_dir=out_dir,
        )


def test_download_total_precipitation_from_hourly_era5_land_same_year_month(
    tmp_path,
):
    out_dir = tmp_path / "test_download_tp"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2025-03-15"
    end_date = "2025-03-17"
    fname = inout.download_total_precipitation_from_hourly_era5_land(
        start_date=start_date,
        end_date=end_date,
        area=None,
        out_dir=out_dir,
        base_name="era5_data",
        data_format="netcdf",
        ds_name="reanalysis-era5-land",
        coord_name="valid_time",
        var_name="total_precipitation",
        clean_tmp_files=False,  # keep temporary files for testing
    )
    output_file_name = "era5_data_2025-03-15-2025-03-17_midnight_tp_daily_raw.nc"
    output_file_path = out_dir / output_file_name
    assert output_file_path.exists()
    assert fname == str(output_file_path)

    # check if temporary files are kept
    tmp_file_path = out_dir / output_file_name.replace(
        ".nc", "_tmp_2025-03-16-2025-03-18.nc"
    )
    assert tmp_file_path.exists()

    # manually download data for checking
    dataset = "reanalysis-era5-land"
    request = {
        "variable": ["total_precipitation"],
        "year": "2025",
        "month": "03",
        "day": ["16", "17", "18"],  # move 1 day forward
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp_file = out_dir / "temp_download.nc"
    inout.download_data_by_request(tmp_file, dataset, request)

    # compare data
    with xr.open_dataset(tmp_file) as tmp_ds:
        with xr.open_dataset(output_file_path) as out_ds:
            np.testing.assert_allclose(
                tmp_ds["tp"].values,
                out_ds["tp"].values,
                rtol=1e-5,
                atol=1e-5,
            )

    # clean up
    output_file_path.unlink()
    tmp_file.unlink()
    tmp_file_path.unlink()


def test_download_total_precipitation_from_hourly_era5_land_diff_year(
    tmp_path,
):
    out_dir = tmp_path / "test_download_tp_diff_year"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2024-12-30"
    end_date = "2025-01-02"
    _ = inout.download_total_precipitation_from_hourly_era5_land(
        start_date=start_date,
        end_date=end_date,
        area=[0, -1, 0, 1],
        out_dir=out_dir,
        base_name="era5_data",
        data_format="netcdf",
        ds_name="reanalysis-era5-land",
        coord_name="valid_time",
        var_name="total_precipitation",
        clean_tmp_files=True,  # remove temporary files after merging
    )
    output_file_name = "era5_data_2024-12-30-2025-01-02_midnight_tp_daily_area_raw.nc"
    output_file_path = out_dir / output_file_name
    assert output_file_path.exists()

    # check if temporary files are removed
    tmp_file_path_1 = out_dir / output_file_name.replace(
        ".nc", "_tmp_2024-12-31-2024-12-31.nc"
    )
    assert not tmp_file_path_1.exists()

    # check if dates are correct in the downloaed dataset
    with xr.open_dataset(output_file_path) as out_ds:
        times = out_ds["valid_time"].values
        dates = np.array(
            [
                np.datetime64("2024-12-30"),
                np.datetime64("2024-12-31"),
                np.datetime64("2025-01-01"),
                np.datetime64("2025-01-02"),
            ]
        )
        np.testing.assert_array_equal(times, dates)


def test_download_total_precipitation_from_hourly_era5_land_truncate(
    tmp_path,
):
    out_dir = tmp_path / "test_download_tp_truncate"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2024-10-15"
    end_date = "2024-11-15"
    fname = inout.download_total_precipitation_from_hourly_era5_land(
        start_date=start_date,
        end_date=end_date,
        area=[0, -1, 0, 1],
        out_dir=out_dir,
        base_name="era5_data",
        data_format="netcdf",
        ds_name="reanalysis-era5-land",
        coord_name="valid_time",
        var_name="total_precipitation",
        clean_tmp_files=True,
    )

    output_file_name = "era5_data_2024-10-15-2024-11-15_midnight_tp_daily_area_raw.nc"
    output_file_path = out_dir / output_file_name
    assert fname == str(output_file_path)
    assert output_file_path.exists()

    # check if dates are correct in the downloaed dataset
    with xr.open_dataset(output_file_path) as out_ds:
        times = out_ds["valid_time"].values
        dates = np.array(
            [np.datetime64(f"2024-10-{i:02d}") for i in range(15, 32)]
            + [np.datetime64(f"2024-11-{i:02d}") for i in range(1, 16)]
        )
        np.testing.assert_array_equal(times, dates)


def test_download_total_precipitation_from_hourly_era5_land_existing_file(tmp_path):
    existing_file = (
        tmp_path / "era5_data_2025-03-18-2025-03-19_midnight_tp_daily_raw.nc"
    )
    existing_file.touch()  # create an empty file

    fname = inout.download_total_precipitation_from_hourly_era5_land(
        start_date="2025-03-18",
        end_date="2025-03-19",
        area=None,
        out_dir=tmp_path,
        base_name="era5_data",
        data_format="netcdf",
        ds_name="reanalysis-era5-land",
        coord_name="valid_time",
    )

    assert fname == str(existing_file)

    # check if the existing file is empty
    assert existing_file.stat().st_size == 0
