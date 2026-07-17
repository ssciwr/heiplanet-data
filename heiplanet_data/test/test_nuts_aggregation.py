from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from heiplanet_data import nuts_aggregation


def test_prepare_for_aggregation_normalize(get_dataset):
    # change time to mid-day
    get_dataset["time"] = get_dataset["time"] + np.timedelta64(12, "h")

    # prepare data without time normalization
    ds, _ = nuts_aggregation._prepare_for_aggregation(
        get_dataset, normalize_time=False, agg_dict=None
    )
    assert np.unique(ds.time.dt.hour).tolist() == [12]

    # prepare data with time normalization
    ds, _ = nuts_aggregation._prepare_for_aggregation(
        get_dataset, normalize_time=True, agg_dict=None
    )
    assert np.unique(ds.time.dt.hour).tolist() == [0]


def test_prepare_for_aggregation_agg_dict(get_dataset):
    # None case
    _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
        get_dataset, normalize_time=False, agg_dict=None
    )
    expected_agg_dict = {
        "t2m": "mean",
        "tp": "mean",
    }
    assert p_agg_dict == expected_agg_dict

    # custom aggregation dictionary
    o_agg_dict = {
        "t2m": "mean",
        "tp": "sum",
    }
    _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
        get_dataset, normalize_time=False, agg_dict=o_agg_dict
    )
    assert p_agg_dict == o_agg_dict

    # invalid cases
    with pytest.warns(UserWarning):
        _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={"t2m": 1}
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict="something"
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={}
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = nuts_aggregation._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={"invalid_key": "mean"}
        )
    assert p_agg_dict == expected_agg_dict


def test_aggregate_netcdf_nuts_gpd_invalid(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # change coordinates to invalid names
    get_dataset = get_dataset.rename({"latitude": "lat", "longitude": "lon"})
    get_dataset.to_netcdf(file_path)

    with pytest.raises(ValueError):
        nuts_aggregation._aggregate_netcdf_nuts_gpd(
            get_nuts_data, file_path, agg_dict=None, normalize_time=False
        )


def test_aggregate_netcdf_nuts_gpd_normalize_none_aggdict(
    tmp_path, get_dataset, get_nuts_data
):
    file_path = tmp_path / "test_data.nc"
    # change time to mid-day
    get_dataset["time"] = get_dataset["time"] + np.timedelta64(12, "h")
    get_dataset.to_netcdf(file_path)

    # aggregate data without time normalization
    out_data, var_names = nuts_aggregation._aggregate_netcdf_nuts_gpd(
        get_nuts_data, file_path, agg_dict=None, normalize_time=True
    )

    assert "NUTS_ID" in out_data.columns
    assert "time" in out_data.columns
    assert "t2m" in out_data.columns
    assert "tp" in out_data.columns
    assert "latitude" not in out_data.columns
    assert var_names == ["t2m", "tp"]
    assert len(out_data) == 4  # two NUTS regions with two time points each
    assert out_data["time"].dt.hour.unique().tolist() == [
        0
    ]  # check if time is midnight
    assert np.isclose(
        out_data.iloc[0]["t2m"], get_dataset["t2m"].values[0, :, 0].mean()
    )
    assert np.isclose(out_data.iloc[0]["tp"], get_dataset["tp"].values[0, :, 0].mean())
    assert np.isclose(
        out_data.iloc[2]["t2m"], get_dataset["t2m"].values[0, :, 1:].mean()
    )


def test_aggregate_netcdf_nuts_gpd_custom_agg_dict(
    tmp_path, get_dataset, get_nuts_data
):
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    # aggregate data with custom aggregation dictionary
    agg_dict = {
        "t2m": "mean",
        "tp": "sum",
    }
    out_data, _ = nuts_aggregation._aggregate_netcdf_nuts_gpd(
        get_nuts_data, file_path, agg_dict=agg_dict, normalize_time=False
    )

    assert "NUTS_ID" in out_data.columns
    assert np.isclose(
        out_data.iloc[0]["t2m"], get_dataset["t2m"].values[0, :, 0].mean()
    )
    assert np.isclose(out_data.iloc[0]["tp"], get_dataset["tp"].values[0, :, 0].sum())


def test_aggregate_netcdf_nuts_gpd_too_large_ds(tmp_path, get_nuts_data):
    file_path = tmp_path / "large_test_data.nc"
    # create a large dataset
    # with 12 monthly data for 1 year, global 0.1 degree grid
    time = pd.date_range("2025-01-01", periods=12, freq="ME")
    lat = np.arange(-90.0, 90.1, 0.1)
    lon = np.arange(-180.0, 180.1, 0.1)
    rng = np.random.default_rng(seed=42)
    data = xr.DataArray(
        rng.random(
            (len(time), len(lat), len(lon))
        ),  # rng.random takes shape as a tuple
        coords=[time, lat, lon],
        dims=["time", "latitude", "longitude"],
    )
    large_dataset = xr.Dataset({"large_var": data})
    large_dataset.to_netcdf(file_path)

    with pytest.raises(ValueError):
        nuts_aggregation._aggregate_netcdf_nuts_gpd(
            get_nuts_data, file_path, agg_dict=None, normalize_time=False
        )


def test_aggregate_netcdf_nuts_ee_invalid(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # change coordinates to invalid names
    get_dataset = get_dataset.rename({"latitude": "lat", "longitude": "lon"})
    get_dataset.to_netcdf(file_path)

    with pytest.raises(ValueError):
        nuts_aggregation._aggregate_netcdf_nuts_ee(
            get_nuts_data, file_path, agg_dict=None, normalize_time=False
        )


def test_aggregate_netcdf_nuts_ee_normalize_none_aggdict(
    tmp_path, get_dataset, get_nuts_data
):
    file_path = tmp_path / "test_data.nc"
    # change time to mid-day
    get_dataset["time"] = get_dataset["time"] + np.timedelta64(12, "h")
    get_dataset.to_netcdf(file_path)

    # aggregate data without time normalization
    out_data, var_names = nuts_aggregation._aggregate_netcdf_nuts_ee(
        get_nuts_data, file_path, agg_dict=None, normalize_time=True
    )

    assert "NUTS_ID" in out_data.columns
    assert "time" in out_data.columns
    assert "t2m" in out_data.columns
    assert "tp" in out_data.columns
    assert "latitude" not in out_data.columns
    assert var_names == ["t2m", "tp"]
    assert len(out_data) == 4  # two NUTS regions with two time points each
    assert out_data["time"].dt.hour.unique().tolist() == [
        0
    ]  # check if time is midnight

    # sort by NUTS_ID and time
    # since the order is different from geopandas aggregation
    out_data = out_data.sort_values(by=["NUTS_ID", "time"]).reset_index(drop=True)

    assert np.isclose(
        out_data.iloc[0]["t2m"], get_dataset["t2m"].values[0, :, 0].mean()
    )
    assert np.isclose(out_data.iloc[0]["tp"], get_dataset["tp"].values[0, :, 0].mean())
    assert np.isclose(
        out_data.iloc[2]["t2m"], get_dataset["t2m"].values[0, :, 1:].mean()
    )


def test_aggregate_netcdf_nuts_ee_custom_agg_dict(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    # aggregate data with custom aggregation dictionary
    agg_dict = {
        "t2m": "mean",
        "tp": "sum",
    }
    out_data, _ = nuts_aggregation._aggregate_netcdf_nuts_ee(
        get_nuts_data, file_path, agg_dict=agg_dict, normalize_time=False
    )

    assert "NUTS_ID" in out_data.columns
    assert np.isclose(
        out_data.iloc[0]["t2m"], get_dataset["t2m"].values[0, :, 0].mean()
    )
    assert np.isclose(out_data.iloc[0]["tp"], get_dataset["tp"].values[0, :, 0].sum())


def test_aggregate_netcdf_nuts_ee_minus_in_name(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # change data variable names to have '-' in them
    get_dataset = get_dataset.rename({"t2m": "t-2m", "tp": "t-p"})
    get_dataset.to_netcdf(file_path)

    # aggregate data
    out_data, variable_names = nuts_aggregation._aggregate_netcdf_nuts_ee(
        get_nuts_data, file_path, agg_dict=None, normalize_time=False
    )

    assert "NUTS_ID" in out_data.columns
    assert "t-2m" in out_data.columns
    assert "t-p" in out_data.columns
    assert variable_names == ["t-2m", "t-p"]

    # sort by NUTS_ID and time
    # since the order is different from geopandas aggregation
    out_data = out_data.sort_values(by=["NUTS_ID", "time"]).reset_index(drop=True)

    assert np.isclose(
        out_data.iloc[0]["t-p"], get_dataset["t-p"].values[0, :, 0].mean()
    )


def test_aggregate_netcdf_nuts_ee_3_data_vars(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # add a third data variable
    get_dataset["humidity"] = get_dataset["t2m"] * 0.5
    get_dataset.to_netcdf(file_path)

    # aggregate data
    out_data, variable_names = nuts_aggregation._aggregate_netcdf_nuts_ee(
        get_nuts_data, file_path, agg_dict=None, normalize_time=False
    )

    assert "NUTS_ID" in out_data.columns
    assert "t2m" in out_data.columns
    assert "tp" in out_data.columns
    assert "humidity" in out_data.columns
    assert set(variable_names) == {"t2m", "tp", "humidity"}

    # sort by NUTS_ID and time
    # since the order is different from geopandas aggregation
    out_data = out_data.sort_values(by=["NUTS_ID", "time"]).reset_index(drop=True)

    assert np.isclose(
        out_data.iloc[0]["humidity"], get_dataset["humidity"].values[0, :, 0].mean()
    )


def test_check_aggregation_inputs_invalid(tmp_path):
    # non dict
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts("something", tmp_path / "nuts.shp")

    # empty dict
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts({}, tmp_path / "nuts.shp")

    # dict with non-exist file
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts(
            {"era5": (Path("something"), None)}, tmp_path / "nuts.shp"
        )

    # dict with empty file
    nc_file = tmp_path / "test_data.nc"
    nc_file.touch()  # create an empty file
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts(
            {"era5": (nc_file, None)}, tmp_path / "nuts.shp"
        )

    # dict with non-nuts data
    with open(nc_file, "w") as f:
        f.write("This is a test file.")
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts(
            {"era5": (nc_file, None)}, tmp_path / "nuts.shp"
        )


def test_aggregate_data_by_nuts_invalid(tmp_path):
    nc_file = tmp_path / "test_data.nc"
    with open(nc_file, "w") as f:
        f.write("This is a test file.")

    # dict with nust data but no NUTS_ID and geometry columns
    data = {
        "nuts_name": ["name1", "name2"],
        "geometry": [None, None],
    }
    nuts_data = gpd.GeoDataFrame(data, crs="EPSG:4326")
    nuts_data.to_file(tmp_path / "nuts.shp")
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts(
            {"era5": (nc_file, None)}, tmp_path / "nuts.shp"
        )

    # invalid aggregation lib
    data = {
        "NUTS_ID": ["ID1", "ID2"],
        "nuts_name": ["name1", "name2"],
        "geometry": [None, None],
    }
    nuts_data = gpd.GeoDataFrame(data, crs="EPSG:4326")
    nuts_data.to_file(tmp_path / "nuts.shp")
    with pytest.raises(ValueError):
        nuts_aggregation.aggregate_data_by_nuts(
            {"era5": (nc_file, None)},
            tmp_path / "nuts.shp",
            agg_lib="invalid_lib",
        )


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts(tmp_path, get_dataset, get_nuts_data, tmpdir, agg_lib):
    out_dir = Path(tmpdir) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    # save nuts data to a temporary file
    get_nuts_data.to_file(tmp_path / "nuts.shp")

    # aggregate data by NUTS regions
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path, None)},
        tmp_path / "nuts.shp",
        normalize_time=True,
        output_dir=out_dir,
        agg_lib=agg_lib,
    )

    # check if the output file is created
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir
    with xr.open_dataset(out_file) as ds:
        # check if the data is aggregated correctly
        assert "NUTS_ID" in ds.coords
        assert "time" in ds.coords
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
        assert ds.sizes.get("NUTS_ID") == 2  # two NUTS regions
        assert ds.sizes.get("time") == 2  # two time points
        assert len(ds.data_vars) == 2  # only two variables

        # check if the time is normalized to midnight
        assert np.all(ds["time"].dt.hour == 0)

    # clean up the output directory
    for file in out_dir.glob("*"):
        file.unlink()
    out_dir.rmdir()  # remove the output directory after test


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts_outdir(tmp_path, get_dataset, get_nuts_data, agg_lib):
    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    # save nuts data to a temporary file
    nuts_file = tmp_path / "nuts.shp"
    get_nuts_data.to_file(nuts_file)

    # aggregate data by NUTS regions with output directory
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path, None)},
        nuts_file,
        normalize_time=True,
        output_dir=None,
        agg_lib=agg_lib,
    )

    # check if the output file is created in folder of nuts file
    out_dir = nuts_file.parent
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir

    # clean up the output directory
    for file in out_dir.glob("*"):
        file.unlink()
    out_dir.rmdir()  # remove the output directory after test


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts_diff_netcdfs(
    tmp_path, get_dataset, get_nuts_data, tmpdir, agg_lib
):
    out_dir = Path(tmpdir) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save dataset to a temporary file
    file_path1 = tmp_path / "test_data1.nc"
    file_path2 = tmp_path / "test_data2.nc"
    get_dataset.to_netcdf(file_path1)
    # modify the dataset for the second file
    # to create ds with different time values
    modified_dataset = get_dataset.copy()
    modified_dataset["time"] = modified_dataset["time"] + np.timedelta64(12, "h")
    # change variable names
    modified_dataset = modified_dataset.rename({"t2m": "t2m_mod", "tp": "tp_mod"})
    modified_dataset.to_netcdf(file_path2)

    # save nuts data to a temporary file
    get_nuts_data.to_file(tmp_path / "nuts.shp")

    # aggregate data by NUTS regions
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path1, None), "era5_mod": (file_path2, None)},  # disjoint case
        tmp_path / "nuts.shp",
        normalize_time=True,
        output_dir=out_dir,
        agg_lib=agg_lib,
    )

    # check if the output file is created
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir
    with xr.open_dataset(out_file) as ds:
        # check if the data is aggregated correctly
        assert "NUTS_ID" in ds.coords
        assert "time" in ds.coords
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
        assert "t2m_mod" in ds.data_vars
        assert "tp_mod" in ds.data_vars
        assert ds.sizes.get("NUTS_ID") == 2  # two NUTS regions
        assert ds.sizes.get("time") == 2  # two time points
        assert len(ds.data_vars) == 4  # four variables

        # check if the time is normalized to midnight
        assert np.all(ds["time"].dt.hour == 0)

    # clean up the output directory
    for file in out_dir.glob("*"):
        file.unlink()
    out_dir.rmdir()  # remove the output directory after test


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts_diff_netcdfs_diff_times(
    tmp_path, get_dataset, get_nuts_data, tmpdir, agg_lib
):
    out_dir = Path(tmpdir) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save dataset to a temporary file
    file_path1 = tmp_path / "test_data1.nc"
    file_path2 = tmp_path / "test_data2.nc"
    get_dataset.to_netcdf(file_path1)
    # modify the dataset for the second file
    # to create ds with different time values
    modified_dataset = get_dataset.copy()
    modified_dataset["time"] = np.array(
        ["2029-01-01T00:00:00", "2030-01-01T00:00:00"], dtype="datetime64"
    )
    # change variable names
    modified_dataset = modified_dataset.rename({"t2m": "t2m_mod", "tp": "tp_mod"})
    # save the modified dataset to a new file
    modified_dataset.to_netcdf(file_path2)

    # save nuts data to a temporary file
    get_nuts_data.to_file(tmp_path / "nuts.shp")

    # aggregate data by NUTS regions with different time values
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path1, None), "era5_mod": (file_path2, None)},
        tmp_path / "nuts.shp",
        normalize_time=True,
        output_dir=out_dir,
        agg_lib=agg_lib,
    )

    # check if the output file is created
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir
    with xr.open_dataset(out_file) as ds:
        # check if the data is aggregated correctly
        assert "NUTS_ID" in ds.coords
        assert "time" in ds.coords
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
        assert "t2m_mod" in ds.data_vars
        assert "tp_mod" in ds.data_vars

        # check if the time values are doubled
        assert len(ds["time"]) == 4
        assert ds["time"].values.min() == np.datetime64("2024-01-01T00:00:00")
        assert ds["time"].values.max() == np.datetime64("2030-01-01T00:00:00")

        # check if the total number of entries is correct
        assert (
            len(ds["t2m"].values.reshape(-1)) == 8
        )  # two NUTS regions with four time points each

    # clean up the output directory
    for file in out_dir.glob("*"):
        file.unlink()
    out_dir.rmdir()  # remove the output directory after test


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts_dup_netcdfs(
    tmp_path, get_dataset, get_nuts_data, tmpdir, agg_lib
):
    out_dir = Path(tmpdir) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    # save nuts data to a temporary file
    get_nuts_data.to_file(tmp_path / "nuts.shp")

    # aggregate data by NUTS regions with duplicate netcdf files
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path, None), "era5_dup": (file_path, None)},
        tmp_path / "nuts.shp",
        normalize_time=True,
        output_dir=out_dir,
        agg_lib=agg_lib,
    )

    # check if the output file is created
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir
    with xr.open_dataset(out_file) as ds:
        # check if the data is aggregated correctly
        assert "NUTS_ID" in ds.coords
        assert "time" in ds.coords
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
        assert (
            len(ds["t2m"].values.reshape(-1)) == 4
        )  # two NUTS regions with two time points each
        assert len(ds.data_vars) == 2  # only two variables

    # clean up the output directory
    for file in out_dir.glob("*"):
        file.unlink()
    out_dir.rmdir()  # remove the output directory after test


@pytest.mark.parametrize("agg_lib", ["geopandas", "exactextract"])
def test_aggregate_data_by_nuts_overlapping_netcdfs(
    tmp_path, get_dataset, get_nuts_data, tmpdir, agg_lib
):
    out_dir = Path(tmpdir) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save dataset to a temporary file
    file_path1 = tmp_path / "test_data1.nc"
    file_path2 = tmp_path / "test_data2.nc"
    get_dataset.to_netcdf(file_path1)
    # modify the dataset for the second file
    # to create ds with overlapping time values
    modified_dataset = get_dataset.copy()
    modified_dataset["time"] = np.array(
        ["2025-01-01T00:00:00", "2026-01-01T00:00:00"], dtype="datetime64"
    )
    # change variable names
    modified_dataset = modified_dataset.rename({"tp": "tp_mod"})
    modified_dataset.to_netcdf(file_path2)

    # save nuts data to a temporary file
    get_nuts_data.to_file(tmp_path / "nuts.shp")

    # aggregate data by NUTS regions with overlapping time values
    out_file = nuts_aggregation.aggregate_data_by_nuts(
        {"era5": (file_path1, None), "era5_mod": (file_path2, None)},
        tmp_path / "nuts.shp",
        normalize_time=True,
        output_dir=out_dir,
        agg_lib=agg_lib,
    )

    # check if the output file is created
    assert out_file.exists()
    assert out_file.suffix == ".nc"
    assert out_file.parent == out_dir
    with xr.open_dataset(out_file) as ds:
        # check if the data is aggregated correctly
        assert "NUTS_ID" in ds.coords
        assert "time" in ds.coords
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
        assert "tp_mod" in ds.data_vars

        # check if the time values are correct
        assert len(ds["time"]) == 3
        assert ds["time"].values.min() == np.datetime64("2024-01-01T00:00:00")
        assert ds["time"].values.max() == np.datetime64("2026-01-01T00:00:00")

        # check if the total number of entries is correct
        assert ds.sizes.get("NUTS_ID") == 2  # two NUTS regions
        assert len(ds.data_vars) == 3  # three variables

        # check if t2m values are updated correctly for overlapping time
        t2m_time0 = ds["t2m"].sel(time="2024-01-01").values
        t2m_time1 = ds["t2m"].sel(time="2025-01-01").values
        t2m_time2 = ds["t2m"].sel(time="2026-01-01").values
        assert np.allclose(t2m_time0, t2m_time1)
        assert np.allclose(t2m_time2[0], get_dataset["t2m"].values[1, :, 0].mean())
        assert np.allclose(t2m_time2[1], get_dataset["t2m"].values[1, :, 1:].mean())
