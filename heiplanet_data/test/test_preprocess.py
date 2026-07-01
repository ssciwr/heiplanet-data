import pytest
import numpy as np
import xarray as xr
from heiplanet_data import preprocess
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
from conftest import get_files
import json
from datetime import datetime
import xesmf as xe
from cdo import Cdo
import textwrap
import pandas as pd


@pytest.fixture()
def get_data():
    time_points = np.array(["2024-01-01", "2025-01-01"], dtype="datetime64")
    latitude = [0, 0.5]
    longitude = [0, 0.5, 1]
    longitude_first = np.float64(0.0)
    longitude_last = np.float64(359.9)

    # create random data for t2m and tp
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 2, 3)) * 1000 + 273.15
    data_array_t2m = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_points, "latitude": latitude, "longitude": longitude},
    )

    data = rng.random((2, 2, 3)) / 1000
    data_array_precip = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_points, "latitude": latitude, "longitude": longitude},
    )
    data_array_t2m.attrs = {
        "GRIB_units": "K",
        "units": "K",
        "GRIB_longitudeOfFirstGridPointInDegrees": longitude_first,
        "GRIB_longitudeOfLastGridPointInDegrees": longitude_last,
    }
    data_array_precip.attrs = {
        "GRIB_units": "m",
        "units": "m",
        "GRIB_longitudeOfFirstGridPointInDegrees": longitude_first,
        "GRIB_longitudeOfLastGridPointInDegrees": longitude_last,
    }
    return data_array_t2m, data_array_precip


@pytest.fixture()
def get_dataset(get_data):
    data_t2m = get_data[0]
    data_tp = get_data[1]
    dataset = xr.Dataset(
        {"t2m": data_t2m, "tp": data_tp},
        coords={
            "time": data_t2m.time,
            "latitude": (
                "latitude",
                data_t2m.latitude.data,
                {"units": "degrees_north"},
            ),
            "longitude": (
                "longitude",
                data_t2m.longitude.data,
                {"units": "degrees_east"},
            ),
        },
    )
    # create attributes for the dataset
    dataset.attrs.update({"GRIB_centre": "ecmf"})
    return dataset


@pytest.fixture()
def get_nuts_data():
    # create a simple GeoDataFrame with NUTS regions
    data = {
        "NUTS_ID": ["NUTS1", "NUTS2"],
        "geometry": [
            Polygon(
                [
                    (-0.25, -0.25),
                    (-0.25, 1.0),
                    (0.25, 1.0),
                    (0.25, -0.25),
                    (-0.25, -0.25),
                ]
            ),
            Polygon(
                [(0.25, -0.25), (0.25, 1.0), (1.25, 1.0), (1.25, -0.25), (0.25, -0.25)]
            ),
        ],
    }
    nuts_data = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return nuts_data


def test_convert_360_to_180(get_data):
    # convert 360 to 180, xarray
    t2m_data = get_data[0]
    converted_array = preprocess.convert_360_to_180(t2m_data)
    expected_array = (t2m_data + 180) % 360 - 180
    assert np.allclose(converted_array.values, expected_array.values)
    assert converted_array.dims == expected_array.dims
    assert all(
        converted_array.coords[dim].equals(expected_array.coords[dim])
        for dim in converted_array.dims
    )

    # convert with float values
    num = 360.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 0.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 180.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 90.0
    converted_num = preprocess.convert_360_to_180(num)
    assert np.isclose(converted_num, num)

    num = -90.0
    converted_num = preprocess.convert_360_to_180(num)
    assert np.isclose(converted_num, num)


def test_adjust_longitude_360_to_180(get_dataset):
    # invalid lon name
    with pytest.raises(ValueError):
        preprocess.adjust_longitude_360_to_180(get_dataset, lon_name="invalid_lon")
    # full area
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(
        get_dataset, limited_area=False
    )
    expected_dataset = get_dataset.assign_coords(
        longitude=((get_dataset.longitude + 180) % 360 - 180)
    ).sortby("longitude")

    # check if the attributes are preserved
    assert adjusted_dataset.attrs == get_dataset.attrs
    assert adjusted_dataset["t2m"].attrs.get("units") == get_dataset["t2m"].attrs.get(
        "units"
    )
    assert adjusted_dataset["longitude"].attrs == get_dataset["longitude"].attrs
    for var in adjusted_dataset.data_vars.keys():
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfFirstGridPointInDegrees"
        ) == np.float64(-179.9)
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfLastGridPointInDegrees"
        ) == np.float64(180.0)

    # check if the data is adjusted correctly
    assert np.allclose(adjusted_dataset["t2m"].values, expected_dataset["t2m"].values)
    assert adjusted_dataset["t2m"].dims == expected_dataset["t2m"].dims
    assert all(
        adjusted_dataset["t2m"].coords[dim].equals(expected_dataset["t2m"].coords[dim])
        for dim in adjusted_dataset["t2m"].dims
    )

    # limited area
    for var in get_dataset.data_vars.keys():
        get_dataset[var].attrs.update(
            {
                "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-45.0),
                "GRIB_longitudeOfLastGridPointInDegrees": np.float64(45.0),
            }
        )
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(
        get_dataset, limited_area=True
    )
    for var in adjusted_dataset.data_vars.keys():
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfFirstGridPointInDegrees"
        ) == np.float64(-45.0)
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfLastGridPointInDegrees"
        ) == np.float64(45.0)


def test_convert_to_celsius(get_data):
    t2m_data = get_data[0]
    # convert to Celsius, xarray
    celsius_array = preprocess.convert_to_celsius(t2m_data)
    expected_celsius_array = t2m_data - 273.15
    assert np.allclose(celsius_array.values, expected_celsius_array.values)
    assert celsius_array.dims == expected_celsius_array.dims
    assert all(
        celsius_array.coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_array.dims
    )

    # float numbers
    kelvin_temp = 300.0
    celsius_temp = preprocess.convert_to_celsius(kelvin_temp)
    expected_temp = kelvin_temp - 273.15
    assert np.isclose(celsius_temp, expected_temp)


def test_convert_to_celsius_with_attributes_no_inplace(get_dataset):
    # invalid var name
    with pytest.raises(ValueError):
        preprocess.convert_to_celsius_with_attributes(get_dataset, var_name="invalid")
    # convert to Celsius
    celsius_dataset = preprocess.convert_to_celsius_with_attributes(
        get_dataset, inplace=False
    )
    expected_celsius_array = get_dataset["t2m"] - 273.15

    # check if the attributes are preserved
    assert celsius_dataset.attrs == get_dataset.attrs
    assert celsius_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert celsius_dataset["t2m"].attrs.get("units") == "C"

    # check if the data is converted correctly
    assert np.allclose(celsius_dataset["t2m"].values, expected_celsius_array.values)
    assert celsius_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        celsius_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_dataset["t2m"].dims
    )


def test_convert_to_celsius_with_attributes_inplace(get_dataset):
    # convert to Celsius
    org_data_array = get_dataset["t2m"].copy()
    org_ds_attrs = get_dataset.attrs.copy()
    preprocess.convert_to_celsius_with_attributes(get_dataset, inplace=True)
    expected_celsius_array = org_data_array - 273.15

    # check if the attributes are preserved
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert get_dataset["t2m"].attrs.get("units") == "C"

    # check if the data is converted correctly
    assert np.allclose(get_dataset["t2m"].values, expected_celsius_array.values)
    assert get_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        get_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in get_dataset["t2m"].dims
    )


def test_rename_coords(get_dataset):
    renamed_dataset = preprocess.rename_coords(get_dataset, {"longitude": "lon"})

    # check if the coordinates are renamed
    assert "lon" in renamed_dataset.coords
    assert "longitude" not in renamed_dataset.coords

    # check if other data is preserved
    assert np.allclose(renamed_dataset["t2m"].values, get_dataset["t2m"].values)
    assert renamed_dataset["t2m"].dims[0] == get_dataset["t2m"].dims[0]


def test_rename_coords_invalid_mapping(get_dataset):
    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping="")

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping={})

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping=1)

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping={"lon": 2.2})


def test_rename_coords_notexist_coords(get_dataset):
    with pytest.warns(UserWarning):
        renamed_dataset = preprocess.rename_coords(
            get_dataset, {"notexist": "lon", "latitude": "lat"}
        )

    # check if the coordinates are not renamed
    assert "notexist" not in renamed_dataset.coords
    assert "lon" not in renamed_dataset.coords
    assert "longitude" in renamed_dataset.coords
    assert "latitude" not in renamed_dataset.coords
    assert "lat" in renamed_dataset.coords


def test_convert_m_to_mm(get_data):
    tp_data = get_data[1]
    # convert m to mm, xarray
    mm_array = preprocess.convert_m_to_mm(tp_data)
    expected_mm_array = tp_data * 1000.0
    assert np.allclose(mm_array.values, expected_mm_array.values)
    assert mm_array.dims == expected_mm_array.dims
    assert all(
        mm_array.coords[dim].equals(expected_mm_array.coords[dim])
        for dim in mm_array.dims
    )

    # float numbers
    m_precip = 0.001
    mm_precip = preprocess.convert_m_to_mm(m_precip)
    expected_precip = m_precip * 1000.0
    assert np.isclose(mm_precip, expected_precip)


def test_convert_m_to_mm_with_attributes_no_inplace(get_dataset):
    # invalid var name
    with pytest.raises(ValueError):
        preprocess.convert_m_to_mm_with_attributes(get_dataset, var_name="invalid")
    # convert m to mm
    mm_dataset = preprocess.convert_m_to_mm_with_attributes(
        get_dataset, inplace=False, var_name="tp"
    )
    expected_mm_array = get_dataset["tp"] * 1000.0

    # check if the attributes are preserved
    assert mm_dataset.attrs == get_dataset.attrs
    assert mm_dataset["tp"].attrs.get("GRIB_units") == "mm"
    assert mm_dataset["tp"].attrs.get("units") == "mm"

    # check if the data is converted correctly
    assert np.allclose(mm_dataset["tp"].values, expected_mm_array.values)
    assert mm_dataset["tp"].dims == expected_mm_array.dims
    assert all(
        mm_dataset["tp"].coords[dim].equals(expected_mm_array.coords[dim])
        for dim in mm_dataset["tp"].dims
    )


def test_convert_m_to_mm_with_attributes_inplace(get_dataset):
    # convert m to mm
    org_data_array = get_dataset["tp"].copy()
    org_ds_attrs = get_dataset.attrs.copy()
    preprocess.convert_m_to_mm_with_attributes(get_dataset, inplace=True, var_name="tp")
    expected_mm_array = org_data_array * 1000.0

    # check if the attributes are preserved
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["tp"].attrs.get("GRIB_units") == "mm"
    assert get_dataset["tp"].attrs.get("units") == "mm"

    # check if the data is converted correctly
    assert np.allclose(get_dataset["tp"].values, expected_mm_array.values)
    assert get_dataset["tp"].dims == expected_mm_array.dims
    assert all(
        get_dataset["tp"].coords[dim].equals(expected_mm_array.coords[dim])
        for dim in get_dataset["tp"].dims
    )


def test_check_downsample_condition(get_dataset):
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(get_dataset, new_resolution=0.2)
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(
            get_dataset, new_resolution=1.0, agg_funcs="invalid"
        )
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(
            get_dataset,
            new_resolution=1.0,
            lat_name="invalid_lat",
            lon_name="longitude",
        )
    with pytest.raises(ValueError):
        preprocess.check_downsample_condition(
            get_dataset,
            new_resolution=1.0,
            lat_name="latitude",
            lon_name="invalid_lon",
        )


def test_check_agg_funcs():
    with pytest.raises(ValueError):
        preprocess.check_agg_funcs(agg_funcs="invalid", valid_agg_funcs={"mean"})
    with pytest.raises(ValueError):
        preprocess.check_agg_funcs(agg_funcs={}, valid_agg_funcs={"mean"})
    with pytest.raises(ValueError):
        preprocess.check_agg_funcs(
            agg_funcs={"t2m": "invalid"}, valid_agg_funcs={"mean"}
        )
    assert preprocess.check_agg_funcs(agg_funcs=None, valid_agg_funcs={"mean"}) is None


def test_downsample_resolution_with_xarray_default(get_dataset):
    # downsample resolution
    downsampled_dataset = preprocess.downsample_resolution_with_xarray(
        get_dataset, new_resolution=1.0
    )

    # check if the number of dimensions is kept
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(downsampled_dataset["t2m"].latitude.values, [0.25])
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.25])

    # check agg. values
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        np.mean(get_dataset["t2m"][:, :, :2], axis=(1, 2)),
    )

    # check attributes
    assert downsampled_dataset.attrs == get_dataset.attrs
    for var in downsampled_dataset.data_vars.keys():
        assert downsampled_dataset[var].attrs == get_dataset[var].attrs


def test_downsample_resolution_with_xarray_custom(get_dataset):
    # downsample resolution with custom aggregation functions
    agg_funcs = {
        "t2m": "mean",
        "tp": "sum",
    }
    downsampled_dataset = preprocess.downsample_resolution_with_xarray(
        get_dataset, new_resolution=1.0, agg_funcs=agg_funcs
    )

    # check if the number of dimensions is kept
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(downsampled_dataset["t2m"].latitude.values, [0.25])
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.25])

    # check agg. values
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        np.mean(get_dataset["t2m"][:, :, :2], axis=(1, 2)),
    )
    assert np.allclose(
        downsampled_dataset["tp"].values.flatten(),
        np.sum(get_dataset["tp"][:, :, :2], axis=(1, 2)),
    )

    # check attributes
    assert downsampled_dataset.attrs == get_dataset.attrs
    for var in downsampled_dataset.data_vars.keys():
        assert downsampled_dataset[var].attrs == get_dataset[var].attrs


def test_downsample_resolution_with_xarray_missing_agg_func(get_dataset):
    # downsample resolution with missing aggregation functions
    with pytest.warns(UserWarning):
        downsampled_dataset = preprocess.downsample_resolution_with_xarray(
            get_dataset,
            new_resolution=1.0,
            agg_funcs={"tp": "sum"},  # t2m will use mean
        )

    # check agg. values
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        np.mean(get_dataset["t2m"][:, :, :2], axis=(1, 2)),
    )


def test_downsample_resolution_with_xesmf_custom(get_dataset):
    # modify lat lon of the original dataset
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", [0.0, 0.5]),
        longitude=("longitude", [0.0, 0.5, 1.0]),
    )
    # downsample resolution with xesmf
    downsampled_dataset = preprocess.downsample_resolution_with_xesmf(
        get_dataset,
        new_resolution=1.0,
        new_min_lat=0.0,
        new_max_lat=0.5,
        new_min_lon=0.0,
        new_max_lon=1.0,
        lat_name="latitude",
        lon_name="longitude",
        agg_funcs={"t2m": "bilinear", "tp": "conservative"},
    )

    # check if the number of dimensions is kept
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(downsampled_dataset["t2m"].latitude.values, [0.5])
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.0, 1.0])

    # check attributes
    assert downsampled_dataset.attrs == get_dataset.attrs
    for var in downsampled_dataset.data_vars.keys():
        for att in downsampled_dataset[var].attrs.keys():
            if att != "regrid_method":
                assert (
                    downsampled_dataset[var].attrs[att] == get_dataset[var].attrs[att]
                )

    # manually use xesmf to downsample for comparison
    old_lats = get_dataset["latitude"].values
    old_lons = get_dataset["longitude"].values
    old_lat_b = np.arange(max(old_lats) + 0.5, min(old_lats) - 0.5, -0.5)
    old_lon_b = np.arange(min(old_lons) - 0.5, max(old_lons) + 0.5, 0.5)
    get_dataset = get_dataset.assign_coords(
        {
            "lat_b": (["lat_b"], old_lat_b, get_dataset["latitude"].attrs),
            "lon_b": (["lon_b"], old_lon_b, get_dataset["longitude"].attrs),
        }
    )

    new_lats = np.arange(0.5, 0.0 - 0.001, -1.0)
    new_lons = np.arange(0.0, 1.0 + 0.001, 1.0)
    new_lat_b = np.arange(
        max(new_lats) + 1.0,
        min(new_lats) - 1.0,
        -1.0,
    )
    new_lon_b = np.arange(
        min(new_lons) - 1.0,
        max(new_lons) + 1.0,
        1.0,
    )
    ds_out = xr.Dataset(
        {
            "latitude": (["latitude"], new_lats, get_dataset["latitude"].attrs),
            "longitude": (["longitude"], new_lons, get_dataset["longitude"].attrs),
            "lat_b": (["lat_b"], new_lat_b, get_dataset["latitude"].attrs),
            "lon_b": (["lon_b"], new_lon_b, get_dataset["longitude"].attrs),
        }
    )

    regridder_t2m = xe.Regridder(get_dataset, ds_out, "bilinear", periodic=True)
    regridder_tp = xe.Regridder(get_dataset, ds_out, "conservative")
    result_t2m = regridder_t2m(get_dataset["t2m"], keep_attrs=True)
    result_tp = regridder_tp(get_dataset["tp"], keep_attrs=True)

    out_ds = xr.Dataset({"t2m": result_t2m, "tp": result_tp})

    # check if the data is downsampled correctly
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(), out_ds["t2m"].values.flatten()
    )
    assert np.allclose(
        downsampled_dataset["tp"].values.flatten(), out_ds["tp"].values.flatten()
    )


def test_downsample_resolution_with_xesmf_missing_agg_func(get_dataset):
    # modify lat lon of the original dataset
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", [0.0, 0.5]),
        longitude=("longitude", [0.0, 0.5, 1.0]),
    )
    # downsample resolution with xesmf
    with pytest.warns(UserWarning):
        downsampled_dataset = preprocess.downsample_resolution_with_xesmf(
            get_dataset,
            new_resolution=1.0,
            new_min_lat=0.0,
            new_max_lat=0.5,
            new_min_lon=0.0,
            new_max_lon=1.0,
            lat_name="latitude",
            lon_name="longitude",
            agg_funcs={"tp": "conservative"},  # t2m will use bilinear
        )

    # bilinear check
    t2m_old = get_dataset.t2m.values
    t2m_new = downsampled_dataset.t2m.values
    assert np.nanmin(t2m_new) >= np.nanmin(t2m_old) - 1e-6
    assert np.nanmax(t2m_new) <= np.nanmax(t2m_old) + 1e-6


def test_downsample_resolution_with_xesmf_default(get_dataset):
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", [0.0, 0.5]),
        longitude=("longitude", [0.0, 0.5, 1.0]),
    )
    # downsample resolution with xesmf
    downsampled_dataset = preprocess.downsample_resolution_with_xesmf(
        get_dataset,
        new_resolution=1.0,
        new_min_lat=None,
        new_max_lat=None,
        new_min_lon=None,
        new_max_lon=None,
        lat_name="latitude",
        lon_name="longitude",
        agg_funcs=None,
    )

    # bilinear check
    t2m_old = get_dataset.t2m.values
    t2m_new = downsampled_dataset.t2m.values
    assert np.nanmin(t2m_new) >= np.nanmin(t2m_old) - 1e-6
    assert np.nanmax(t2m_new) <= np.nanmax(t2m_old) + 1e-6

    tp_old = get_dataset.tp.values
    tp_new = downsampled_dataset.tp.values
    assert np.nanmin(tp_new) >= np.nanmin(tp_old) - 1e-6
    assert np.nanmax(tp_new) <= np.nanmax(tp_old) + 1e-6


def test_downsample_resolution_with_cdo_default(get_dataset, tmp_path):
    # downsample resolution with cdo
    downsampled_dataset = preprocess.downsample_resolution_with_cdo(
        get_dataset,
        new_resolution=1.0,
        new_min_lat=None,
        new_lat_size=None,
        new_min_lon=None,
        new_lon_size=None,
        lat_name="latitude",
        lon_name="longitude",
        agg_funcs=None,
        gridtype="lonlat",
    )

    # check if the number of dimensions is kept
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(
        downsampled_dataset["t2m"].latitude.values, [0.0]
    )  # TODO: check the difference with xesmf
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.0, 1.0])

    # manually use cdo to downsample for comparison
    cdo = Cdo()
    var_tmp_files = {}
    for var in get_dataset.data_vars:
        var_tmp_file = tmp_path / f"{var}_input.nc"
        get_dataset[[var]].to_netcdf(var_tmp_file)
        var_tmp_files[var] = var_tmp_file

    gridspec = """
        gridtype = lonlat
        xfirst = 0.0
        xinc = 1.0
        xsize = 2
        yfirst = 0.0
        yinc = 1.0
        ysize = 1
    """
    gridspec = textwrap.dedent(gridspec).strip()
    gridspec_file = tmp_path / "gridspec.txt"
    with open(gridspec_file, "w") as f:
        f.write(gridspec)

    result = {}
    for var, var_tmp_file in var_tmp_files.items():
        tmp_ds = cdo.remapbil(
            str(gridspec_file),
            input=str(var_tmp_file),
            returnXDataset=True,
        )
        result[var] = tmp_ds

    out_ds = xr.merge(result.values())

    # check if the data is downsampled correctly
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(), out_ds["t2m"].values.flatten()
    )
    assert np.allclose(
        downsampled_dataset["tp"].values.flatten(), out_ds["tp"].values.flatten()
    )


def test_downsample_resolution_with_cdo_custom(get_dataset):
    downsampled_dataset = preprocess.downsample_resolution_with_cdo(
        get_dataset,
        new_resolution=1.0,
        new_min_lat=0.5,
        new_lat_size=1,
        new_min_lon=0.0,
        new_lon_size=2,
        lat_name="latitude",
        lon_name="longitude",
        agg_funcs={"t2m": "nn", "tp": "nn"},  # nearest neighbor
        gridtype="lonlat",
    )

    assert downsampled_dataset["t2m"].shape == (2, 1, 2)

    assert np.allclose(downsampled_dataset["tp"].latitude.values, [0.5])

    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        get_dataset["t2m"][:, 1, [0, 2]].values.flatten(),
    )


def test_downsample_resolution_with_cdo_missing_agg_func(get_dataset):
    with pytest.warns(UserWarning):
        downsampled_dataset = preprocess.downsample_resolution_with_cdo(
            get_dataset,
            new_resolution=1.0,
            new_min_lat=None,
            new_lat_size=None,
            new_min_lon=None,
            new_lon_size=None,
            lat_name="latitude",
            lon_name="longitude",
            agg_funcs={"tp": "nn"},  # t2m will also use nn
            gridtype="lonlat",
        )

    # bilinear check
    t2m_old = get_dataset.t2m.values
    t2m_new = downsampled_dataset.t2m.values
    assert np.nanmin(t2m_new) >= np.nanmin(t2m_old) - 1e-6
    assert np.nanmax(t2m_new) <= np.nanmax(t2m_old) + 1e-6


def test_downsample_resolution_with_cdo_runtimeerror(get_dataset, monkeypatch):
    class FakeCdo:
        def remapbil(self, *args, **kwargs):
            raise RuntimeError("CDO remapbil failed")

    monkeypatch.setattr("heiplanet_data.preprocess.Cdo", FakeCdo)

    with pytest.raises(RuntimeError) as excinfo:
        preprocess.downsample_resolution_with_cdo(
            get_dataset,
            new_resolution=1.0,
            new_min_lat=None,
            new_lat_size=None,
            new_min_lon=None,
            new_lon_size=None,
            lat_name="latitude",
            lon_name="longitude",
            agg_funcs=None,
            gridtype="lonlat",
        )

    assert "CDO remapbil failed" in str(excinfo.value)


def test_align_lon_lat_with_popu_data_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.align_lon_lat_with_popu_data(get_dataset, lat_name="invalid_lat")
    with pytest.raises(ValueError):
        preprocess.align_lon_lat_with_popu_data(get_dataset, lon_name="invalid_lon")


def test_align_lon_lat_with_popu_data_special_case(get_dataset):
    tmp_lat = [89.8, -89.7]
    tmp_lon = [-179.7, -179.2, 179.8]
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", tmp_lat),
        longitude=("longitude", tmp_lon),
    )
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.75)
    )
    expected_lon = np.array([-179.75, -179.25, 179.75])
    expected_lat = np.array([89.75, -89.75])
    assert np.allclose(aligned_dataset["longitude"].values, expected_lon)
    assert np.allclose(aligned_dataset["latitude"].values, expected_lat)


def test_align_lon_lat_with_popu_data_other_cases(get_dataset):
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.75)
    )
    assert np.allclose(
        aligned_dataset["longitude"].values, get_dataset["longitude"].values
    )
    assert np.allclose(
        aligned_dataset["latitude"].values, get_dataset["latitude"].values
    )

    tmp_lat = [89.8, -89.7]
    tmp_lon = [-179.7, -179.2, 179.8]
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", tmp_lat),
        longitude=("longitude", tmp_lon),
    )
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.0)
    )
    assert np.allclose(
        aligned_dataset["longitude"].values, get_dataset["longitude"].values
    )
    assert np.allclose(
        aligned_dataset["latitude"].values, get_dataset["latitude"].values
    )


def test_upsample_resolution_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=1.0)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(
            get_dataset, new_resolution=0.1, method_map="invalid"
        )
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, lat_name="invalid_lat")
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, lon_name="invalid_lon")


def test_upsample_resolution_default(get_dataset):
    # upsample resolution
    upsampled_dataset = preprocess.upsample_resolution(get_dataset, new_resolution=0.1)

    # check if the dimensions are increased
    assert len(upsampled_dataset["t2m"].dims) == 3
    assert len(upsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(
        upsampled_dataset["t2m"].latitude.values, np.arange(0.0, 0.6, 0.1)
    )
    assert np.allclose(
        upsampled_dataset["t2m"].longitude.values, np.arange(0.0, 1.1, 0.1)
    )

    # check interpolated values
    t2m_interp = upsampled_dataset["t2m"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    t2m_expected = get_dataset["t2m"].interp(
        latitude=0.1, longitude=0.1, method="linear"
    )
    assert np.allclose(t2m_interp.values, t2m_expected.values)
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)

    # check attributes
    assert upsampled_dataset.attrs == get_dataset.attrs
    for var in upsampled_dataset.data_vars.keys():
        assert upsampled_dataset[var].attrs == get_dataset[var].attrs


def test_upsample_resolution_custom(get_dataset):
    # upsample resolution with custom interpolation methods
    method_map = {
        "t2m": "linear",
        "tp": "nearest",
    }
    upsampled_dataset = preprocess.upsample_resolution(
        get_dataset, new_resolution=0.1, method_map=method_map
    )

    # check interpolated values
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    assert np.allclose(tp_interp.values, tp_expected.values)

    # custom map with missing variable
    method_map = {
        "t2m": "linear",
    }  # tp will also use linear interpolation
    upsampled_dataset = preprocess.upsample_resolution(
        get_dataset, new_resolution=0.1, method_map=method_map
    )
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)


def test_resample_resolution_invalid(get_dataset):
    with pytest.raises(ValueError):
        resolution_config = preprocess.ResolutionConfig(new_resolution=-0.5)
        preprocess.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = preprocess.ResolutionConfig(lat_name="invalid_lat")
        preprocess.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = preprocess.ResolutionConfig(lon_name="invalid_lon")
        preprocess.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = preprocess.ResolutionConfig(
            new_resolution=1.0, downsample_lib="invalid_lib"
        )
        preprocess.resample_resolution(get_dataset, resolution_config=resolution_config)


def test_resample_resolution(get_dataset):
    # downsample resolution with xarray
    resampled_dataset_xarray = preprocess.resample_resolution(
        get_dataset,
        resolution_config=preprocess.ResolutionConfig(
            new_resolution=1.0, downsample_lib="xarray"
        ),
    )

    # check if the coordinates are adjusted
    assert np.allclose(resampled_dataset_xarray["tp"].latitude.values, [0.25])
    assert np.allclose(resampled_dataset_xarray["tp"].longitude.values, [0.25])

    # check aggregated values
    assert np.allclose(
        resampled_dataset_xarray["tp"].values.flatten(),
        np.mean(get_dataset["tp"][:, :, :2], axis=(1, 2)),
    )

    # downsample resolution with xesmf
    resampled_dataset_xesmf = preprocess.resample_resolution(
        get_dataset,
        resolution_config=preprocess.ResolutionConfig(
            new_resolution=1.0, downsample_lib="xesmf"
        ),
        grid_config=preprocess.GridConfig(
            new_min_lat=0.0, new_max_lat=0.5, new_min_lon=0.0, new_max_lon=1.0
        ),
    )
    # bilinear check
    tp_old = get_dataset.tp.values
    tp_new = resampled_dataset_xesmf.tp.values
    # bilinear check
    assert np.nanmin(tp_new) >= np.nanmin(tp_old) - 1e-6
    assert np.nanmax(tp_new) <= np.nanmax(tp_old) + 1e-6

    # downsample resolution with cdo
    resampled_dataset_cdo = preprocess.resample_resolution(
        get_dataset,
        resolution_config=preprocess.ResolutionConfig(
            new_resolution=1.0,
            downsample_lib="cdo",
            downsample_agg_funcs={"t2m": "nn", "tp": "nn"},
        ),
        grid_config=preprocess.GridConfig(
            new_min_lat=0.0,
            new_lat_size=1,
            new_min_lon=0.0,
            new_lon_size=2,
            gridtype="lonlat",
        ),
    )

    assert resampled_dataset_cdo["tp"].shape == (2, 1, 2)
    assert np.allclose(
        resampled_dataset_cdo["tp"].values.flatten(),
        get_dataset["tp"][:, 0, [0, 2]].values.flatten(),
    )

    # upsample resolution
    resampled_dataset = preprocess.resample_resolution(
        get_dataset, resolution_config=preprocess.ResolutionConfig(new_resolution=0.1)
    )

    # check if the coordinates are adjusted
    assert np.allclose(
        resampled_dataset["tp"].latitude.values, np.arange(0.0, 0.6, 0.1)
    )
    assert np.allclose(
        resampled_dataset["tp"].longitude.values, np.arange(0.0, 1.1, 0.1)
    )

    # check interpolated values
    tp_interp = resampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)


def test_shift_time_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.shift_time(
            get_dataset, offset="invalid", time_unit="D", var_name="time"
        )
    with pytest.raises(ValueError):
        preprocess.shift_time(
            get_dataset, offset=2.5, time_unit="D", var_name="invalid"
        )
    with pytest.raises(ValueError):
        preprocess.shift_time(get_dataset, offset=2, time_unit="Y", var_name="time")
    with pytest.raises(ValueError):
        preprocess.shift_time(get_dataset, offset=2, time_unit="M", var_name="time")


def test_shift_time_forward(get_dataset):
    original_time = get_dataset["time"].copy()
    # shift time by 2 days
    offset = 2
    time_unit = "D"
    time_shift = np.timedelta64(2, "D")
    preprocess.shift_time(
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
    preprocess.shift_time(
        get_dataset, offset=offset, time_unit=time_unit, var_name="time"
    )
    expected_time = original_time + time_shift.astype("timedelta64[ns]")
    assert np.array_equal(
        np.sort(get_dataset["time"].values), np.sort(expected_time.values)
    )
    assert all(get_dataset["time"].dt.hour.values == 22)


def test_parse_date_invalid():
    with pytest.raises(ValueError):
        preprocess._parse_date(date="invalid_date")

    with pytest.raises(ValueError):
        preprocess._parse_date(date="2024-13-01")

    with pytest.raises(ValueError):
        preprocess._parse_date(date=12345)


def test_parse_date():
    date_str = "2024-07-15"
    parsed_date = preprocess._parse_date(date_str)
    expected_date = np.datetime64("2024-07-15")
    assert parsed_date == expected_date

    date_np = np.datetime64("2025-12-31")
    parsed_date = preprocess._parse_date(date_np)
    assert parsed_date == date_np


def test_truncate_data_by_time_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.truncate_data_by_time(
            get_dataset, start_date=None, end_date=None, var_name="time"
        )
    with pytest.raises(ValueError):
        preprocess.truncate_data_by_time(
            get_dataset, start_date="2025-01-01", end_date="2024-01-01", var_name="time"
        )
    with pytest.raises(ValueError):
        preprocess.truncate_data_by_time(
            get_dataset, start_date="2025-01-01", end_date=None, var_name="invalid_var"
        )


def test_truncate_data_by_time(get_dataset):
    # truncate data by time
    truncated_dataset = preprocess.truncate_data_by_time(
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
    truncated_dataset = preprocess.truncate_data_by_time(
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
    truncated_dataset = preprocess.truncate_data_by_time(
        get_dataset,
        start_date=np.datetime64("2024-07-17"),
        end_date=np.datetime64("2025-01-01"),
        var_name="time",
    )
    assert len(truncated_dataset["t2m"].time) == 1
    assert truncated_dataset["t2m"].time.values[0] == np.datetime64("2025-01-01")

    # None end date
    truncated_dataset = preprocess.truncate_data_by_time(
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
    assert preprocess._check_month_start_data(data) is True

    # invalid case
    months = ["2016-01-15", "2016-03-01"]
    data = xr.DataArray(
        data=np.array(months, dtype="datetime64[ns]"),
        dims=["time"],
    )
    assert preprocess._check_month_start_data(data) is False


def test_calculate_monthly_precipitation_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.calculate_monthly_precipitation(
            get_dataset, var_name="error", time_coord="time"
        )
    with pytest.raises(ValueError):
        preprocess.calculate_monthly_precipitation(
            get_dataset, var_name="tp", time_coord="error"
        )
    # modify time to non-monthly start dates
    get_dataset_invalid = get_dataset.copy()
    get_dataset_invalid = get_dataset_invalid.assign_coords(
        time=("time", [np.datetime64("2024-01-15"), np.datetime64("2025-02-15")])
    )
    with pytest.raises(ValueError):
        preprocess.calculate_monthly_precipitation(
            get_dataset_invalid, var_name="tp", time_coord="time"
        )


def test_calculate_monthly_precipitation(get_dataset):
    org_ds = get_dataset.copy()
    # change time to get different days in month
    get_dataset = get_dataset.assign_coords(
        time=("time", [np.datetime64("2024-01-01"), np.datetime64("2024-02-01")])
    )
    # calculate monthly precipitation
    monthly_dataset = preprocess.calculate_monthly_precipitation(
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


def test_replace_decimal_point():
    assert preprocess._replace_decimal_point(1.0) == "1p0"
    assert preprocess._replace_decimal_point(1.234) == "1p234"
    assert preprocess._replace_decimal_point(0.1) == "01"

    with pytest.raises(ValueError):
        preprocess._replace_decimal_point("1.0")


def test_apply_preprocessing_unify_coords(get_dataset):
    fname_base = "test_data"

    setttings = {
        "unify_coords": True,
        "unify_coords_fname": "unicoords",
        "uni_coords": {"latitude": "lat", "longitude": "lon", "time": "valid_time"},
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=setttings
    )
    # check if the coordinates are renamed
    assert "lat" in preprocessed_dataset.coords
    assert "lon" in preprocessed_dataset.coords
    assert "valid_time" in preprocessed_dataset.coords
    # check if file name is updated
    assert updated_fname == f"{fname_base}_unicoords"


def test_apply_preprocessing_adjust_longitude(get_dataset):
    fname_base = "test_data"

    settings = {
        "adjust_longitude": True,
        "adjust_longitude_fname": "adjlon",
        "adjust_longitude_vname": "longitude",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the longitude is adjusted
    assert np.allclose(
        preprocessed_dataset["tp"].longitude.values,
        (get_dataset["tp"].longitude + 180) % 360 - 180,
    )

    # check if file name is updated
    assert updated_fname == f"{fname_base}_adjlon"


def test_apply_preprocessing_convert_to_celsius(get_dataset):
    org_ds = get_dataset.copy()
    fname_base = "test_data"

    settings = {
        "convert_kelvin_to_celsius": True,
        "convert_kelvin_to_celsius_vname": "t2m",
        "convert_kelvin_to_celsius_fname": "celsius",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the temperature is converted to Celsius
    expected_t2m = org_ds["t2m"] - 273.15  # default inplace
    assert np.allclose(preprocessed_dataset["t2m"].values, expected_t2m.values)

    # check if file name is updated
    assert updated_fname == f"{fname_base}_celsius"


def test_apply_preprocessing_convert_m_to_mm(get_dataset):
    org_ds = get_dataset.copy()
    fname_base = "test_data"

    settings = {
        "convert_m_to_mm_precipitation": True,
        "convert_m_to_mm_precipitation_vname": "tp",
        "convert_m_to_mm_precipitation_fname": "mm",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the precipitation is converted to mm
    expected_tp = org_ds["tp"] * 1000.0  # default inplace
    assert np.allclose(preprocessed_dataset["tp"].values, expected_tp.values)

    # check if file name is updated
    assert updated_fname == f"{fname_base}_mm"


def test_apply_preprocessing_downsample_xarray(get_dataset):
    fname_base = "test_data"

    settings = {
        "resample_grid": True,
        "resample_grid_vname": ["latitude", "longitude"],
        "resample_degree": 1.0,
        "resample_grid_fname": "deg_trim",
        "downsample_lib": "xarray",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the dimensions are reduced
    assert np.allclose(preprocessed_dataset["t2m"].latitude.values, [0.25])
    assert np.allclose(preprocessed_dataset["t2m"].longitude.values, [0.25])

    # check if file name is updated
    assert updated_fname == f"{fname_base}_1p0deg_trim"


def test_apply_preprocessing_downsample_xesmf(get_dataset):
    fname_base = "test_data"

    settings = {
        "resample_grid": True,
        "resample_grid_vname": ["latitude", "longitude"],
        "resample_degree": 1.0,
        "resample_grid_fname": "deg",
        "downsample_lib": "xesmf",
        "downsample_new_min_lat": 0.0,
        "downsample_new_max_lat": 0.5,
        "downsample_new_min_lon": 0.0,
        "downsample_new_max_lon": 1.0,
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the dimensions are reduced
    assert np.allclose(preprocessed_dataset["t2m"].latitude.values, [0.5])
    assert np.allclose(preprocessed_dataset["t2m"].longitude.values, [0.0, 1.0])

    # check if file name is updated
    assert updated_fname == f"{fname_base}_1p0deg"


def test_apply_preprocessing_downsample_cdo(get_dataset):
    fname_base = "test_data"

    settings = {
        "resample_grid": True,
        "resample_grid_vname": ["latitude", "longitude"],
        "resample_degree": 1.0,
        "resample_grid_fname": "deg",
        "downsample_lib": "cdo",
        "downsample_new_min_lat": 0.0,
        "downsample_new_lat_size": 1,
        "downsample_new_min_lon": 0.0,
        "downsample_new_lon_size": 2,
        "downsample_gridtype": "lonlat",
    }
    # preprocess the data file
    preprocessed_dataset, _ = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the dimensions are reduced
    assert np.allclose(preprocessed_dataset["t2m"].latitude.values, [0.0])
    assert np.allclose(preprocessed_dataset["t2m"].longitude.values, [0.0, 1.0])


def test_apply_preprocessing_upsample(get_dataset):
    fname_base = "test_data"

    settings = {
        "resample_grid": True,
        "resample_grid_vname": ["latitude", "longitude"],
        "resample_degree": 0.1,
        "resample_grid_fname": "deg_trim",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the dimensions are increased
    assert np.allclose(
        preprocessed_dataset["t2m"].latitude.values, np.arange(0.0, 0.6, 0.1)
    )
    assert np.allclose(
        preprocessed_dataset["t2m"].longitude.values, np.arange(0.0, 1.1, 0.1)
    )

    # check if file name is updated
    assert updated_fname == f"{fname_base}_01deg_trim"


def test_apply_preprocessing_truncate(get_dataset):
    fname_base = "test_data"

    # case where end year is max year
    settings = {
        "truncate_date": True,
        "truncate_date_from": "2024-01-01",
        "truncate_date_to": "2025-01-01",
        "truncate_date_vname": "time",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the time dimension is retained
    assert len(preprocessed_dataset["t2m"].time) == 2
    assert len(preprocessed_dataset["tp"].time) == 2

    # check if file name is updated
    assert updated_fname == f"{fname_base}_2024-2025"

    # case where end year < max year
    settings = {
        "truncate_date": True,
        "truncate_date_from": "2024-01-01",
        "truncate_date_to": "2024-01-01",
        "truncate_date_vname": "time",
    }

    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the time dimension is reduced
    assert len(preprocessed_dataset["t2m"].time) == 1
    assert len(preprocessed_dataset["tp"].time) == 1

    # check if file name is updated
    assert updated_fname == f"{fname_base}_2024-2024"

    # case where end year is None
    settings = {
        "truncate_date": True,
        "truncate_date_from": "2025-01-01",
        "truncate_date_to": None,
        "truncate_date_vname": "time",
    }

    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the time dimension is reduced
    assert len(preprocessed_dataset["t2m"].time) == 1
    assert len(preprocessed_dataset["tp"].time) == 1

    # check if file name is updated
    assert updated_fname == f"{fname_base}_2025-2025"


def test_apply_preprocessing_calculate_monthly_precipitation(get_dataset):
    fname_base = "test_data"

    settings = {
        "cal_monthly_tp": True,
        "cal_monthly_tp_vname": "tp",
        "cal_monthly_tp_tcoord": "time",
        "cal_monthly_tp_fname": "montp",
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = preprocess._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the time dimension is retained
    assert len(preprocessed_dataset["tp"].time) == 2

    # check if file name is updated
    assert updated_fname == f"{fname_base}_montp"


def test_preprocess_data_file_invalid(tmp_path):
    # invalid file path
    with pytest.raises(ValueError):
        preprocess.preprocess_data_file("", settings="default")

    # non-existing file
    with pytest.raises(ValueError):
        preprocess.preprocess_data_file(tmp_path / "invalid.nc", settings="default")

    # empty file
    empty_file_path = tmp_path / "empty.nc"
    empty_file_path.touch()  # create an empty file
    with pytest.raises(ValueError):
        preprocess.preprocess_data_file(empty_file_path, settings="default")

    # invalid source for settings
    with open(tmp_path / "test_data.nc", "w") as f:
        f.write("This is a test file.")
    with pytest.raises(ValueError):
        preprocess.preprocess_data_file(
            tmp_path / "test_data.nc", source="invalid_source", settings="default"
        )


@pytest.fixture
def get_simple_settings(tmp_path):
    return {
        "output_dir": str(tmp_path),
        "truncate_date": True,
        "truncate_date_from": "2025-01-01",
        "truncate_date_to": "2025-01-01",
        "truncate_date_vname": "time",
    }


def test_preprocess_data_file_tag(tmp_path, get_dataset, get_simple_settings):
    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    with open(tmp_path / "settings.json", "w", newline="", encoding="utf-8") as f:
        json.dump(get_simple_settings, f)

    # preprocess the data file
    preprocessed_dataset, pfname = preprocess.preprocess_data_file(
        netcdf_file=file_path,
        source="era5",
        settings=tmp_path / "settings.json",
        new_settings=None,
        unique_tag="today",
    )

    # check if the time dimension is reduced
    assert len(preprocessed_dataset["t2m"].time) == 1
    assert len(preprocessed_dataset["tp"].time) == 1

    assert pfname == "test_data_2025-2025_today.nc"

    # check if there is new file created
    assert (tmp_path / "test_data_2025-2025_today.nc").exists()
    with xr.open_dataset(tmp_path / "test_data_2025-2025_today.nc") as ds:
        assert len(ds["t2m"].time) == 1
        assert len(ds["tp"].time) == 1
    # check if the settings file is also saved
    assert (tmp_path / "settings_today.json").exists()

    # check when file name ends with raw
    (tmp_path / "test_data_2025-2025_today.nc").unlink()
    file_path = tmp_path / "test_data_raw.nc"
    get_dataset.to_netcdf(file_path)

    _, pfname = preprocess.preprocess_data_file(
        netcdf_file=file_path,
        settings=tmp_path / "settings.json",
        unique_tag="anotherday",
    )
    assert pfname == "test_data_2025-2025_anotherday.nc"
    assert (tmp_path / pfname).exists()
    assert (tmp_path / "settings_anotherday.json").exists()


def test_preprocess_data_file_default_tag(tmp_path, get_dataset, get_simple_settings):
    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    with open(tmp_path / "settings.json", "w", newline="", encoding="utf-8") as f:
        json.dump(get_simple_settings, f)

    # preprocess the data file with auto tag
    _, pfname = preprocess.preprocess_data_file(
        netcdf_file=file_path,
        settings=tmp_path / "settings.json",
        unique_tag=None,
    )

    now = datetime.now()
    prefix_tag = f"ts{now.strftime('%Y%m%d')}-"
    assert prefix_tag in pfname
    # file all files with the prefix tag
    files = get_files(tmp_path, name_phrase=prefix_tag)
    assert len(files) == 2  # one for data and one for settings


def test_preprocess_data_file_diff_outdir(
    tmp_path, get_dataset, tmpdir, get_simple_settings
):
    # save dataset to a temporary file
    file_path = tmp_path / "test_data.nc"
    get_dataset.to_netcdf(file_path)

    settings = get_simple_settings.copy()
    settings["output_dir"] = str(Path(tmpdir) / "data" / "processed")
    with open(tmp_path / "settings.json", "w", newline="", encoding="utf-8") as f:
        json.dump(settings, f)

    # preprocess the data file
    _, pfname = preprocess.preprocess_data_file(
        netcdf_file=file_path,
        settings=tmp_path / "settings.json",
        unique_tag="20250818",
    )

    assert pfname == "test_data_2025-2025_20250818.nc"

    # check if there is new file created in the specified output directory
    # the output dir should be created if it does not exist
    assert (Path(tmpdir) / "data" / "processed" / pfname).exists()
    assert (Path(tmpdir) / "data" / "processed" / "settings_20250818.json").exists()

    # clean up
    (Path(tmpdir) / "data" / "processed" / pfname).unlink()
    (Path(tmpdir) / "data" / "processed" / "settings_20250818.json").unlink()
    (Path(tmpdir) / "data" / "processed").rmdir()


def test_prepare_for_aggregation_normalize(get_dataset):
    # change time to mid-day
    get_dataset["time"] = get_dataset["time"] + np.timedelta64(12, "h")

    # prepare data without time normalization
    ds, _ = preprocess._prepare_for_aggregation(
        get_dataset, normalize_time=False, agg_dict=None
    )
    assert np.unique(ds.time.dt.hour).tolist() == [12]

    # prepare data with time normalization
    ds, _ = preprocess._prepare_for_aggregation(
        get_dataset, normalize_time=True, agg_dict=None
    )
    assert np.unique(ds.time.dt.hour).tolist() == [0]


def test_prepare_for_aggregation_agg_dict(get_dataset):
    # None case
    _, p_agg_dict = preprocess._prepare_for_aggregation(
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
    _, p_agg_dict = preprocess._prepare_for_aggregation(
        get_dataset, normalize_time=False, agg_dict=o_agg_dict
    )
    assert p_agg_dict == o_agg_dict

    # invalid cases
    with pytest.warns(UserWarning):
        _, p_agg_dict = preprocess._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={"t2m": 1}
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = preprocess._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict="something"
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = preprocess._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={}
        )
    assert p_agg_dict == expected_agg_dict
    with pytest.warns(UserWarning):
        _, p_agg_dict = preprocess._prepare_for_aggregation(
            get_dataset, normalize_time=False, agg_dict={"invalid_key": "mean"}
        )
    assert p_agg_dict == expected_agg_dict


def test_aggregate_netcdf_nuts_gpd_invalid(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # change coordinates to invalid names
    get_dataset = get_dataset.rename({"latitude": "lat", "longitude": "lon"})
    get_dataset.to_netcdf(file_path)

    with pytest.raises(ValueError):
        preprocess._aggregate_netcdf_nuts_gpd(
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
    out_data, var_names = preprocess._aggregate_netcdf_nuts_gpd(
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
    out_data, _ = preprocess._aggregate_netcdf_nuts_gpd(
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
        preprocess._aggregate_netcdf_nuts_gpd(
            get_nuts_data, file_path, agg_dict=None, normalize_time=False
        )


def test_aggregate_netcdf_nuts_ee_invalid(tmp_path, get_dataset, get_nuts_data):
    file_path = tmp_path / "test_data.nc"
    # change coordinates to invalid names
    get_dataset = get_dataset.rename({"latitude": "lat", "longitude": "lon"})
    get_dataset.to_netcdf(file_path)

    with pytest.raises(ValueError):
        preprocess._aggregate_netcdf_nuts_ee(
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
    out_data, var_names = preprocess._aggregate_netcdf_nuts_ee(
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
    out_data, _ = preprocess._aggregate_netcdf_nuts_ee(
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
    out_data, variable_names = preprocess._aggregate_netcdf_nuts_ee(
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
    out_data, variable_names = preprocess._aggregate_netcdf_nuts_ee(
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
        preprocess.aggregate_data_by_nuts("something", tmp_path / "nuts.shp")

    # empty dict
    with pytest.raises(ValueError):
        preprocess.aggregate_data_by_nuts({}, tmp_path / "nuts.shp")

    # dict with non-exist file
    with pytest.raises(ValueError):
        preprocess.aggregate_data_by_nuts(
            {"era5": (Path("something"), None)}, tmp_path / "nuts.shp"
        )

    # dict with empty file
    nc_file = tmp_path / "test_data.nc"
    nc_file.touch()  # create an empty file
    with pytest.raises(ValueError):
        preprocess.aggregate_data_by_nuts(
            {"era5": (nc_file, None)}, tmp_path / "nuts.shp"
        )

    # dict with non-nuts data
    with open(nc_file, "w") as f:
        f.write("This is a test file.")
    with pytest.raises(ValueError):
        preprocess.aggregate_data_by_nuts(
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
        preprocess.aggregate_data_by_nuts(
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
        preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
    out_file = preprocess.aggregate_data_by_nuts(
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
