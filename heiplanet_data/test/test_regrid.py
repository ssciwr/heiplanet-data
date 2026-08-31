import textwrap

import numpy as np
import pytest
import xarray as xr
import xesmf as xe
from cdo import Cdo

from heiplanet_data import regrid


def test_check_downsample_condition(get_dataset):
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(get_dataset, new_resolution=0.2)
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(
            get_dataset, new_resolution=1.0, agg_funcs="invalid"
        )
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(
            get_dataset,
            new_resolution=1.0,
            lat_name="invalid_lat",
            lon_name="longitude",
        )
    with pytest.raises(ValueError):
        regrid.check_downsample_condition(
            get_dataset,
            new_resolution=1.0,
            lat_name="latitude",
            lon_name="invalid_lon",
        )


def test_check_agg_funcs():
    with pytest.raises(ValueError):
        regrid.check_agg_funcs(agg_funcs="invalid", valid_agg_funcs={"mean"})
    with pytest.raises(ValueError):
        regrid.check_agg_funcs(agg_funcs={}, valid_agg_funcs={"mean"})
    with pytest.raises(ValueError):
        regrid.check_agg_funcs(agg_funcs={"t2m": "invalid"}, valid_agg_funcs={"mean"})
    assert regrid.check_agg_funcs(agg_funcs=None, valid_agg_funcs={"mean"}) is None


def test_downsample_resolution_with_xarray_default(get_dataset):
    # downsample resolution
    downsampled_dataset = regrid.downsample_resolution_with_xarray(
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
    downsampled_dataset = regrid.downsample_resolution_with_xarray(
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
        downsampled_dataset = regrid.downsample_resolution_with_xarray(
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
    downsampled_dataset = regrid.downsample_resolution_with_xesmf(
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
        downsampled_dataset = regrid.downsample_resolution_with_xesmf(
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
    downsampled_dataset = regrid.downsample_resolution_with_xesmf(
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
    downsampled_dataset = regrid.downsample_resolution_with_cdo(
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
    downsampled_dataset = regrid.downsample_resolution_with_cdo(
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
        downsampled_dataset = regrid.downsample_resolution_with_cdo(
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

    monkeypatch.setattr("heiplanet_data.regrid.Cdo", FakeCdo)

    with pytest.raises(RuntimeError) as excinfo:
        regrid.downsample_resolution_with_cdo(
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
        regrid.align_lon_lat_with_popu_data(get_dataset, lat_name="invalid_lat")
    with pytest.raises(ValueError):
        regrid.align_lon_lat_with_popu_data(get_dataset, lon_name="invalid_lon")


def test_align_lon_lat_with_popu_data_special_case(get_dataset):
    tmp_lat = [89.8, -89.7]
    tmp_lon = [-179.7, -179.2, 179.8]
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", tmp_lat),
        longitude=("longitude", tmp_lon),
    )
    aligned_dataset = regrid.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.75)
    )
    expected_lon = np.array([-179.75, -179.25, 179.75])
    expected_lat = np.array([89.75, -89.75])
    assert np.allclose(aligned_dataset["longitude"].values, expected_lon)
    assert np.allclose(aligned_dataset["latitude"].values, expected_lat)


def test_align_lon_lat_with_popu_data_other_cases(get_dataset):
    aligned_dataset = regrid.align_lon_lat_with_popu_data(
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
    aligned_dataset = regrid.align_lon_lat_with_popu_data(
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
        regrid.upsample_resolution(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        regrid.upsample_resolution(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        regrid.upsample_resolution(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        regrid.upsample_resolution(get_dataset, new_resolution=1.0)
    with pytest.raises(ValueError):
        regrid.upsample_resolution(
            get_dataset, new_resolution=0.1, method_map="invalid"
        )
    with pytest.raises(ValueError):
        regrid.upsample_resolution(get_dataset, lat_name="invalid_lat")
    with pytest.raises(ValueError):
        regrid.upsample_resolution(get_dataset, lon_name="invalid_lon")


def test_upsample_resolution_default(get_dataset):
    # upsample resolution
    upsampled_dataset = regrid.upsample_resolution(get_dataset, new_resolution=0.1)

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
    upsampled_dataset = regrid.upsample_resolution(
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
    upsampled_dataset = regrid.upsample_resolution(
        get_dataset, new_resolution=0.1, method_map=method_map
    )
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)


def test_resample_resolution_invalid(get_dataset):
    with pytest.raises(ValueError):
        resolution_config = regrid.ResolutionConfig(new_resolution=-0.5)
        regrid.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = regrid.ResolutionConfig(lat_name="invalid_lat")
        regrid.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = regrid.ResolutionConfig(lon_name="invalid_lon")
        regrid.resample_resolution(get_dataset, resolution_config=resolution_config)
    with pytest.raises(ValueError):
        resolution_config = regrid.ResolutionConfig(
            new_resolution=1.0, downsample_lib="invalid_lib"
        )
        regrid.resample_resolution(get_dataset, resolution_config=resolution_config)


def test_resample_resolution(get_dataset):
    # downsample resolution with xarray
    resampled_dataset_xarray = regrid.resample_resolution(
        get_dataset,
        resolution_config=regrid.ResolutionConfig(
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
    resampled_dataset_xesmf = regrid.resample_resolution(
        get_dataset,
        resolution_config=regrid.ResolutionConfig(
            new_resolution=1.0, downsample_lib="xesmf"
        ),
        grid_config=regrid.GridConfig(
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
    resampled_dataset_cdo = regrid.resample_resolution(
        get_dataset,
        resolution_config=regrid.ResolutionConfig(
            new_resolution=1.0,
            downsample_lib="cdo",
            downsample_agg_funcs={"t2m": "nn", "tp": "nn"},
        ),
        grid_config=regrid.GridConfig(
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
    resampled_dataset = regrid.resample_resolution(
        get_dataset, resolution_config=regrid.ResolutionConfig(new_resolution=0.1)
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
