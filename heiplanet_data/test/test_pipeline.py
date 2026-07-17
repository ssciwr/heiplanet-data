import json
from datetime import datetime
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from conftest import get_files
from heiplanet_data import pipeline, utils


def test_apply_preprocessing_unify_coords(get_dataset):
    fname_base = "test_data"

    settings = {
        "unify_coords": True,
        "unify_coords_fname": "unicoords",
        "uni_coords": {"latitude": "lat", "longitude": "lon", "time": "valid_time"},
    }
    # preprocess the data file
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
        get_dataset, fname_base, settings=settings
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, _ = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
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
    preprocessed_dataset, updated_fname = pipeline._apply_preprocessing(
        get_dataset, fname_base, settings=settings
    )

    # check if the time dimension is retained
    assert len(preprocessed_dataset["tp"].time) == 2

    # check if file name is updated
    assert updated_fname == f"{fname_base}_montp"


def test_preprocess_data_file_invalid(tmp_path):
    # invalid file path
    with pytest.raises(ValueError):
        pipeline.preprocess_data_file("", settings="default")

    # non-existing file
    with pytest.raises(ValueError):
        pipeline.preprocess_data_file(tmp_path / "invalid.nc", settings="default")

    # empty file
    empty_file_path = tmp_path / "empty.nc"
    empty_file_path.touch()  # create an empty file
    with pytest.raises(ValueError):
        pipeline.preprocess_data_file(empty_file_path, settings="default")

    # invalid source for settings
    with open(tmp_path / "test_data.nc", "w") as f:
        f.write("This is a test file.")
    with pytest.raises(ValueError):
        pipeline.preprocess_data_file(
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
    preprocessed_dataset, pfname = pipeline.preprocess_data_file(
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

    _, pfname = pipeline.preprocess_data_file(
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
    _, pfname = pipeline.preprocess_data_file(
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
    _, pfname = pipeline.preprocess_data_file(
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


def test_apply_preprocessing_wind_height_unchanged():
    # create a dataset with an additional "height" dimension (ERA5 wind-like)
    time_points = np.array(["2024-01-01", "2025-01-01"], dtype="datetime64[ns]")
    height = [10.0, 100.0]
    latitude = [0.0, 0.5]
    longitude = [0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 2, 2, 3))
    data_array_u = xr.DataArray(
        data,
        dims=["time", "height", "latitude", "longitude"],
        coords={
            "time": time_points,
            "height": height,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    dataset = xr.Dataset({"u10": data_array_u})

    # record original height and data
    org_height = dataset["height"].values.copy()
    org_data = dataset["u10"].values.copy()

    # no preprocessing step should touch the "height" coordinate
    settings = {
        "unify_coords": False,
        "adjust_longitude": False,
        "convert_kelvin_to_celsius": False,
        "convert_m_to_mm_precipitation": False,
        "resample_grid": False,
        "truncate_date": False,
        "cal_monthly_tp": False,
    }
    preprocessed_dataset, _ = pipeline._apply_preprocessing(
        dataset, "test_wind", settings=settings
    )

    # check if height is unchanged
    assert "height" in preprocessed_dataset.coords
    assert np.array_equal(preprocessed_dataset["height"].values, org_height)

    # check if data is unchanged
    assert "height" in preprocessed_dataset["u10"].dims
    assert np.array_equal(preprocessed_dataset["u10"].values, org_data)

    # no preprocessing step should touch the "height" coordinate
    settings = {
        "unify_coords": True,
        "uni_coords": {
            "latitude": "latitude",
            "longitude": "longitude",
            "height": "new_height",
        },
        "adjust_longitude": False,
        "convert_kelvin_to_celsius": False,
        "convert_m_to_mm_precipitation": False,
        "resample_grid": False,
        "truncate_date": False,
        "cal_monthly_tp": False,
    }
    preprocessed_dataset, _ = pipeline._apply_preprocessing(
        dataset, "test_wind", settings=settings
    )

    # check if height is unchanged
    assert "new_height" in preprocessed_dataset.coords
    assert np.array_equal(preprocessed_dataset["new_height"].values, org_height)


def test_registered_steps_have_schema_entries():
    """Guard the registry<->schema link in both directions.

    Every preprocessing step registered via ``pipeline.register_step`` must
    have a matching enable-flag property in ``setting_schema.json``, so a newly
    added step cannot silently drift from its configuration schema. Conversely,
    every boolean enable-flag in the schema must have a registered step, so a
    flag cannot be added to the schema without wiring up its step. Also check
    that step order values are unique, keeping the execution sequence
    deterministic.
    """

    schema_path = resources.files("heiplanet_data") / "setting_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_props = set(schema["properties"])
    schema_flags = {
        name
        for name, spec in schema["properties"].items()
        if spec.get("type") == "boolean"
    }

    step_names = {name for _order, name, _fn in pipeline._STEP_REGISTRY}

    missing = step_names - schema_props
    assert not missing, f"registered steps missing from schema: {sorted(missing)}"

    orphan_flags = schema_flags - step_names
    assert not orphan_flags, (
        f"schema enable-flags without a registered step: {sorted(orphan_flags)}"
    )

    orders = [order for order, _name, _fn in pipeline._STEP_REGISTRY]
    assert len(orders) == len(set(orders)), f"duplicate step order values: {orders}"


@pytest.mark.parametrize("source", ["era5", "isimip"])
def test_shipped_settings_files_validate_against_schema(source):
    """Regression guard: shipped default settings files honour the schema.

    Each default settings JSON (``era5_settings.json``, ``isimip_settings.json``)
    must validate against ``setting_schema.json``, so an edit that introduces an
    unknown key, a wrong type, or a missing conditionally-required field is
    caught here instead of at runtime.
    """

    setting_path = utils.DEFAULT_SETTINGS_FILE[source]
    settings = json.loads(setting_path.read_text(encoding="utf-8"))
    assert utils.is_valid_settings(settings), (
        f"shipped settings file {setting_path} does not validate against schema"
    )


def test_replace_decimal_point():
    assert pipeline._replace_decimal_point(1.0) == "1p0"
    assert pipeline._replace_decimal_point(1.234) == "1p234"
    assert pipeline._replace_decimal_point(0.1) == "01"

    with pytest.raises(ValueError):
        pipeline._replace_decimal_point("1.0")
