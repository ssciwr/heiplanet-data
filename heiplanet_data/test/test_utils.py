import pytest
import json
from pathlib import Path
from heiplanet_data import utils
from datetime import datetime
from conftest import get_files
from itertools import product
import pandas as pd


def test_is_non_empty_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    # file is not created yet
    assert utils.is_non_empty_file(file_path) is False

    # create an empty file
    file_path.touch()
    assert utils.is_non_empty_file(file_path) is False

    # create a non-empty file
    file_path.write_text("test")
    assert utils.is_non_empty_file(file_path) is True


def test_is_valid_settings():
    settings = {"output_dir": "data/processed"}
    assert utils.is_valid_settings(settings) is True
    settings = {"output_dir": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"adjust_longitude": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude_vname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"adjust_longitude_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude_fname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"adjust_longitude_fname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude_fname": None}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude_fname": ""}
    assert utils.is_valid_settings(settings) is True
    settings = {"adjust_longitude": True}
    assert utils.is_valid_settings(settings) is False
    settings = {"adjust_longitude": True, "adjust_longitude_fname": "test"}
    assert utils.is_valid_settings(settings) is False

    settings = {"convert_kelvin_to_celsius": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_kelvin_to_celsius": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_kelvin_to_celsius_vname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_kelvin_to_celsius_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_kelvin_to_celsius_fname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_kelvin_to_celsius_fname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_kelvin_to_celsius": True}
    assert utils.is_valid_settings(settings) is False
    settings = {
        "convert_kelvin_to_celsius": True,
        "convert_kelvin_to_celsius_fname": "test",
    }
    assert utils.is_valid_settings(settings) is False

    settings = {"convert_m_to_mm_precipitation": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_m_to_mm_precipitation": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_m_to_mm_precipitation_vname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_m_to_mm_precipitation_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_m_to_mm_precipitation_fname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_m_to_mm_precipitation_fname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"convert_m_to_mm_precipitation": True}
    assert utils.is_valid_settings(settings) is False
    settings = {
        "convert_m_to_mm_precipitation": True,
        "convert_m_to_mm_precipitation_fname": "test",
    }
    assert utils.is_valid_settings(settings) is False

    settings = {"resample_grid": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"resample_grid": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"resample_degree": 1}
    assert utils.is_valid_settings(settings) is True
    settings = {"resample_degree": 1.5}
    assert utils.is_valid_settings(settings) is True
    settings = {"resample_degree": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"resample_grid_vname": ["test1", "test2"]}
    assert utils.is_valid_settings(settings) is True
    settings = {"resample_grid_vname": "test"}
    assert utils.is_valid_settings(settings) is False
    settings = {"resample_grid_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"resample_grid_fname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"resample_grid_fname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"resample_grid": True}
    assert utils.is_valid_settings(settings) is False
    settings = {
        "resample_grid": True,
        "resample_grid_vname": ["test1", "test2"],
        "resample_grid_fname": "test",
    }
    assert utils.is_valid_settings(settings) is False
    settings = {
        "resample_grid": True,
        "resample_degree": 1.0,
        "resample_grid_vname": ["test1", "test2"],
        "resample_grid_fname": "test",
        "downsample_lib": "xesmf",
    }
    assert utils.is_valid_settings(settings) is True

    settings = {"truncate_date": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"truncate_date": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date_from": "2025-02-01"}
    assert utils.is_valid_settings(settings) is True
    settings = {"truncate_date_from": 1.5}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date_from": 2025}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date_to": "2025-02-01"}
    assert utils.is_valid_settings(settings) is True
    settings = {"truncate_date_to": 1.5}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date_to": 2025}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date_vname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"truncate_date_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date": True}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_date": True, "truncate_date_from": "2025-02-01"}
    assert utils.is_valid_settings(settings) is False

    settings = {"unify_coords": False}
    assert utils.is_valid_settings(settings) is True
    settings = {"unify_coords": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"unify_coords_fname": "test"}
    assert utils.is_valid_settings(settings) is True
    settings = {"unify_coords_fname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"uni_coords": {"t2m": "temperature"}}
    assert utils.is_valid_settings(settings) is True
    settings = {"runi_coordsname": {"t2m": 1}}
    assert utils.is_valid_settings(settings) is False
    settings = {"uni_coords": {"t2m": "temperature", "error": 1}}
    assert utils.is_valid_settings(settings) is False
    settings = {"uni_coords": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"unify_coords": True}
    assert utils.is_valid_settings(settings) is False
    settings = {"unify_coords": True, "unify_coords_fname": "test"}
    assert utils.is_valid_settings(settings) is False

    settings = {
        "cal_monthly_tp": True,
        "cal_monthly_tp_vname": "tp",
        "cal_monthly_tp_tcoord": "time",
        "cal_monthly_tp_fname": "montp",
    }
    assert utils.is_valid_settings(settings) is True
    settings = {"cal_monthly_tp": "error"}
    assert utils.is_valid_settings(settings) is False
    settings = {"cal_monthly_tp": True}
    assert utils.is_valid_settings(settings) is False
    settings = {"cal_monthly_tp_vname": 1}
    assert utils.is_valid_settings(settings) is False
    settings = {"cal_monthly_tp_tcoord": True}
    assert utils.is_valid_settings(settings) is False
    settings = {"cal_monthly_tp_fname": {}}
    assert utils.is_valid_settings(settings) is False


def test_update_new_settings_empty():
    updated = utils._update_new_settings({"test": "test"}, {})
    assert updated is False

    with pytest.raises(ValueError):
        utils._update_new_settings({}, {"test": "test"})


def test_update_new_settings_not_updated():
    # invalid key
    with pytest.warns(UserWarning):
        updated = utils._update_new_settings(
            {"adjust_longitude": True}, {"test": "test"}
        )
    assert updated is False

    # invalid structure
    updated = utils._update_new_settings(
        {"adjust_longitude": True}, {"adjust_longitude": 1}
    )
    assert updated is False
    with pytest.warns(UserWarning):
        updated = utils._update_new_settings(
            {"uni_coords": {"t2m": "temperature"}}, {"uni_coords": {"t2m": 1}}
        )
    assert updated is False

    # same value
    updated = utils._update_new_settings(
        {"adjust_longitude": False}, {"adjust_longitude": False}
    )
    assert updated is False
    updated = utils._update_new_settings(
        {"uni_coords": {"t2m": "temperature"}}, {"uni_coords": {"t2m": "temperature"}}
    )
    assert updated is False


def test_update_new_settings_updated():
    settings = {
        "adjust_longitude": True,
        "adjust_longitude_vname": "test",
        "adjust_longitude_fname": "test",
    }
    updated = utils._update_new_settings(settings, {"adjust_longitude": False})
    assert updated is True
    assert settings.get("adjust_longitude") is False

    settings = {"uni_coords": {"t2m": "temperature"}}
    updated = utils._update_new_settings(
        settings, {"uni_coords": {"t2m": "temp", "tcc": "cloud_cover"}}
    )
    assert updated is True
    assert settings.get("uni_coords") == {"t2m": "temp", "tcc": "cloud_cover"}


def test_save_settings_to_file(tmpdir):
    settings = {"adjust_longitude": False}

    # none dir path
    utils.save_settings_to_file(settings)
    saved_files = get_files(Path.cwd(), "updated_settings")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False
    saved_files[0].unlink()  # remove the file

    # valid dir path
    directory = Path(tmpdir.mkdir("test"))
    utils.save_settings_to_file(settings, directory)
    saved_files = get_files(directory, "updated_settings")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False

    # invalid dir path
    file_path = Path(__file__).absolute()
    with pytest.raises(ValueError):
        utils.save_settings_to_file(settings, file_path)

    # different file name
    utils.save_settings_to_file(settings, directory, "test_settings.json")
    saved_files = get_files(directory, "test_settings")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False


def test_load_settings_default():
    settings, _ = utils.load_settings()
    assert settings.get("adjust_longitude") is True

    settings, fname = utils.load_settings(source="era5", setting_path="default")
    assert settings.get("adjust_longitude") is True
    assert fname == "era5_settings"


def test_load_settings_file(tmp_path):
    setting_path = tmp_path / "settings.json"

    # invalid cases
    # no default settings file
    with pytest.raises(ValueError):
        utils.load_settings(source="invalid_source")

    # not existing file
    with pytest.warns(UserWarning):
        settings, fname = utils.load_settings("era5", setting_path)
    assert settings.get("adjust_longitude") is True
    assert fname == "era5_settings"

    # empty file
    open(setting_path, "w", newline="", encoding="utf-8").close()
    with pytest.warns(UserWarning):
        settings, _ = utils.load_settings("era5", setting_path)
    assert settings.get("adjust_longitude") is True

    # invalid json file
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        f.write("test")
    with pytest.warns(UserWarning):
        settings, fname = utils.load_settings("era5", setting_path)
    assert settings.get("adjust_longitude") is True
    assert fname == "era5_settings"

    # invalid json file against the schema
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"test": "test"}, f)
    with pytest.warns(UserWarning):
        settings, _ = utils.load_settings("era5", setting_path)
    assert settings.get("adjust_longitude") is True

    # valid json file
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"adjust_longitude": False}, f)
    settings, fname = utils.load_settings(setting_path=setting_path)
    assert settings.get("adjust_longitude") is False
    assert fname == "settings"


def test_load_settings_new_settings(tmp_path, tmpdir):
    new_settings = {"adjust_longitude": False}

    # update default settings
    settings, _ = utils.load_settings(new_settings=new_settings)
    assert settings.get("adjust_longitude") is False

    # update settings from file
    setting_path = tmp_path / "settings.json"
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump(
            {
                "adjust_longitude": True,
                "adjust_longitude_vname": "test",
                "adjust_longitude_fname": "test",
            },
            f,
        )
    settings, _ = utils.load_settings(setting_path=setting_path)
    assert settings.get("adjust_longitude_vname") == "test"
    settings, _ = utils.load_settings(
        setting_path=setting_path, new_settings=new_settings
    )
    assert settings.get("adjust_longitude") is False

    # update settings from file with invalid new settings
    new_settings = {"test": "test"}
    with pytest.warns(UserWarning):
        settings, _ = utils.load_settings(
            setting_path=setting_path, new_settings=new_settings
        )
    assert settings.get("adjust_longitude") is True


def test_generate_unique_tag():
    unique_tag = utils.generate_unique_tag()
    assert isinstance(unique_tag, str)
    assert (
        len(unique_tag.split("_")) == 2
    )  # should be in the format "YYYYMMDD-HHMMSS_hostname"

    # Check if the timestamp is in the correct format
    datetime_part, hostname_part = unique_tag.split("_")
    assert "ts" in datetime_part  # should start with "ts"
    datetime.strptime(datetime_part[2:], "%Y%m%d-%H%M%S")

    # Check if the hostname is a valid string
    assert "h" in hostname_part  # should start with "h"
    assert isinstance(hostname_part, str) and len(hostname_part) > 0


def test_split_date_range_by_full_years():
    # sample date
    start_time = datetime.strptime("2016-01-02", "%Y-%m-%d")
    end_time = datetime.strptime("2018-01-01", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 3
    assert ranges[0] == (start_time, datetime.strptime("2016-12-31", "%Y-%m-%d"))
    assert ranges[1] == (
        datetime.strptime("2017-01-01", "%Y-%m-%d"),
        datetime.strptime("2017-12-31", "%Y-%m-%d"),
    )
    assert ranges[2] == (datetime.strptime("2018-01-01", "%Y-%m-%d"), end_time)

    # same year
    start_time = datetime.strptime("2025-03-15", "%Y-%m-%d")
    end_time = datetime.strptime("2025-10-20", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 1
    assert ranges[0] == (start_time, end_time)

    # same year, month
    start_time = datetime.strptime("2025-03-15", "%Y-%m-%d")
    end_time = datetime.strptime("2025-03-20", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 1
    assert ranges[0] == (start_time, end_time)

    # mid at both ends
    start_time = datetime.strptime("2025-03-15", "%Y-%m-%d")
    end_time = datetime.strptime("2027-10-20", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 3
    assert ranges[0] == (start_time, datetime.strptime("2025-12-31", "%Y-%m-%d"))
    assert ranges[1] == (
        datetime.strptime("2026-01-01", "%Y-%m-%d"),
        datetime.strptime("2026-12-31", "%Y-%m-%d"),
    )
    assert ranges[2] == (datetime.strptime("2027-01-01", "%Y-%m-%d"), end_time)

    # mid at both ends, 1 year apart
    start_time = datetime.strptime("2025-03-15", "%Y-%m-%d")
    end_time = datetime.strptime("2026-10-20", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 2
    assert ranges[0] == (start_time, datetime.strptime("2025-12-31", "%Y-%m-%d"))
    assert ranges[1] == (datetime.strptime("2026-01-01", "%Y-%m-%d"), end_time)

    # full years at both ends
    start_time = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_time = datetime.strptime("2026-12-31", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 1
    assert ranges[0] == (start_time, end_time)

    # full year at start, mid at end
    start_time = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_time = datetime.strptime("2026-10-20", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 2
    assert ranges[0] == (start_time, datetime.strptime("2025-12-31", "%Y-%m-%d"))
    assert ranges[1] == (datetime.strptime("2026-01-01", "%Y-%m-%d"), end_time)

    # mid at start, full year at end
    start_time = datetime.strptime("2025-03-15", "%Y-%m-%d")
    end_time = datetime.strptime("2026-12-31", "%Y-%m-%d")
    ranges = utils.split_date_range_by_full_years(start_time, end_time)
    assert len(ranges) == 2
    assert ranges[0] == (start_time, datetime.strptime("2025-12-31", "%Y-%m-%d"))
    assert ranges[1] == (datetime.strptime("2026-01-01", "%Y-%m-%d"), end_time)


def test_extract_time_from_range():
    all_months = [str(i).zfill(2) for i in range(1, 13)]
    all_days = [str(i).zfill(2) for i in range(1, 32)]
    all_hours = [f"{hour:02d}:00" for hour in range(24)]

    # diff years, full months, days, and hours
    start_time = datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-12-31 23:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2024", "2025"]
    assert months == all_months
    assert days == all_days
    assert hours == all_hours
    assert truncate is False

    # diff years, partial months, days, or hours
    start_time = datetime.strptime("2026-03-15 01:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2027-10-20 15:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2026", "2027"]
    assert months == all_months
    assert days == all_days
    assert hours == all_hours
    assert truncate is True

    # same years, full months, days, and hours
    start_time = datetime.strptime("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-12-31 23:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2025"]
    assert months == all_months
    assert days == all_days
    assert hours == all_hours
    assert truncate is False

    # same years, diff months, midnight at both ends
    start_time = datetime.strptime("2025-03-10 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-10-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2025"]
    assert months == [str(i).zfill(2) for i in range(3, 11)]
    assert days == all_days
    assert hours == ["00:00"]
    assert truncate is True

    # same years, same months, diff days, diff hours
    start_time = datetime.strptime("2025-05-10 12:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-05-25 21:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2025"]
    assert months == ["05"]
    assert days == [str(i).zfill(2) for i in range(10, 26)]
    assert hours == all_hours
    assert truncate is True

    # same years, same months, diff days, midnight at both ends
    start_time = datetime.strptime("2025-05-10 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-05-25 00:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2025"]
    assert months == ["05"]
    assert days == [str(i).zfill(2) for i in range(10, 26)]
    assert hours == ["00:00"]
    assert truncate is False

    # same years, same months, same days, diff hours
    start_time = datetime.strptime("2025-05-10 12:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-05-10 21:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert years == ["2025"]
    assert months == ["05"]
    assert days == ["10"]
    assert hours == [f"{hour:02d}:00" for hour in range(12, 22)]
    assert truncate is False

    # same years, same months, same days, midnight at both ends
    start_time = datetime.strptime("2025-05-10 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2025-05-10 00:00:00", "%Y-%m-%d %H:%M:%S")
    years, months, days, hours, truncate = utils.extract_time_from_range(
        start_time, end_time
    )
    assert hours == ["00:00"]
    assert truncate is False


def test_compress_time_points_to_ymdt_from_missingtps():
    # full time points
    full_y = [str(i) for i in range(2020, 2023)]
    full_m = [str(i).zfill(2) for i in range(1, 5)]
    full_d = [str(i).zfill(2) for i in range(1, 5)]
    full_t = [f"{hour:02d}:00" for hour in range(0, 12)]
    full_tps = set(product(full_y, full_m, full_d, full_t))

    # exising time points
    e_y = ["2020"]
    e_m = ["01", "02"]
    e_d = ["01", "02"]
    e_t = ["00:00", "01:00"]
    existing_tps = set(product(e_y, e_m, e_d, e_t))

    # missing time points
    missing_tps = full_tps - existing_tps

    # compress time points
    compressed = utils.compress_time_points_to_ymdt(missing_tps)

    # check if the compressed time points are correct
    assert len(compressed) == 4
    y_compressed = compressed[3]
    m_compressed = compressed[2]
    d_compressed = compressed[1]
    t_compressed = compressed[0]

    assert y_compressed.get("year") == sorted(list(set(full_y) - set(e_y)))
    assert y_compressed.get("month") == full_m
    assert y_compressed.get("day") == full_d
    assert y_compressed.get("time") == full_t

    assert m_compressed.get("year") == e_y
    assert m_compressed.get("month") == sorted(list(set(full_m) - set(e_m)))
    assert m_compressed.get("day") == full_d
    assert m_compressed.get("time") == full_t

    assert d_compressed.get("year") == e_y
    assert d_compressed.get("month") == e_m
    assert d_compressed.get("day") == sorted(list(set(full_d) - set(e_d)))
    assert d_compressed.get("time") == full_t

    assert t_compressed.get("year") == e_y
    assert t_compressed.get("month") == e_m
    assert t_compressed.get("day") == e_d
    assert t_compressed.get("time") == sorted(list(set(full_t) - set(e_t)))


def test_compress_time_points_to_ymdt_complex_synthetic():
    start_date = "2026-01-01"
    end_date = "2027-09-05"

    date_rng = pd.date_range(start=start_date, end=end_date, freq="h", inclusive="both")

    full_tps = set()
    for drng in date_rng:
        time_tuples = [
            f"{drng.year:04d}",
            f"{drng.month:02d}",
            f"{drng.day:02d}",
            f"{drng.hour:02d}:00",
        ]
        full_tps.add(tuple(time_tuples))

    compressed = utils.compress_time_points_to_ymdt(full_tps)

    # TODO: assertment for this case is quite complicated
    # the function even grouped month with same number of days,
    # e.g. Jan, March, May, etc.
    assert len(compressed) > 2  # 2016 and 2017


def test_compress_time_points_to_ymdt_simple_synthetic():
    start_date = "2025-01-01 00:00:00"
    end_date = "2025-03-31 23:00:00"

    date_rng = pd.date_range(start=start_date, end=end_date, freq="h", inclusive="both")

    full_tps = set()
    for drng in date_rng:
        time_tuples = [
            f"{drng.year:04d}",
            f"{drng.month:02d}",
            f"{drng.day:02d}",
            f"{drng.hour:02d}:00",
        ]
        full_tps.add(tuple(time_tuples))

    compressed = utils.compress_time_points_to_ymdt(full_tps)

    assert len(compressed) == 2
    assert compressed[0].get("year") == ["2025"]
    assert compressed[0].get("month") == ["01", "03"]
    assert compressed[0].get("day") == [str(i).zfill(2) for i in range(1, 32)]
    assert compressed[0].get("time") == [f"{hour:02d}:00" for hour in range(24)]
    assert compressed[1].get("year") == ["2025"]
    assert compressed[1].get("month") == ["02"]
    assert compressed[1].get("day") == [str(i).zfill(2) for i in range(1, 29)]
    assert compressed[1].get("time") == [f"{hour:02d}:00" for hour in range(24)]


def test_compress_time_points_to_ymdt_one_group():
    full_tps = set(product(["2026"], ["01", "02"], ["01", "02"], ["00:00"]))

    compressed = utils.compress_time_points_to_ymdt(full_tps)

    assert len(compressed) == 1
    assert compressed[0].get("year") == ["2026"]
    assert compressed[0].get("month") == ["01", "02"]
    assert compressed[0].get("day") == ["01", "02"]
    assert compressed[0].get("time") == ["00:00"]
