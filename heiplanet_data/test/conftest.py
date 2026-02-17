from pathlib import Path
import pytest
from tinydb import TinyDB
from heiplanet_data import data_lake
import hashlib


def get_files(dir_path: Path, name_phrase: str) -> list[Path]:
    """
    Get all files in a directory that contain the name_phrase in their name.
    """
    return [
        file
        for file in dir_path.iterdir()
        if file.is_file() and name_phrase in file.name
    ]


# data lake test fixtures
@pytest.fixture
def get_empty_db_query(tmp_path):
    # create a sample data file
    db_fpath = tmp_path / "test_db.json"
    db = TinyDB(db_fpath)
    query = data_lake.Query()

    yield db, query

    db.close()


@pytest.fixture
def get_mock_data():
    ds_source = "era5-land"

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["t2m", "tp"],
        "year": ["2016", "2017"],
        "month": [
            "01",
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    signatures = [
        {
            "ds_name": "era5-land",
            "product_type": ["monthly_averaged_reanalysis"],
            "data_var": "t2m",
        },
        {
            "ds_name": "era5-land",
            "product_type": ["monthly_averaged_reanalysis"],
            "data_var": "tp",
        },
    ]

    signature_strs = [
        "data_var-t2m|ds_name-era5-land|product_type-monthly_averaged_reanalysis",
        "data_var-tp|ds_name-era5-land|product_type-monthly_averaged_reanalysis",
    ]

    return ds_source, request, signatures, signature_strs


@pytest.fixture
def get_mock_download_info():
    downloaded_fpath = "test.nc"
    downloaded_at = "2026-02-13"
    status = "active"
    return downloaded_fpath, downloaded_at, status


@pytest.fixture
def get_mock_items(get_mock_data, get_mock_download_info):
    _, request, signatures, signature_strs = get_mock_data
    downloaded_fpath, downloaded_at, status = get_mock_download_info
    items = []
    for signature, signature_str in zip(signatures, signature_strs):
        hash_value = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()
        item = {
            "hash": hash_value,
            "signature": signature,
            "year": request.get("year", []),
            "month": request.get("month", []),
            "day": request.get("day", []),
            "time": request.get("time", []),
            "file_path": downloaded_fpath,
            "data_format": request.get("data_format", ""),
            "download_format": request.get("download_format", ""),
            "downloaded_at": downloaded_at,
            "status": status,
        }
        items.append(item)
    return items


@pytest.fixture
def get_db_with_mock_data(get_empty_db_query, get_mock_items):
    db, _ = get_empty_db_query

    db.insert_multiple(get_mock_items)

    yield db

    db.close()
