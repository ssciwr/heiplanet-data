from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon


def get_files(dir_path: Path, name_phrase: str) -> list[Path]:
    """
    Get all files in a directory that contain the name_phrase in their name.
    """
    return [
        file
        for file in dir_path.iterdir()
        if file.is_file() and name_phrase in file.name
    ]


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
