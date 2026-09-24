import numpy as np
import pytest
import xarray as xr
from cdo import Cdo

from heiplanet_data import population


@pytest.fixture
def get_popu_dataset():
    # global 10 degree grid, cell centers, latitude descending as in ISIMIP data
    latitude = np.arange(85.0, -90.0, -10.0)
    longitude = np.arange(-175.0, 180.0, 10.0)
    time_points = np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, latitude.size, longitude.size)).astype(np.float32) * 1e5
    data[:, 0, :] = np.nan  # e.g. ocean or ice without data
    popu = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_points, "latitude": latitude, "longitude": longitude},
        attrs={
            "standard_name": "total population",
            "long_name": "total population",
            "units": "1",
        },
    )
    return xr.Dataset({"total-population": popu, "urban-population": popu / 2})


def test_calculate_grid_cell_area(get_popu_dataset):
    area = population.calculate_grid_cell_area(get_popu_dataset)

    assert area.dims == ("latitude", "longitude")
    assert area.shape == (18, 36)
    assert area.attrs["units"] == "km2"

    # cells cover the whole sphere
    sphere = 4 * np.pi * population.EARTH_RADIUS_KM**2
    assert np.isclose(area.sum().item(), sphere)

    # area only depends on latitude and is symmetric around the equator
    assert np.allclose(area.values, area.values[:, :1])
    assert np.allclose(area.values[:, 0], area.values[::-1, 0])
    assert area.values[9, 0] > area.values[0, 0]


def test_calculate_grid_cell_area_matches_cdo(tmp_path):
    latitude = np.arange(89.75, -90.0, -0.5)
    longitude = np.arange(-179.75, 180.0, 0.5)
    dataset = xr.Dataset(
        {"total-population": (("lat", "lon"), np.ones((360, 720), np.float32))},
        coords={
            "lat": ("lat", latitude, {"units": "degrees_north"}),
            "lon": ("lon", longitude, {"units": "degrees_east"}),
        },
    )
    nc_file = tmp_path / "grid.nc"
    dataset.to_netcdf(nc_file)

    cdo_area = Cdo().gridarea(input=str(nc_file), returnXDataset=True)["cell_area"]
    area = population.calculate_grid_cell_area(dataset, lat_name="lat", lon_name="lon")

    assert np.allclose(area.values, cdo_area.values / 1e6, rtol=1e-4)


def test_calculate_grid_cell_area_invalid(get_popu_dataset):
    with pytest.raises(ValueError):
        population.calculate_grid_cell_area(get_popu_dataset, lat_name="lat")

    single_lon = get_popu_dataset.isel(longitude=[0])
    with pytest.raises(ValueError):
        population.calculate_grid_cell_area(single_lon)

    irregular = get_popu_dataset.isel(latitude=[0, 1, 3])
    with pytest.raises(ValueError):
        population.calculate_grid_cell_area(irregular)


def test_load_grid_cell_area(tmp_path, get_popu_dataset):
    area = population.calculate_grid_cell_area(get_popu_dataset)
    area_file = tmp_path / "area.nc"
    area.rename({"latitude": "lat", "longitude": "lon"}).to_netcdf(area_file)

    loaded = population.load_grid_cell_area(area_file, get_popu_dataset)
    assert loaded.dims == ("latitude", "longitude")
    assert np.allclose(loaded.values, area.values)

    # areas in m2 are converted to km2
    area_m2 = area * 1e6
    area_m2.attrs["units"] = "m2"
    area_m2.to_netcdf(area_file)
    loaded = population.load_grid_cell_area(area_file, get_popu_dataset)
    assert np.allclose(loaded.values, area.values)
    assert loaded.attrs["units"] == "km2"


def test_load_grid_cell_area_invalid(tmp_path, get_popu_dataset):
    area = population.calculate_grid_cell_area(get_popu_dataset)
    area_file = tmp_path / "area.nc"
    area.to_netcdf(area_file)

    with pytest.raises(ValueError):
        population.load_grid_cell_area(area_file, get_popu_dataset, var_name="area")

    # grid does not match
    mismatched_dataset = get_popu_dataset.isel(latitude=slice(0, 5)).assign_coords(
        latitude=np.arange(5.0) + 0.3
    )
    with pytest.raises(ValueError):
        population.load_grid_cell_area(area_file, mismatched_dataset)

    bad_units = area.copy()
    bad_units.attrs["units"] = "ha"
    bad_units.to_netcdf(area_file)
    with pytest.raises(ValueError):
        population.load_grid_cell_area(area_file, get_popu_dataset)


def test_calculate_population_density(get_popu_dataset):
    area = population.calculate_grid_cell_area(get_popu_dataset)
    dataset = population.calculate_population_density(
        get_popu_dataset, ["total-population", "urban-population"], area
    )

    density = dataset["total-population-density"]
    assert "total-population" in dataset.data_vars
    assert "urban-population-density" in dataset.data_vars
    assert density.dims == ("time", "latitude", "longitude")
    assert density.dtype == np.float32
    assert density.attrs["units"] == "km-2"
    assert density.attrs["long_name"] == "total population density"

    expected = get_popu_dataset["total-population"] / area
    assert np.allclose(density.values, expected.values, equal_nan=True, rtol=1e-6)
    # cells without data stay NaN
    assert density.isel(latitude=0).isnull().all()


def test_calculate_population_density_integer_counts(get_popu_dataset):
    counts = xr.full_like(
        get_popu_dataset["total-population"].fillna(0), 1, dtype=np.int32
    )
    dataset = xr.Dataset({"total-population": counts})
    area = population.calculate_grid_cell_area(dataset)
    dataset = population.calculate_population_density(
        dataset, ["total-population"], area
    )

    density = dataset["total-population-density"]
    # densities below 1 person per km2 are not truncated to 0
    assert np.issubdtype(density.dtype, np.floating)
    assert (density > 0).all()
    assert np.allclose(density.values, 1 / area.values)


def test_calculate_population_density_invalid(get_popu_dataset):
    area = population.calculate_grid_cell_area(get_popu_dataset)
    with pytest.raises(ValueError):
        population.calculate_population_density(get_popu_dataset, ["popu"], area)
