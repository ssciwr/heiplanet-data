"""Unit tests for the NUTS aggregation step of `heiplanet_data.script`.

Unlike `test_script.py`, these do not download anything: the NUTS shape file
is written locally and `pooch.retrieve` is patched to return it.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from heiplanet_data import script, utils

TIMES = np.array(["2016-01-01", "2016-02-01"], dtype="datetime64[ns]")
LAT = np.array([1.5, 1.0, 0.5, 0.0])  # descending, like the gold output
LON = np.array([0.0, 0.5, 1.0, 1.5])


def _write_gold(path: Path, variables: list[str], seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            v: (("time", "latitude", "longitude"), rng.random((2, 4, 4)))
            for v in variables
        },
        coords={"time": TIMES, "latitude": LAT, "longitude": LON},
    )
    ds.to_netcdf(path)
    return ds


@pytest.fixture
def nuts_setup(tmp_path, monkeypatch):
    gold = tmp_path / "gold"
    gold.mkdir()
    jmodel = _write_gold(gold / "jmodel.nc", ["R0"], seed=1)
    pmodel = _write_gold(
        gold / "pmodel.nc", ["eggs", "ed", "juv", "imm", "adults"], seed=2
    )

    nuts_file = tmp_path / "nuts.shp"
    gpd.GeoDataFrame(
        {"NUTS_ID": ["A", "B"]},
        geometry=[box(-0.25, -0.25, 0.75, 1.75), box(0.75, -0.25, 1.75, 1.75)],
        crs="EPSG:4326",
    ).to_file(nuts_file)

    retrieved = {}

    def fake_retrieve(url, known_hash, fname, path):
        retrieved.update(url=url, known_hash=known_hash, fname=fname, path=path)
        return str(nuts_file)

    monkeypatch.setattr(script.pooch, "retrieve", fake_retrieve)

    config = {
        "data_format": "netcdf",
        "data_folder": str(tmp_path / "bronze"),
        "data_folder_out": str(tmp_path / "silver"),
        "datasets": {},
        "nuts_aggregation": {
            "enabled": True,
            "data_folder": str(gold),
            "data_folder_out": str(tmp_path / "out"),
            "nuts_file": "nuts.shp",
            "nuts_url": "https://example.org/nuts.zip",
            "nuts_hash": "abc",
            "datasets": {
                "jmodel": {"file": "jmodel.nc", "agg": {"R0": "mean"}},
                "pmodel": {"file": "pmodel.nc", "agg": {"adults": "mean"}},
            },
        },
    }
    return config, jmodel, pmodel, retrieved


def test_run_nuts_aggregation(nuts_setup, tmp_path):
    config, jmodel, pmodel, retrieved = nuts_setup

    out_file = script._run_nuts_aggregation(
        config["nuts_aggregation"], Path(config["data_folder"])
    )

    # the NUTS file is fetched into the bronze folder with url and hash from config
    assert retrieved == {
        "url": "https://example.org/nuts.zip",
        "known_hash": "abc",
        "fname": "nuts.shp",
        "path": Path(config["data_folder"]),
    }

    assert out_file.parent == tmp_path / "out"
    with xr.open_dataset(out_file) as out:
        # only the variables listed in `agg` are aggregated
        assert set(out.data_vars) == {"R0", "adults"}
        assert set(out["NUTS_ID"].values) == {"A", "B"}
        assert out.sizes["time"] == 2

        # region A covers the two western columns completely
        expected = jmodel["R0"].isel(time=0, longitude=[0, 1]).mean().item()
        assert np.isclose(out["R0"].sel(NUTS_ID="A").isel(time=0).item(), expected)
        expected = pmodel["adults"].isel(time=1, longitude=[2, 3]).mean().item()
        assert np.isclose(out["adults"].sel(NUTS_ID="B").isel(time=1).item(), expected)


def test_run_nuts_aggregation_missing_input(nuts_setup):
    config, *_ = nuts_setup
    config["nuts_aggregation"]["datasets"]["pmodel"]["file"] = "missing.nc"

    with pytest.raises(FileNotFoundError, match="missing.nc"):
        script._run_nuts_aggregation(
            config["nuts_aggregation"], Path(config["data_folder"])
        )


def test_main_only_nuts_step(nuts_setup, monkeypatch, tmp_path):
    config, *_ = nuts_setup
    config["datasets"] = {"era5": {"enabled": True}, "isimip": {"enabled": True}}

    def fail(*args, **kwargs):
        raise AssertionError("step should have been skipped")

    monkeypatch.setattr(script, "_run_era5", fail)
    monkeypatch.setattr(script, "_run_isimip", fail)

    script.main(config, steps=["nuts_aggregation"])

    assert len(list((tmp_path / "out").glob("*.nc"))) == 1


def test_main_nuts_disabled(nuts_setup, monkeypatch):
    config, *_ = nuts_setup
    config["nuts_aggregation"]["enabled"] = False
    monkeypatch.setattr(
        script, "_run_nuts_aggregation", lambda *a: pytest.fail("should not run")
    )

    script.main(config)


def test_main_unknown_step():
    with pytest.raises(ValueError, match="Unknown steps"):
        script.main({"datasets": {}}, steps=["nuts"])


def test_parse_args():
    args = script._parse_args(["cfg.json", "--steps", "nuts_aggregation"])
    assert args.config == "cfg.json"
    assert args.steps == ["nuts_aggregation"]

    args = script._parse_args([])
    assert args.config == "default"
    assert args.steps is None

    with pytest.raises(SystemExit):
        script._parse_args(["--steps", "nuts"])


@pytest.mark.parametrize(
    ("config_path", "enabled"),
    [
        ("default", False),
        (Path(script.__file__).parent / "production/script_config.json", True),
    ],
)
def test_packaged_configs_nuts_block(config_path, enabled):
    nuts = utils.load_script_config(config_path)["nuts_aggregation"]
    assert nuts["enabled"] is enabled
    assert nuts["datasets"]["jmodel"]["agg"] == {"R0": "mean"}
    assert nuts["datasets"]["pmodel"]["agg"] == {"adults": "mean"}
    assert (
        nuts["datasets"]["pmodel"]["file"] == "Mosquito_abundance_Global_2014-2018.nc"
    )
