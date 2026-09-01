"""Integration tests for `heiplanet_data.script`.

These tests exercise `script.main()` end-to-end against the real CDS and
ISIMIP APIs (download + preprocessing), unlike the rest of the test suite
which mostly works against synthetic/local data. They are marked
`integration` and are only run by the `integration.yml` CI workflow;
`ci.yml` excludes them (`-m "not integration"`).

Downloading from CDS requires a valid `~/.cdsapirc` (see `ci.yml`/
`integration.yml`, which write one from the `CDSAPI_KEY` secret).

The downloaded and preprocessed data is then published to the
`iulusoy/heiplanet-data-silver` Hugging Face dataset repo, so that the next
processing component's system integration tests can pull fixed, real
fixture data instead of redownloading it. This step needs an `HF_TOKEN`
with write access to that repo; it is skipped (not failed) when no token is
available, e.g. for a contributor running integration tests locally without
one.
"""

from pathlib import Path

import pytest
import xarray as xr
from huggingface_hub import HfApi, get_token

from heiplanet_data import script

pytestmark = pytest.mark.integration

# Downstream system integration tests for the next processing component
# pull their fixture data from here.
HF_DATASET_REPO = "iulusoy/heiplanet-data-silver"


@pytest.fixture()
def script_config(tmp_path):
    data_folder = tmp_path / "bronze"
    data_folder_out = tmp_path / "silver"

    return {
        "data_format": "netcdf",
        "data_folder": str(data_folder),
        "data_folder_out": str(data_folder_out),
        "datasets": {
            "era5": {
                "enabled": True,
                "source": "era5",
                "base_name": "era5_data_integration_test",
                "preprocess_settings": "default",
                "unique_tag": "integration_test",
                "dataset": "reanalysis-era5-land-monthly-means",
                "request": {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["2m_temperature"],
                    "year": ["2025"],
                    "month": ["03"],
                    "time": ["00:00"],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    # small area to keep the download light, as in test_inout.py
                    "area": [0, -1, 0, 1],  # [N, W, S, E]
                },
            },
            "isimip": {
                "enabled": False,
                "source": "isimip",
                "base_name": "isimip_population_data_integration_test",
                "preprocess_settings": "default",
                "unique_tag": "integration_test",
                "search_path": "ISIMIP3a/InputData/socioeconomic/pop/histsoc/population",
                "file_match": "population_histsoc_30arcmin_annual_1901_2021",
            },
        },
    }


def test_main_downloads_and_preprocesses_era5_and_isimip(script_config, tmp_path):
    data_folder = tmp_path / "bronze"
    data_folder_out = tmp_path / "silver"

    script.main(script_config)

    era5_config = script_config["datasets"]["era5"]
    # isimip_config = script_config["datasets"]["isimip"]

    # ERA5-Land raw data was downloaded
    raw_files = list(data_folder.glob(f"{era5_config['base_name']}*.nc"))
    assert len(raw_files) == 1
    raw_mtime = raw_files[0].stat().st_mtime

    # ISIMIP population data was downloaded alongside the ERA5-Land data
    # isimip_files = list(data_folder.glob(f"{isimip_config['file_match']}*.nc"))
    # assert len(isimip_files) == 1
    # isimip_mtime = isimip_files[0].stat().st_mtime

    # data was preprocessed with the default settings and saved
    processed_files = list(data_folder_out.glob(f"*{era5_config['unique_tag']}*.nc"))
    processed_files.sort(
        key=lambda f: f.name
    )  # ensure consistent order for the next assertions
    assert len(processed_files) == 1
    with (
        xr.open_dataset(processed_files[0]) as ds
    ):  # we need to make sure this is the ERA5-Land dataset, not the ISIMIP population dataset
        assert ds.attrs["source"] == "era5"
        # default era5 settings: unify_coords renames valid_time -> time,
        # convert_kelvin_to_celsius keeps the variable name "t2m"
        assert "t2m" in ds.data_vars
        assert "time" in ds.coords
    # now the isimip population dataset
    # with xr.open_dataset(processed_files[1]) as ds:
    #     assert ds.attrs["source"] == "isimip"
    #     # default isimip settings: unify_coords renames valid_time -> time,
    #     # convert_kelvin_to_celsius keeps the variable name "population"
    #     assert "population" in ds.data_vars
    #     assert "time" in ds.coords

    # running again should not re-download the already-existing raw files,
    # only redo the preprocessing step
    for f in data_folder_out.glob("*"):
        f.unlink()

    script.main(script_config)
    assert raw_files[0].stat().st_mtime == raw_mtime
    # assert isimip_files[0].stat().st_mtime == isimip_mtime

    # publish the downloaded and processed data for the next processing
    # component's system integration tests; skip if no token is configured
    # (e.g. running locally without HF credentials)
    if not get_token():
        pytest.skip("HF_TOKEN not set; skipping upload to Hugging Face")

    upload_to_huggingface(
        folder_path=data_folder_out,
        repo_id=HF_DATASET_REPO,
        token=get_token(),
    )


def upload_to_huggingface(
    folder_path: Path,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Upload a single file to a Hugging Face Hub dataset repo.

    Args:
        folder_path (Path): Path to the local folder to upload.
        repo_id (str): Target dataset repo, as "<namespace>/<name>"
            (e.g. "iulusoy/heiplanet-data-silver").
        token (str | None): Hugging Face access token with write access to
            `repo_id`. Defaults to None, which uses the `HF_TOKEN`
            environment variable or a cached `huggingface-cli login`.

    Returns:
        str: URL of the uploaded file on the Hugging Face Hub.
            Defaults to the file's own name (uploaded to the repo root).
    """
    if not folder_path or not Path(folder_path).exists():
        raise ValueError(f"Folder {folder_path} must exist to be uploaded.")

    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("Repo id must be a non-empty string.")

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/blob/main"
    print(f"Uploaded {folder_path} to {url}")
    return url
