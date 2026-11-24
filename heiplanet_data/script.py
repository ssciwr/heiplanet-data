import argparse
import time
from pathlib import Path
from typing import Any

import pooch

from heiplanet_data import aggregate_data_by_nuts, preprocess_data_file, utils
from heiplanet_data.inout import (
    download_data,
    download_isimip_data,
    download_total_precipitation_from_hourly_era5_land,
    find_isimip_file,
    suggest_filename,
)


def _run_era5(
    era5_config: dict[str, Any],
    data_format: str,
    data_folder: Path,
    data_folder_out: Path,
) -> None:
    """Download and preprocess the ERA5-Land dataset.

    Args:
        era5_config (Dict[str, Any]): The `datasets.era5` block of the script
            config.
        data_format (str): Data format shared across datasets (e.g. "netcdf").
        data_folder (Path): Folder raw data is downloaded to.
        data_folder_out (Path): Folder preprocessed data is written to.
    """
    dataset = era5_config["dataset"]
    request = era5_config["request"]

    file_name = suggest_filename(
        ds_name=dataset,
        data_format=data_format,
        years=request["year"],
        months=request["month"],
        has_area=False,
        base_name=era5_config["base_name"],
        variables=request["variable"],
    )
    output_file = data_folder / file_name

    if not output_file.exists():
        print("Downloading ERA5-Land data...")
        download_data(output_file, dataset, request)
    else:
        print(f"Data already exists at {output_file}")

    print(f"Preprocessing ERA5-Land data: {output_file}")
    t0 = time.time()
    _, era5_pfname = preprocess_data_file(
        netcdf_file=output_file,
        source=era5_config["source"],
        settings=era5_config["preprocess_settings"],
        new_settings={"output_dir": str(data_folder_out)},
        unique_tag=era5_config["unique_tag"],
    )
    t_preprocess = time.time()
    print(f"Preprocessing completed in {t_preprocess - t0:.2f} seconds.")
    print(f"Name of preprocessed file: {era5_pfname}")


def _run_era5_daily(
    era5_daily_config: dict[str, Any],
    data_format: str,
    data_folder: Path,
    data_folder_out: Path,
) -> None:
    """Download and preprocess the daily ERA5-Land total precipitation dataset.

    Unlike `_run_era5`, this pulls from the hourly `reanalysis-era5-land`
    dataset via `download_total_precipitation_from_hourly_era5_land`, which
    takes a `start_date`/`end_date` range rather than year/month lists (the
    monthly-means product used by `era5` has no daily equivalent on CDS).

    Args:
        era5_daily_config (Dict[str, Any]): The `datasets.era5_daily` block
            of the script config.
        data_format (str): Data format shared across datasets (e.g. "netcdf").
        data_folder (Path): Folder raw data is downloaded to.
        data_folder_out (Path): Folder preprocessed data is written to.
    """
    output_file = Path(
        download_total_precipitation_from_hourly_era5_land(
            start_date=era5_daily_config["start_date"],
            end_date=era5_daily_config["end_date"],
            area=era5_daily_config.get("area"),
            out_dir=data_folder,
            base_name=era5_daily_config["base_name"],
            data_format=data_format,
            ds_name=era5_daily_config["dataset"],
            var_name=era5_daily_config.get("variable", "total_precipitation"),
            # the per-range download chunks are merged into the raw file,
            # keeping them doubles the disk usage
            clean_tmp_files=era5_daily_config.get("clean_tmp_files", True),
        )
    )

    print(f"Preprocessing ERA5-Land daily data: {output_file}")
    t0 = time.time()
    _, era5_daily_pfname = preprocess_data_file(
        netcdf_file=output_file,
        source=era5_daily_config["source"],
        settings=era5_daily_config["preprocess_settings"],
        # the default era5 settings' cal_monthly_tp step scales tp by
        # days-in-month to turn a monthly *mean* into a monthly *total* -
        # daily data is already a daily total, and its dates aren't month
        # starts, so that step must stay off here.
        new_settings={
            "output_dir": str(data_folder_out),
            "cal_monthly_tp": False,
        },
        unique_tag=era5_daily_config["unique_tag"],
    )
    t_preprocess = time.time()
    print(f"Preprocessing completed in {t_preprocess - t0:.2f} seconds.")
    print(f"Name of preprocessed file: {era5_daily_pfname}")


def _run_isimip(
    isimip_config: dict[str, Any], data_folder: Path, data_folder_out: Path
) -> None:
    """Download and preprocess the ISIMIP population dataset.

    Args:
        isimip_config (Dict[str, Any]): The `datasets.isimip` block of the
            script config.
        data_folder (Path): Folder raw data is downloaded to.
        data_folder_out (Path): Folder preprocessed data is written to.
    """
    isimip_name, isimip_url = find_isimip_file(
        search_path=isimip_config["search_path"],
        file_match=isimip_config["file_match"],
    )
    isimip_file = data_folder / isimip_name

    if not isimip_file.exists():
        print("Downloading ISIMIP population data...")
        download_isimip_data(isimip_file, isimip_url)
    else:
        print(f"Data already exists at {isimip_file}")

    print(f"Preprocessing ISIMIP population data: {isimip_file}")
    t0 = time.time()
    _, isimip_pfname = preprocess_data_file(
        netcdf_file=isimip_file,
        source=isimip_config["source"],
        settings=isimip_config["preprocess_settings"],
        new_settings={"output_dir": str(data_folder_out)},
        unique_tag=isimip_config["unique_tag"],
    )
    t_preprocess = time.time()
    print(f"Preprocessing completed in {t_preprocess - t0:.2f} seconds.")
    print(f"Name of preprocessed file: {isimip_pfname}")


def _run_nuts_aggregation(nuts_config: dict[str, Any], data_folder: Path) -> Path:
    """Aggregate gold model output (e.g. J-Model, P-Model) over NUTS regions.

    The NUTS shape file is downloaded to `data_folder` with `pooch`, unless it
    is already there with the expected hash. All datasets are aggregated into
    a single NetCDF file indexed by "NUTS_ID" and "time".

    Args:
        nuts_config (Dict[str, Any]): The `nuts_aggregation` block of the
            script config, with keys `data_folder` (folder of the gold model
            output), `data_folder_out`, `nuts_file`, `nuts_url`, `nuts_hash`,
            optionally `normalize_time`, and `datasets`: a mapping of dataset
            name to `{"file": <file name in data_folder>, "agg": {<var>: <func>}}`.
            Only the variables listed in `agg` are aggregated; `agg: null`
            aggregates all variables with the mean.
        data_folder (Path): Folder raw data, here the NUTS shape file, is
            downloaded to.

    Returns:
        Path: Path to the aggregated NetCDF file.
    """
    nuts_file = Path(
        pooch.retrieve(
            url=nuts_config["nuts_url"],
            known_hash=nuts_config["nuts_hash"],
            fname=nuts_config["nuts_file"],
            path=data_folder,
        )
    )

    gold_folder = Path(nuts_config["data_folder"])
    netcdf_files = {
        name: (gold_folder / ds["file"], ds.get("agg"))
        for name, ds in nuts_config["datasets"].items()
    }
    missing = [str(f) for f, _ in netcdf_files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Input files for NUTS aggregation not found: {missing}. "
            f"These are produced by heiplanet-models; check `data_folder` "
            f"in the `nuts_aggregation` config (relative to the working directory)."
        )

    data_folder_out = Path(nuts_config["data_folder_out"])
    data_folder_out.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {list(netcdf_files)} over NUTS regions from {nuts_file}")
    t0 = time.time()
    out_file = aggregate_data_by_nuts(
        netcdf_files,
        nuts_file,
        normalize_time=nuts_config.get("normalize_time", True),
        output_dir=data_folder_out,
    )
    print(f"NUTS aggregation completed in {time.time() - t0:.2f} seconds.")
    print(f"Name of aggregated file: {out_file}")
    return out_file


STEPS = ("era5", "era5_daily", "isimip", "nuts_aggregation")


def main(config: dict[str, Any] | None = None, steps: list[str] | None = None) -> None:
    """Download and preprocess the datasets enabled in `config["datasets"]`
    (ERA5-Land monthly and daily, and/or ISIMIP population data), and
    aggregate gold model output over NUTS regions if `config["nuts_aggregation"]`
    is enabled.

    Each entry under `config["datasets"]` (currently "era5", "era5_daily"
    and "isimip") shares a common shape: `enabled`, `source`, `base_name`,
    `preprocess_settings`, `unique_tag`, plus dataset-specific download
    parameters (`dataset`/`request` for era5, `dataset`/`start_date`/
    `end_date` for era5_daily, `search_path`/`file_match` for isimip). Set
    `enabled` to `false` (or omit the entry) to skip a dataset.

    The NUTS aggregation runs last, as it needs the gold output of
    heiplanet-models, which in turn needs the silver data from the steps
    before. Use `steps` to run only some steps, e.g. only
    `["nuts_aggregation"]` once the models have run.

    Args:
        config (Dict[str, Any] | None): User-configurable variables, in the
            shape of `script_config.json`. Defaults to None, which loads the
            packaged `script_config.json` (see `utils.load_script_config`).
        steps (list[str] | None): Steps to run, from `STEPS`. Steps not listed
            are skipped even if enabled in the config. Defaults to None, which
            runs all steps enabled in the config.
    """
    if config is None:
        config = utils.load_script_config()

    if steps is not None:
        unknown = set(steps) - set(STEPS)
        if unknown:
            raise ValueError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}.")

    def _selected(name: str, step_config: dict[str, Any] | None) -> bool:
        if steps is not None and name not in steps:
            return False
        return bool(step_config) and step_config.get("enabled", True)

    data_format = config["data_format"]
    data_folder = Path(config["data_folder"])
    data_folder_out = Path(config["data_folder_out"])

    datasets = config["datasets"]

    era5_config = datasets.get("era5")
    if _selected("era5", era5_config):
        _run_era5(era5_config, data_format, data_folder, data_folder_out)
    else:
        print("Skipping ERA5-Land dataset (disabled in config).")

    era5_daily_config = datasets.get("era5_daily")
    if _selected("era5_daily", era5_daily_config):
        _run_era5_daily(era5_daily_config, data_format, data_folder, data_folder_out)
    else:
        print("Skipping ERA5-Land daily dataset (disabled in config).")

    isimip_config = datasets.get("isimip")
    if _selected("isimip", isimip_config):
        _run_isimip(isimip_config, data_folder, data_folder_out)
    else:
        print("Skipping ISIMIP dataset (disabled in config).")

    nuts_config = config.get("nuts_aggregation")
    if _selected("nuts_aggregation", nuts_config):
        _run_nuts_aggregation(nuts_config, data_folder)
    else:
        print("Skipping NUTS aggregation (disabled in config).")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess data, and aggregate model output "
        "over NUTS regions."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="default",
        help="Path to the script config; the packaged default config if omitted.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        help="Run only these steps (default: all steps enabled in the config).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # e.g. `python script.py production/script_config.json` for the production
    # run, or `... --steps nuts_aggregation` once heiplanet-models has run;
    # without a config argument, the packaged default config is used
    args = _parse_args()
    main(utils.load_script_config(args.config), steps=args.steps)
