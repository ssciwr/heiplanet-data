import time
from pathlib import Path
from typing import Any

from heiplanet_data import preprocess_data_file, utils
from heiplanet_data.inout import (
    download_data,
    download_isimip_data,
    find_isimip_file,
    suggest_filename,
)


def main(config: dict[str, Any] | None = None) -> None:
    """Download ERA5-Land and ISIMIP population data, and preprocess the
    ERA5-Land data.

    Args:
        config (Dict[str, Any] | None): User-configurable variables, in the
            shape of `script_config.json`. Defaults to None, which loads the
            packaged `script_config.json` (see `utils.load_script_config`).
    """
    if config is None:
        config = utils.load_script_config()

    data_format = config["data_format"]
    data_folder = Path(config["data_folder"])
    data_folder_out = Path(config["data_folder_out"])

    dataset = config["dataset"]
    request = config["request"]

    file_name = suggest_filename(
        ds_name=dataset,
        data_format=data_format,
        years=request["year"],
        months=request["month"],
        has_area=False,
        base_name=config["base_name"],
        variables=request["variable"],
    )
    output_file = data_folder / file_name

    if not output_file.exists():
        print("Downloading data...")
        download_data(output_file, dataset, request)
    else:
        print(f"Data already exists at {output_file}")

    print(f"Preprocessing ERA5-Land data: {output_file}")
    t0 = time.time()
    _, era5_pfname = preprocess_data_file(
        netcdf_file=output_file,
        source=config["source"],
        settings=config["preprocess_settings"],
        new_settings={"output_dir": str(data_folder_out)},
        unique_tag=config["unique_tag"],
    )
    t_preprocess = time.time()
    print(f"Preprocessing completed in {t_preprocess - t0:.2f} seconds.")
    print(f"Name of preprocessed file: {era5_pfname}")

    # get the ISIMIP population data
    isimip_config = config["isimip"]
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


if __name__ == "__main__":
    main()
