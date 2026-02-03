from pathlib import Path
from heiplanet_data.inout import (
    download_data_by_request,
    get_filename,
)
from heiplanet_data.preprocess import (
    preprocess_data_file,
)
from heiplanet_data import utils

if __name__ == "__main__":
    # get the era5 land data for 2016 and 2017
    data_format = "netcdf"
    data_folder = Path(".data_heiplanet_db/bronze/")
    data_folder_out = Path(".data_heiplanet_db/silver/")

    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature", "total_precipitation"],
        "year": ["2016", "2017"],
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "time": ["00:00"],
        "data_format": data_format,
        "download_format": "unarchived",
    }
    file_name = get_filename(
        dataset,
        data_format,
        request["year"],
        request["month"],
        "area" in request,
        "era5_data",
        request["variable"],
    )
    output_file = data_folder / file_name

    if not output_file.exists():
        print("Downloading data...")
        download_data_by_request(output_file, dataset, request)
    else:
        print("Data already exists at {}".format(output_file))

    settings = utils.get_settings(
        setting_path="default",
        new_settings={},
        updated_setting_dir=None,
        save_updated_settings=False,
    )

    # disable truncation of dates
    settings["truncate_date"] = False
    print("Preprocessing ERA5-Land data...")
    preprocessed_dataset = preprocess_data_file(
        netcdf_file=output_file,
        settings=settings,
    )
    # here we need to provide output folder
    # preprocess the population data
    popu_file = data_folder / "population_histsoc_30arcmin_annual_1901_2021_renamed.nc"
    settings["truncate_date"] = True
    # disable uncessary preprocessing steps
    settings["adjust_longitude"] = False
    settings["convert_kelvin_to_celsius"] = False
    settings["convert_m_to_mm_precipitation"] = False
    settings["resample_grid"] = False

    print("Preprocessing population data...")
    preprocessed_popu = preprocess_data_file(
        netcdf_file=popu_file,
        settings=settings,
    )
