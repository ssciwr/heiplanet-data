import time
from pathlib import Path
from heiplanet_data.inout import (
    download_data,
    get_filename,
)
from heiplanet_data import preprocess

if __name__ == "__main__":
    # get the era5 land data for 2016 and 2017
    data_format = "netcdf"
    data_folder = Path(".data_heiplanet_db/bronze/")
    data_folder_out = Path(".data_heiplanet_db/silver/")

    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2025"],
        "month": [
            "05",
            "06",
            "07",
        ],
        "time": ["00:00"],
        "data_format": data_format,
        "download_format": "unarchived",
    }
    file_name = get_filename(
        ds_name=dataset,
        data_format=data_format,
        years=request["year"],
        months=request["month"],
        has_area=False,
        base_name="era5_data",
        variables=request["variable"],
    )
    output_file = data_folder / file_name

    if not output_file.exists():
        print("Downloading data...")
        download_data(output_file, dataset, request)
    else:
        print("Data already exists at {}".format(output_file))

    print(f"Preprocessing ERA5-Land data: {output_file}")
    t0 = time.time()
    preprocessed_dataset, era5_pfname = preprocess.preprocess_data_file(
        netcdf_file=output_file,
        source="era5",
        settings="default",
        new_settings={"output_dir": str(data_folder_out)},
        unique_tag="silver",
    )
    t_preprocess = time.time()
    print(f"Preprocessing completed in {t_preprocess - t0:.2f} seconds.")
    print(f"Name of preprocessed file: {era5_pfname}")
