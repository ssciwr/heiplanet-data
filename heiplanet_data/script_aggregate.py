import time
from pathlib import Path

import pooch

from heiplanet_data import aggregate_data_by_nuts

# change to your own data folder, if needed
DATA_FOLDER = Path("../heiplanet-db/.data_heiplanet_db/gold")
NUTS_FILE = DATA_FOLDER / "NUTS_RG_20M_2024_4326.shp.zip"


def get_nuts_data():
    # download the NUTS shapefile
    filename = "NUTS_RG_20M_2024_4326.shp.zip"
    url = "https://heibox.uni-heidelberg.de/f/e95125307161470a8906/?dl=1"
    filehash = "246a5e1a3901380f194879cff5a65bbac86c9c906a7871027a3a918dbe6c7e46"

    try:
        file = pooch.retrieve(
            url=url, known_hash=filehash, fname=filename, path=DATA_FOLDER
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise RuntimeError(f"Failed to fetch data from {url}") from e
    print(f"Data fetched and saved to {file}")


if __name__ == "__main__":
    # specify which files will be aggregated by NUTS regions
    # jmodel_fname = "era5_data_2025_07_2t_monthly_unicoords_adjlon_celsius_silver_output_JModel_global_ts20260206-161638_hssclap09.nc"
    jmodel_fname = "era5_data_2025_05-07_2t_monthly_unicoords_adjlon_celsius_silver_output_JModel_global_ts20260206-161821_hssclap09.nc"
    non_nuts_data = {
        "jmodel": (DATA_FOLDER / jmodel_fname, None),
    }

    # aggregate data by NUTS regions
    t0 = time.time()
    aggregated_file = aggregate_data_by_nuts(
        non_nuts_data, NUTS_FILE, normalize_time=True, output_dir=DATA_FOLDER
    )
    t1 = time.time()
    print(f"Aggregation completed in {t1 - t0:.2f} seconds")
