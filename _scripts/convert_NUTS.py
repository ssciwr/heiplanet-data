import time
from pathlib import Path

import xarray as xr

from heiplanet_data import aggregate_data_by_nuts

# change to your own data folder, if needed
data_root = Path("./data/")
data_folder = data_root / "in"

# NUTS shapefile
nuts_file = data_folder / "NUTS_RG_20M_2024_4326.shp.zip"
# forming a dictionary for non-NUTS data
# key is name of the dataset, value is a tuple of (file path, aggregation mapping dict.)
non_nuts_data = {
    "jmodel": (
        data_root
        / "processed"
        / "output_JModel_global_02deg_trim_ts20250924-150129_hssc-laptop01.nc",
        None,
    ),
}

# aggregate data by NUTS regions
t0 = time.time()
aggregated_file = aggregate_data_by_nuts(
    non_nuts_data, nuts_file, normalize_time=True, output_dir=data_root / "processed"
)
t1 = time.time()
print(f"Aggregation completed in {t1 - t0:.2f} seconds")

# inspect the NUTS aggregated data of R0
agg_ds = xr.open_dataset(aggregated_file)
print(agg_ds)
