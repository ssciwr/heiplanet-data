---
title: Data Lake Backend
hide:
- navigation
---

# Data Lake Backend

## Data flowchart
Downloaded data files will go through preprocessing steps as follows:

![heiplanet_data_flow.jpg](source/_static/heiplanet_data_flow.jpg)

## Data Lake architecture

The data lake is separated into bronze (raw data), silver (preprocessed data), and gold (processed data). The data preprocessing package deals with levels bronze and silver. The data is stored in `.data_heiplanet_db/bronze` and `.data_heiplanet_db/silver`.

## Production run

The package defaults are meant for tutorials and integration tests. The settings for the production data are kept separately in `heiplanet_data/production/`:

* `script_config.json`: the datasets to download and the data folders,
* `era5_settings.json`: preprocessing settings for ERA5-Land, monthly and daily,
* `isimip_settings.json`: preprocessing settings for the ISIMIP population data.

To run the download and preprocessing, set up the [CDS API](data.md#set-up-cds-api), activate your environment, and call the script with the production config from the `heiplanet_data` folder:

```bash
cd heiplanet_data
python script.py production/script_config.json
```

Without the argument, the script uses the default `script_config.json`. Please note that the data folders in the config are relative to the working directory, so running the script from a different folder downloads all data again to a new `.data_heiplanet_db` folder. Paths to settings files in the config are relative to the config file.

The production settings differ from the defaults as follows:

| | Default | Production |
|---|---|---|
| ERA5-Land monthly | disabled | enabled |
| ERA5-Land years | 2024 | 2014 to 2018 |
| ERA5-Land grid | resampled to 0.5° | native 0.1° |
| ISIMIP years | 2020 to 2024 | 2014 to 2018 |
| Population density grid | ISIMIP 0.5° grid | ERA5-Land 0.1° grid |

The population density must be on exactly the same grid as ERA5-Land for the subsequent processing in `heiplanet-models`. For this, the `regrid_pop_density` step reads the target grid from the monthly ERA5-Land file in the silver folder. Therefore:

* ERA5-Land has to be processed before the ISIMIP data, which is the order in the script. If the file is missing, preprocessing of the ISIMIP data stops with an error.
* The name of that file depends on the ERA5-Land request. If the request changes, `regrid_pop_density_grid_file` in `production/isimip_settings.json` has to be updated.
* The population file only contains the density, not the population counts.

See [Tutorial B](source/notebooks/tutorial_B_preprocess_data.ipynb) for details on the population density settings.
