---
title: Data
hide:
- navigation
---

# Data

The supported models rely on data from the [Copernicus's CDS](https://cds.climate.copernicus.eu/), [Eurostat's NUTS definition](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010), and [ISIMIP's population data](https://data.isimip.org/).

## Copernicus Data

The [CDS's ERA5-Land monthly](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview) dataset is currently being used for now. You can either download the data directly from CDS website or use the provided Python script, [`inout module`](reference/inout.md).

For the latter option, please set up the CDS API as outlined below and take note of the naming convention used for the downloaded files.

### Set up CDS API
To use [CDS](https://cds.climate.copernicus.eu/) API for downloading data, you need to first create an account on CDS to obtain your personal access token.

Create a `.cdsapirc` file containing your personal access token by following [this instruction](https://cds.climate.copernicus.eu/how-to-api).

### Naming convention
The filenames of the downloaded netCDF files follow this structure:
```text linenums="0"
{base_name}_{year_str}_{month_str}_{day_str}_{time_str}_{var_str}_{ds_type}_{area_str}_raw.{ext}
```

* `base_name` is `"era5_data"`,
* For list of numbers, i.e. years/months/days/times, the rule below is applied
    * If the values are continuous, the string representation is a concatenate of `min` and `max` values, separated by `-`
    * Otherwise, the string is a join of all values, separated by `_`
    * However, if there are more than 5 values, we only keep the first 5 ones and replace the rest by `"_etc"`
    * If the values are empty (e.g. no days or times in the download request), their string representation and the corresponding separator (i.e. `"_"`) are omitted from the file name.
* `year_str` is the string representation of list of years using the rule above.
* Similarly for `month_str`. However, if the download requests all 12 months, `month_str` would be `"allm"`
* `day_str` and `time_str` follows the same pattern, assuming that a month has at most 31 days (`"alld"`) and a day has at most 24 hours (`"allt"`).
    * Special case: if data is downloaded at time `00:00` per day only, `time_str` would be `"midnight"` (e.g. precipitation data for P-model)
* For `var_str`, each variable has an abbreviation derived by the first letter of each word in the variable name (e.g. `tp` for `total precipitation`).
    * All abbreviations are then concatenated by `_`
    * If this concatenated string is longer than 30 characters, we only keep the first 2 characters and replace the the rest by `"_etc"`
* As for `ds_type`:
    * If the file was downloaded from a monthly dataset, `"monthly"` is set to `ds_type`. This means the data is recorded only on the first day of each month.
    * For other datasets, when data is downloaded only at midnight (`time_str` = `"midnight"`), the ds_type is `"daily"`, meaning one data record for one day of each month.
    * `ds_type` would be an empty string in other cases, i.e. multiple data records for each day of a month.
* For `area_str`, if the downloaded data is only for an area of the grid (instead of the whole map), `"area"` would represent for `area_str`.
* If the part before `"_raw"` is longer than 100 characters, only the first 100 characters are kept and the rest is replaced by `"_etc"`
* `"_raw"` is added at the end to indicate that the file is raw data
* Extension `ext` of the file can be `.nc` or `.grib`
* If any of these fields (from `year_str` to `area_str`) are missing from the download request, the corresponding string and the preceding `_` are removed from the file name.

#### Special case

As for total precipitation data downloaded from dataset `ERA5-Land hourly data from 1950 to present`, the file name is structured as:

```text linenums="0"
{base_name}_{start_date}-{end_date}_{time_str}_{var_str}_{ds_type}_{area_str}_raw.{ext}
```

In this case, `time_str` is `"midnight"` and `ds_type` is `"daily"`.


## Eurostat's NUTS definition 
The regions are set [here](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010) and corresponding shapefiles can be downloaded [here](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics).

For downloading, please choose:

* The latest year from NUTS year,
* File format: `SHP`,
* Geometry type: `Polygons (RG)`,
* Scale: `20M`
* CRS: `EPSG: 4326`

???+ note
    * After downloading the file, unzip it to access the root folder containing the NUTS data (e.g. folder named `NUTS_RG_20M_2024_4326.shp`)
        * Inside the unzipped folder, there are five different shapefiles, which are all required to display and extract the NUTS regions data.
        ```
        shape data folder
        |____.shp file: geometry data (e.g. polygons)
        |____.shx file: index for geometry data
        |____.dbf file: attribute data for each NUTS region (e.g NUTS name, NUTS ID)
        |____.prj file: information on CRS
        |____.cpg file: character encoding data
        ```
    * These NUTS definition files are for Europe only.
    * If a country does not have NUTS level $x \in [1,3]$, the corresponding data for these levels is excluded from the shapefiles.

#### `NUTS_ID` explanation:
* Structure of `NUTS_ID`: `<country><level>`
* `country`: 2 letters, representing name of a country, e.g. DE
* `level`: 0 to 3 letters or numbers, signifying the level of the NUTS region

## ISIMIP Data
To download population data, please perform the following steps:

* go to [ISIMIP website](https://data.isimip.org/)
* search `population` from the search bar
* choose simulation round `ISIMIP3a`
* click `Input Data` -> `Direct human forcing` -> `Population data` -> `histsoc`
* choose `population_histsoc_30arcmin_annual`
* download file `population_histsoc_30arcmin_annual_1901_2021.nc`
