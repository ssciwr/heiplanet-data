---
title: About
hide:
- navigation
---

# About

Heidelberg Planetary Health Hub ([Hei-Planet](https://hei-planet.com/))

Scientific Software Center ([SSC](https://www.ssc.uni-heidelberg.de/en)

## Package modules

The `heiplanet_data` package is organized by pipeline stage:

| Module | Responsibility |
|--------|----------------|
| [`inout`](reference/inout.md) | Download of raw data (bronze level) from data sources such as CDS/ERA5-Land and ISIMIP, and file handling. |
| [`converters`](reference/converters.md) | Elementary transformations: longitude 0-360 to -180-180, Kelvin to Celsius, meters to millimeters, unification of coordinate names. |
| [`regrid`](reference/regrid.md) | Spatial resolution resampling: downsampling via `xarray`, `xESMF`, or `CDO`, and upsampling via interpolation. Only module using the heavy `xesmf`/`cdo` dependencies. |
| [`temporal`](reference/temporal.md) | Time-axis operations: shifting time points, truncating to a date range, monthly precipitation totals. |
| [`pipeline`](reference/pipeline.md) | Settings-driven orchestration of the preprocessing steps (bronze to silver): step registry and the `preprocess_data_file` entry point. |
| [`nuts_aggregation`](reference/nuts_aggregation.md) | Aggregation of preprocessed gridded data over NUTS regions via `exactextract` or `geopandas`: the `aggregate_data_by_nuts` entry point. |
| [`utils`](reference/utils.md) | Shared helpers: settings loading and validation, file checks, unique tags. |
| [`preprocess`](reference/preprocess.md) | Backward-compatible facade re-exporting the API of the former monolithic `preprocess` module; import from the dedicated modules in new code. |)