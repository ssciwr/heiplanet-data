"""heiplanet_data public API.

Building blocks (import directly for custom, notebook-style pipelines):

* :mod:`heiplanet_data.converters` — unit and coordinate conversions
* :mod:`heiplanet_data.regrid` — spatial resolution resampling
* :mod:`heiplanet_data.temporal` — time-axis operations

Orchestrated entry points (settings-driven, used by the CLI scripts):

* `preprocess_data_file` — run the registered preprocessing pipeline over a
  NetCDF file (see :mod:`heiplanet_data.pipeline`)
* `aggregate_data_by_nuts` — aggregate NetCDF data over NUTS regions (see
  :mod:`heiplanet_data.nuts_aggregation`)
"""

# Export the version defined in project metadata
try:
    from importlib.metadata import version

    __version__ = version("heiplanet-data")
except ImportError:
    __version__ = "unknown"

from heiplanet_data.converters import (
    T,
    adjust_longitude_360_to_180,
    convert_360_to_180,
    convert_m_to_mm,
    convert_m_to_mm_with_attributes,
    convert_to_celsius,
    convert_to_celsius_with_attributes,
    rename_coords,
)
from heiplanet_data.nuts_aggregation import (
    CRS,
    aggregate_data_by_nuts,
)
from heiplanet_data.pipeline import (
    StepFn,
    preprocess_data_file,
    register_step,
)
from heiplanet_data.regrid import (
    GridConfig,
    ResolutionConfig,
    align_lon_lat_with_popu_data,
    check_agg_funcs,
    check_downsample_condition,
    downsample_resolution_with_cdo,
    downsample_resolution_with_xarray,
    downsample_resolution_with_xesmf,
    resample_resolution,
    upsample_resolution,
    warn_positive_resolution,
)
from heiplanet_data.temporal import (
    calculate_monthly_precipitation,
    shift_time,
    truncate_data_by_time,
)

__all__ = [
    # nuts_aggregation
    "CRS",
    # regrid
    "GridConfig",
    "ResolutionConfig",
    # pipeline
    "StepFn",
    # converters
    "T",
    "__version__",
    "adjust_longitude_360_to_180",
    "aggregate_data_by_nuts",
    "align_lon_lat_with_popu_data",
    # temporal
    "calculate_monthly_precipitation",
    "check_agg_funcs",
    "check_downsample_condition",
    "convert_360_to_180",
    "convert_m_to_mm",
    "convert_m_to_mm_with_attributes",
    "convert_to_celsius",
    "convert_to_celsius_with_attributes",
    "downsample_resolution_with_cdo",
    "downsample_resolution_with_xarray",
    "downsample_resolution_with_xesmf",
    "preprocess_data_file",
    "register_step",
    "rename_coords",
    "resample_resolution",
    "shift_time",
    "truncate_data_by_time",
    "upsample_resolution",
    "warn_positive_resolution",
]
