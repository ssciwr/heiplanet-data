"""Backward-compatible facade for the preprocessing modules.

The implementation was split into dedicated modules; import from them
directly in new code:

* :mod:`heiplanet_data.converters` — unit and coordinate conversions
* :mod:`heiplanet_data.regrid` — spatial resolution resampling
* :mod:`heiplanet_data.temporal` — time-axis operations
* :mod:`heiplanet_data.pipeline` — step registry and orchestration
* :mod:`heiplanet_data.nuts_aggregation` — aggregation over NUTS regions

This module re-exports the previous public API so existing imports of
``heiplanet_data.preprocess`` keep working.
"""

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
from heiplanet_data.regrid import (
    GridConfig,
    ResolutionConfig,
    _replace_decimal_point,
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
    _check_month_start_data,
    _parse_date,
    calculate_monthly_precipitation,
    shift_time,
    truncate_data_by_time,
)
from heiplanet_data.pipeline import (
    _STEP_REGISTRY,
    StepFn,
    _apply_preprocessing,
    _apply_simple_step,
    preprocess_data_file,
    register_step,
)
from heiplanet_data.nuts_aggregation import (
    CRS,
    _aggregate_netcdf_nuts_ee,
    _aggregate_netcdf_nuts_gpd,
    _check_aggregation_inputs,
    _prepare_for_aggregation,
    aggregate_data_by_nuts,
)

__all__ = [
    "T",
    "adjust_longitude_360_to_180",
    "convert_360_to_180",
    "convert_m_to_mm",
    "convert_m_to_mm_with_attributes",
    "convert_to_celsius",
    "convert_to_celsius_with_attributes",
    "rename_coords",
    "GridConfig",
    "ResolutionConfig",
    "align_lon_lat_with_popu_data",
    "check_agg_funcs",
    "check_downsample_condition",
    "downsample_resolution_with_cdo",
    "downsample_resolution_with_xarray",
    "downsample_resolution_with_xesmf",
    "resample_resolution",
    "upsample_resolution",
    "warn_positive_resolution",
    "calculate_monthly_precipitation",
    "shift_time",
    "truncate_data_by_time",
    "StepFn",
    "preprocess_data_file",
    "register_step",
    "CRS",
    "aggregate_data_by_nuts",
    # private names re-exported so existing tests and callers keep working
    "_replace_decimal_point",
    "_check_month_start_data",
    "_parse_date",
    "_STEP_REGISTRY",
    "_apply_preprocessing",
    "_apply_simple_step",
    "_aggregate_netcdf_nuts_ee",
    "_aggregate_netcdf_nuts_gpd",
    "_check_aggregation_inputs",
    "_prepare_for_aggregation",
]
