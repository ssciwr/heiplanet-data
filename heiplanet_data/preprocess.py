"""Backward-compatible facade for the preprocessing modules.

The implementation was split into dedicated modules; import from them
directly in new code:

* :mod:`heiplanet_data.converters` — unit and coordinate conversions
* :mod:`heiplanet_data.regrid` — spatial resolution resampling
* :mod:`heiplanet_data.temporal` — time-axis operations
* :mod:`heiplanet_data.pipeline` — step registry and orchestration
* :mod:`heiplanet_data.nuts_aggregation` — aggregation over NUTS regions

This module re-exports the previous public API so existing imports of
``heiplanet_data.preprocess`` keep working. Each name uses the
``name as name`` re-export idiom, so the import list below is the single
authoritative export list.

Note: log records are emitted under the implementation modules' logger
names (e.g. ``heiplanet_data.pipeline``) instead of
``heiplanet_data.preprocess``; configure logging on the ``heiplanet_data``
parent logger to capture them.
"""

from heiplanet_data.converters import (
    T as T,
    adjust_longitude_360_to_180 as adjust_longitude_360_to_180,
    convert_360_to_180 as convert_360_to_180,
    convert_m_to_mm as convert_m_to_mm,
    convert_m_to_mm_with_attributes as convert_m_to_mm_with_attributes,
    convert_to_celsius as convert_to_celsius,
    convert_to_celsius_with_attributes as convert_to_celsius_with_attributes,
    rename_coords as rename_coords,
)
from heiplanet_data.regrid import (
    GridConfig as GridConfig,
    ResolutionConfig as ResolutionConfig,
    align_lon_lat_with_popu_data as align_lon_lat_with_popu_data,
    check_agg_funcs as check_agg_funcs,
    check_downsample_condition as check_downsample_condition,
    downsample_resolution_with_cdo as downsample_resolution_with_cdo,
    downsample_resolution_with_xarray as downsample_resolution_with_xarray,
    downsample_resolution_with_xesmf as downsample_resolution_with_xesmf,
    resample_resolution as resample_resolution,
    upsample_resolution as upsample_resolution,
    warn_positive_resolution as warn_positive_resolution,
)
from heiplanet_data.temporal import (
    _check_month_start_data as _check_month_start_data,
    _parse_date as _parse_date,
    calculate_monthly_precipitation as calculate_monthly_precipitation,
    shift_time as shift_time,
    truncate_data_by_time as truncate_data_by_time,
)
from heiplanet_data.pipeline import (
    _STEP_REGISTRY as _STEP_REGISTRY,
    StepFn as StepFn,
    _apply_preprocessing as _apply_preprocessing,
    _apply_simple_step as _apply_simple_step,
    _replace_decimal_point as _replace_decimal_point,
    preprocess_data_file as preprocess_data_file,
    register_step as register_step,
)
from heiplanet_data.nuts_aggregation import (
    CRS as CRS,
    _aggregate_netcdf_nuts_ee as _aggregate_netcdf_nuts_ee,
    _aggregate_netcdf_nuts_gpd as _aggregate_netcdf_nuts_gpd,
    _check_aggregation_inputs as _check_aggregation_inputs,
    _prepare_for_aggregation as _prepare_for_aggregation,
    aggregate_data_by_nuts as aggregate_data_by_nuts,
)
