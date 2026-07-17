"""Preprocessing step registry and pipeline orchestration.

This module wires the individual transformations from
`heiplanet_data.converters`, `heiplanet_data.regrid`, and
`heiplanet_data.temporal` into a settings-driven pipeline:

* each preprocessing step is a small wrapper function registered with the
  `register_step` decorator and an explicit ``order`` value that fixes the
  execution sequence (and the order of filename suffixes),
* `preprocess_data_file` is the public entry point: it loads the settings
  for a data source, runs all enabled steps over a NetCDF file, and writes
  the result (plus the settings used) to the output directory.

To add a new preprocessing method, write one ``@register_step`` function
here and add its keys to ``setting_schema.json`` and the settings JSON
files; the orchestrator does not need to change.
"""

from typing import Dict, Any, Tuple, Literal, Callable
import xarray as xr
import numpy as np
from pathlib import Path
import logging
from heiplanet_data import utils
from heiplanet_data.converters import (
    adjust_longitude_360_to_180,
    convert_m_to_mm_with_attributes,
    convert_to_celsius_with_attributes,
    rename_coords,
)
from heiplanet_data.regrid import (
    GridConfig,
    ResolutionConfig,
    _replace_decimal_point,
    resample_resolution,
)
from heiplanet_data.temporal import (
    calculate_monthly_precipitation,
    truncate_data_by_time,
)


logger = logging.getLogger(__name__)


# --- Preprocessing step registry -------------------------------------------
# Each step has the signature
#   (dataset, file_name_base, settings, logger) -> (dataset, file_name_base)
# and registers itself with an explicit `order` that fixes the execution
# sequence (and therefore the order of filename suffixes). To add a new
# preprocessing method: write one @register_step function below and add its
# keys to setting_schema.json and the settings JSON files. No edit to the
# orchestrator `_apply_preprocessing` is needed.
StepFn = Callable[
    [xr.Dataset, str, Dict[str, Any], logging.Logger],
    Tuple[xr.Dataset, str],
]

_STEP_REGISTRY: list[Tuple[int, str, StepFn]] = []


def register_step(name: str, order: int) -> Callable[[StepFn], StepFn]:
    """Register a preprocessing step so ``_apply_preprocessing`` runs it.

    Args:
        name (str): Unique step name (matches its settings key).
        order (int): Execution order; steps with a lower value run first.

    Returns:
        Callable[[StepFn], StepFn]: Decorator that records the step.
    """

    def decorator(fn: StepFn) -> StepFn:
        _STEP_REGISTRY.append((order, name, fn))
        return fn

    return decorator


def _apply_simple_step(
    ds: xr.Dataset,
    fname_base: str,
    *,
    enabled: bool,
    present: bool,
    message: str,
    transform: Callable[[xr.Dataset], xr.Dataset],
    suffix: str | None,
    logger: logging.Logger,
) -> Tuple[xr.Dataset, str]:
    """Run a step following the flag/condition/transform/suffix pattern.

    Args:
        ds (xr.Dataset): Dataset to transform.
        fname_base (str): Current file name base.
        enabled (bool): Whether the step is turned on in the settings.
        present (bool): Whether the target variable/coordinate exists.
        message (str): Log message emitted when the step runs.
        transform (Callable[[xr.Dataset], xr.Dataset]): Transformation to apply.
        suffix (str | None): Filename suffix appended on success.
        logger (logging.Logger): Logger for progress messages.

    Returns:
        Tuple[xr.Dataset, str]: Updated dataset and file name base.
    """
    if not (enabled and present):
        return ds, fname_base
    logger.info(message)
    ds = transform(ds)
    if suffix:
        fname_base += f"_{suffix}"
    return ds, fname_base


@register_step("unify_coords", order=10)
def _step_unify_coords(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Rename coordinates to unify them across datasets."""
    return _apply_simple_step(
        ds,
        fname_base,
        enabled=s.get("unify_coords", False),
        present=True,
        message="Renaming coordinates to unify them across datasets...",
        transform=lambda d: rename_coords(d, s.get("uni_coords")),
        suffix=s.get("unify_coords_fname"),
        logger=logger,
    )


@register_step("adjust_longitude", order=20)
def _step_adjust_longitude(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Adjust longitude from 0-360 to -180-180 (full map only)."""
    vname = s.get("adjust_longitude_vname")
    return _apply_simple_step(
        ds,
        fname_base,
        enabled=s.get("adjust_longitude", False),
        present=vname in ds.coords,
        message="Adjusting longitude from 0-360 to -180-180...",
        # only consider full map for now, i.e. limited_area=False
        transform=lambda d: adjust_longitude_360_to_180(d, lon_name=vname),
        suffix=s.get("adjust_longitude_fname"),
        logger=logger,
    )


@register_step("convert_kelvin_to_celsius", order=30)
def _step_convert_kelvin_to_celsius(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Convert temperature from Kelvin to Celsius."""
    vname = s.get("convert_kelvin_to_celsius_vname")
    return _apply_simple_step(
        ds,
        fname_base,
        enabled=s.get("convert_kelvin_to_celsius", False),
        present=vname in ds.data_vars,
        message="Converting temperature from Kelvin to Celsius...",
        transform=lambda d: convert_to_celsius_with_attributes(d, var_name=vname),
        suffix=s.get("convert_kelvin_to_celsius_fname"),
        logger=logger,
    )


@register_step("convert_m_to_mm_precipitation", order=40)
def _step_convert_m_to_mm_precipitation(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Convert precipitation from meters to millimeters."""
    vname = s.get("convert_m_to_mm_precipitation_vname")
    return _apply_simple_step(
        ds,
        fname_base,
        enabled=s.get("convert_m_to_mm_precipitation", False),
        present=vname in ds.data_vars,
        message="Converting precipitation from meters to millimeters...",
        transform=lambda d: convert_m_to_mm_with_attributes(d, var_name=vname),
        suffix=s.get("convert_m_to_mm_precipitation_fname"),
        logger=logger,
    )


@register_step("cal_monthly_tp", order=45)
def _step_cal_monthly_tp(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Calculate monthly total precipitation from downloaded monthly data."""
    vname = s.get("cal_monthly_tp_vname")
    tcoord = s.get("cal_monthly_tp_tcoord")
    return _apply_simple_step(
        ds,
        fname_base,
        enabled=s.get("cal_monthly_tp", False),
        present=all((vname in ds.data_vars, tcoord in ds.coords)),
        message=(
            "Calculating monthly total precipitation = "
            "downloaded data * number of days in month..."
        ),
        transform=lambda d: calculate_monthly_precipitation(
            d, var_name=vname, time_coord=tcoord
        ),
        suffix=s.get("cal_monthly_tp_fname"),
        logger=logger,
    )


@register_step("resample_grid", order=50)
def _step_resample_grid(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Resample the grid to a new resolution (multi-library step)."""
    resample_grid_vname = s.get("resample_grid_vname")
    lat_name, lon_name = resample_grid_vname if resample_grid_vname else (None, None)
    enabled = s.get("resample_grid", False)
    if not (enabled and all((lat_name in ds.coords, lon_name in ds.coords))):
        return ds, fname_base

    logger.info("Resampling grid to a new resolution...")
    resample_degree = s.get("resample_degree")
    ds = resample_resolution(
        ds,
        resolution_config=ResolutionConfig(
            new_resolution=resample_degree,
            lat_name=lat_name,
            lon_name=lon_name,
            downsample_lib=s.get("downsample_lib", "xesmf"),
            downsample_agg_funcs=s.get("downsample_agg_funcs", None),
            upsample_method_map=s.get("upsample_method_map", None),
        ),
        grid_config=GridConfig(
            expected_longitude_max_xarray=s.get(
                "downsample_max_lon_xarray", np.float64(179.75)
            ),
            new_min_lat=s.get("downsample_new_min_lat", None),
            new_max_lat=s.get("downsample_new_max_lat", None),
            new_min_lon=s.get("downsample_new_min_lon", None),
            new_max_lon=s.get("downsample_new_max_lon", None),
            new_lat_size=s.get("downsample_new_lat_size", None),
            new_lon_size=s.get("downsample_new_lon_size", None),
            gridtype=s.get("downsample_gridtype", "lonlat"),
        ),
    )
    degree_str = _replace_decimal_point(resample_degree)
    fname_base += f"_{degree_str}{s.get('resample_grid_fname')}"
    return ds, fname_base


@register_step("truncate_date", order=60)
def _step_truncate_date(
    ds: xr.Dataset, fname_base: str, s: Dict[str, Any], logger: logging.Logger
) -> Tuple[xr.Dataset, str]:
    """Truncate the time series to a date range."""
    truncate_date_vname = s.get("truncate_date_vname")
    if not (s.get("truncate_date", False) and truncate_date_vname in ds.coords):
        return ds, fname_base

    logger.info("Truncating data from a specific start date...")
    truncate_date_from = s.get("truncate_date_from")
    truncate_date_to = s.get("truncate_date_to")
    ds = truncate_data_by_time(
        ds,
        start_date=truncate_date_from,
        end_date=truncate_date_to,
        var_name=truncate_date_vname,
    )

    min_year = truncate_date_from[:4]
    max_time = ds[truncate_date_vname].max().values
    end_date = truncate_date_to or max_time
    max_year = np.datetime64(end_date, "Y")
    fname_base += f"_{min_year}-{max_year}"
    return ds, fname_base


def _apply_preprocessing(
    dataset: xr.Dataset,
    file_name_base: str,
    settings: Dict[str, Any],
) -> Tuple[xr.Dataset, str]:
    """Apply registered preprocessing steps to the dataset based on settings.

    Steps run in ascending order of their ``register_step`` ``order`` value,
    which determines both execution sequence and the order of filename
    suffixes.

    Args:
        dataset (xr.Dataset): Dataset to preprocess.
        file_name_base (str): Base name for the output file.
        settings (Dict[str, Any]): Settings for preprocessing.

    Returns:
        Tuple[xr.Dataset, str]: Preprocessed dataset and updated file name.
    """
    for _order, _name, step_fn in sorted(_STEP_REGISTRY, key=lambda item: item[0]):
        dataset, file_name_base = step_fn(dataset, file_name_base, settings, logger)
    return dataset, file_name_base


def preprocess_data_file(
    netcdf_file: Path,
    source: Literal["era5", "isimip"] = "era5",
    settings: Path | str = "default",
    new_settings: Dict[str, Any] | None = None,
    unique_tag: str | None = None,
) -> Tuple[xr.Dataset, str]:
    """Preprocess the dataset based on provided settings.
    If the settings path is "default", use the default settings of the source.
    The settings and preprocessed files are saved in the directory,
    which is specified by the settings file and unique number.

    Args:
        netcdf_file (Path): Path to the NetCDF file to preprocess.
        source (Literal["era5", "isimip"]): Source of the data.
            Defaults to "era5".
        settings (Path | str): Path to the settings file or "default" for default settings.
        new_settings (Dict[str, Any] | None): Additional settings to overwrite defaults.
            Defaults to None.
        unique_tag (str | None): Unique tag to append to the output file name
            and settings file.
            Defaults to None.
    Returns:
        Tuple[xr.Dataset, str]: Preprocessed dataset and
            the name of the preprocessed file.
    """
    if not utils.is_non_empty_file(netcdf_file):
        raise ValueError(f"NetCDF file {netcdf_file} does not exist or is empty.")

    # generate unique tag for the settings file and output file
    if unique_tag is None or not unique_tag:
        unique_tag = utils.generate_unique_tag()

    # load settings
    settings, settings_fname = utils.load_settings(
        source=source, setting_path=settings, new_settings=new_settings
    )

    folder_path = Path(settings.get("output_dir", "data/processed"))
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    # save settings to a file
    settings_fname_w_tag = f"{settings_fname}_{unique_tag}.json"
    utils.save_settings_to_file(settings, folder_path, settings_fname_w_tag)

    # prepare to preprocess NetCDF file
    file_name = netcdf_file.stem
    file_name = file_name[: -len("_raw")] if file_name.endswith("_raw") else file_name
    file_ext = netcdf_file.suffix

    with xr.open_dataset(netcdf_file, chunks={}) as dataset:
        dataset, file_name_base = _apply_preprocessing(dataset, file_name, settings)
        # save the processed dataset
        output_file = folder_path / f"{file_name_base}_{unique_tag}{file_ext}"
        dataset.to_netcdf(output_file, mode="w", format="NETCDF4")
        logger.info(f"Processed dataset saved to: {output_file}")
        return dataset, str(output_file.name)
