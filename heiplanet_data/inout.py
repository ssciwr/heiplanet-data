import cdsapi
from pathlib import Path
import xarray as xr
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from heiplanet_data import preprocess
from dask.diagnostics.progress import ProgressBar


def download_data(
    output_file: Path, dataset: str, request: Dict[str, Any], overwrite: bool = False
):
    """Download data from Copernicus's CDS using the cdsapi.

    Args:
        output_file (Path): The path to the output file where data will be saved.
        dataset (str): The name of the dataset to download.
        request (Dict[str, Any]): A dictionary containing the request parameters.
        overwrite (bool): Whether to overwrite the output file if it already exists.
            Default is False.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")

    if not dataset or not isinstance(dataset, str):
        raise ValueError("Dataset name must be a non-empty string.")

    if not request or not isinstance(request, dict):
        raise ValueError("Request information must be a dictionary.")

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_file} already exists. Set overwrite=True to overwrite."
        )

    if not output_file.parent.exists():
        # create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    client.retrieve(dataset, request, target=str(output_file))
    print("Data downloaded successfully to {}".format(output_file))


def save_to_netcdf(data: xr.DataArray, filename: str, encoding: Optional[Dict] = None):
    """Save data to a NetCDF file.

    Args:
        data (xr.DataArray): Data to be saved.
        filename (str): The name of the output NetCDF file.
        encoding (Dict): Encoding options for the NetCDF file.
    """

    if not filename:
        raise ValueError("Filename must be provided.")

    data.to_netcdf(filename, encoding=encoding)  # TODO: check structure of encoding
    print("Data saved to {}".format(filename))


def _format_ymdt(numbers: List[str] | None) -> str:
    """Format years, months, days or times into a string representation.
    If numbers is None, return empty string.
    If numbers are continuous, return start and end values (e.g., '01_12').
    If more than 5 numbers, return first 5 numbers followed by '_etc'.
    Otherwise, return all numbers joined by underscores, each formatted as two digits.

    Args:
        numbers (List[str] | None): List of years, months, days, or times

    Returns:
        str: Formatted string representation.
    """
    if numbers is None or len(numbers) == 0:
        return ""

    num_list = sorted(int(num) for num in numbers)
    are_continuous = (len(num_list) == (max(num_list) - min(num_list) + 1)) and (
        len(num_list) > 1
    )
    if are_continuous:
        return f"{min(num_list):02d}-{max(num_list):02d}"
    elif len(num_list) > 5:
        return "_".join(f"{n:02d}" for n in num_list[:5]) + "_etc"
    else:
        return "_".join(f"{n:02d}" for n in num_list)


def _format_months(months: List[str] | None) -> str:
    """Format months into a string representation.
    If months is None, return empty string.
    If all months are selected, return 'allm'.
    If months are continuous, return start and end values.
    If more than 5 months, return first 5 months followed by '_etc'.
    Otherwise, return all months joined by underscores.

    Args:
        months (List[str] | None): List of months.

    Returns:
        str: Formatted string representation.
    """
    if months is None:
        return ""
    elif len(set(months)) == 12:
        return "allm"
    else:
        return _format_ymdt(months)


def _format_days(days: List[str] | None) -> str:
    """Format days into a string representation.
    If days is None, return empty string.
    If all days are selected, return 'alld'.
    If days are continuous, return start and end values.
    If more than 5 days, return first 5 days followed by '_etc'.
    Otherwise, return all days joined by underscores.

    Args:
        days (List[str] | None): List of days.

    Returns:
        str: Formatted string representation.
    """
    if days is None:
        return ""
    elif len(set(days)) == 31:  # TODO how about month with 30 or 28 days?
        return "alld"
    else:
        return _format_ymdt(days)


def _format_times(times: List[str] | None) -> str:
    """Format times into a string representation.
    If times is None, return empty string.
    If only "00:00" is selected, return "midnight".
    If all times are selected, return 'allt'.
    If more than 5 times, return first 5 times followed by '_etc'.
    Otherwise, return all times in two-digit format joined by underscores.

    Args:
        times (List[str] | None): List of times in "HH:MM" format

    Returns:
        str: Formatted string representation.
    """
    if times is None:
        return ""
    elif "00:00" in times and len(times) == 1:
        return "midnight"
    elif len(set(times)) == 24:
        return "allt"
    else:
        return _format_ymdt([t.split(":")[0] for t in times])


def _format_variables(variables: List[str]) -> str:
    """Format variables into a string representation.
    Join all first letters of variable names with underscores.
    E.g. ['2m_temperature', 'total_precipitation'] -> '2t_tp'.
    If the total length exceeds 30 characters,
        limit to first 2 characters then add '_etc'.

    Args:
        variables (List[str]): List of variable names.

    Returns:
        str: Formatted string representation.
    """
    var_str = "_".join(
        ["".join(word[0] for word in var.split("_")) for var in variables]
    )
    if len(var_str) > 30:
        var_str = var_str[:2] + "_etc"  # e.g. 2t_etc
    return var_str


def _format_ds_type(ds_name: str, time_str: str) -> str:
    """Format dataset type based on dataset name.
    If 'monthly' is in the dataset name, return 'monthly'.
    if 'monthly' is not in the dataset name and time_str is '_midnight',
        return 'daily'.
    Otherwise, return empty string.

    Args:
        ds_name (str): Dataset name.
        time_str (str): Formatted time string.

    Returns:
        str: Formatted dataset type.
    """
    if "monthly" in ds_name:
        return "monthly"
    elif time_str == "_midnight":
        return "daily"
    else:
        return ""


def _file_extension(data_format: str) -> str:
    """Get file extension based on data format.
    If data format is 'grib', return 'grib'.
    If data format is 'netcdf', return 'nc'.
    Otherwise, raise ValueError.

    Args:
        data_format (str): Data format (e.g., "netcdf", "grib").

    Returns:
        str: File extension.
    """
    if data_format == "grib":
        return "grib"
    elif data_format == "netcdf":
        return "nc"
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def _area_format(has_area: bool) -> str:
    """Get area string based on has_area flag.
    If has_area is True, return 'area'.
    Otherwise, return empty string.

    Args:
        has_area (bool): Flag indicating if area is included.

    Returns:
        str: Area string.
    """
    return "area" if has_area else ""


def _add_prefix_if_not_empty(s: str, prefix: str = "_") -> str:
    """Prefix a string with a given prefix if the string is not empty.

    Args:
        s (str): The input string.
        prefix (str): The prefix to add. Default is "_".

    Returns:
        str: The prefixed string if not empty, otherwise an empty string.
    """
    return f"{prefix}{s}" if s else ""


def _truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate a string to a maximum length, adding '_etc' if truncated.

    Args:
        s (str): The input string.
        max_length (int): The maximum allowed length of the string. Default is 100.

    Returns:
        str: The truncated string if it exceeds max_length, otherwise the original string.
    """
    return s if len(s) <= max_length else s[:max_length] + "_etc"


def suggest_filename(
    ds_name: str,
    data_format: str,
    years: List[str] | None,
    months: List[str] | None,
    days: List[str] | None = None,
    times: List[str] | None = None,
    has_area: bool = False,
    base_name: str = "era5_data",
    variables: List[str] = ["2m_temperature"],
) -> str:
    """Get file name based on dataset name, base name, years, months and area.

    Args:
        ds_name (str): Dataset name.
        data_format (str): Data format (e.g., "netcdf", "grib").
        years (List[str] | None): List of years.
        months (List[str] | None): List of months.
        days (List[str] | None): List of days.
        times (List[str] | None): List of times.
        has_area (bool): Flag indicating if area is included.
        base_name (str): Base name for the file.
            Default is "era5_data".
        variables (List[str]): List of variables.
            Default is ["2m_temperature"].

    Returns:
        str: Generated file name.
    """
    year_str = _add_prefix_if_not_empty(_format_ymdt(years))
    month_str = _add_prefix_if_not_empty(_format_months(months))
    day_str = _add_prefix_if_not_empty(_format_days(days))
    time_str = _add_prefix_if_not_empty(_format_times(times))
    var_str = _add_prefix_if_not_empty(_format_variables(variables))
    ds_type = _add_prefix_if_not_empty(_format_ds_type(ds_name, time_str))
    area_str = _add_prefix_if_not_empty(_area_format(has_area))

    file_name = f"{base_name}{year_str}{month_str}{day_str}{time_str}{var_str}{ds_type}{area_str}"

    file_name = _truncate_string(file_name, max_length=100)

    # add raw to file name
    file_name = file_name + "_raw"

    file_ext = _file_extension(data_format)
    file_name = file_name + "." + file_ext

    return file_name


def _split_date_range_by_full_years(
    start_time: datetime, end_time: datetime
) -> List[Tuple[datetime, datetime]]:
    """Split a date range into sub-ranges
    with one range covering as many full years as possible.
    The rest of the range is split into two sub-ranges at the start and end, if needed.
    E.g. 2020-10-15 to 2023-03-20
        -> [(2020-10-15, 2020-12-31), (2021-01-01, 2022-12-31), (2023-01-01, 2023-03-20)]
    E.g. 2020-01-01 to 2023-03-20
        -> [(2020-01-01, 2022-12-31), (2023-01-01, 2023-03-20)]
    E.g. 2020-10-15 to 2023-12-31
        -> [(2020-10-15, 2022-12-31), (2023-01-01, 2023-12-31)]
    E.g. 2020-01-01 to 2023-12-31
        -> [(2020-01-01, 2023-12-31)]

    Args:
        start_time (datetime): Start datetime.
        end_time (datetime): End datetime.

    Returns:
        List[Tuple[datetime, datetime]]: List of tuples representing the start and end
            of each sub-range.
    """
    ranges = []
    full_years_range = None

    if end_time.year - start_time.year < 1:
        return [(start_time, end_time)]

    first_full_year_start = (
        datetime(start_time.year + 1, 1, 1)
        if start_time != datetime(start_time.year, 1, 1)
        else start_time
    )
    last_full_year_end = (
        datetime(end_time.year - 1, 12, 31)
        if end_time != datetime(end_time.year, 12, 31)
        else end_time
    )

    if first_full_year_start <= last_full_year_end:
        full_years_range = (first_full_year_start, last_full_year_end)

    # from start to before full years range
    if start_time < first_full_year_start:
        ranges.append((start_time, (first_full_year_start - timedelta(days=1))))

    # full years range
    if full_years_range:
        ranges.append(full_years_range)

    # from after full years range to end
    if end_time > last_full_year_end:
        ranges.append((last_full_year_end + timedelta(days=1), end_time))

    return ranges


def _extract_years_months_days_from_range(
    start_time: datetime, end_time: datetime
) -> Tuple[List[str], List[str], List[str], bool]:
    """Extract years, months, and days from start and end datetime objects.
    For simplicity:
        * If the start and end times are in different years,
            all months and days are included.
        * If they are in the same year but different months,
            all days are included.
        * If they are in the same month,
            only the days between start and end are included.

    Note: This function becomes inefficient when the range covers just a few days
        of different years. Use function _split_date_range_by_full_years()
        to split the range into smaller ranges first.

    Args:
        start_time (datetime): Start datetime.
        end_time (datetime): End datetime.

    Returns:
        Tuple[List[str], List[str], List[str], bool]: Lists of years, months, and days
            as strings and flag to indicate if we need to truncate data later
            to get the exact range.
            Months and days are formatted as two-digit strings.
    """
    truncate_later = False

    years = [str(year) for year in range(start_time.year, end_time.year + 1)]

    not_start_year = start_time.month != 1 or start_time.day != 1
    not_end_year = end_time.month != 12 or end_time.day != 31

    if start_time.year != end_time.year:
        months = [str(month).zfill(2) for month in range(1, 13)]
        days = [str(day).zfill(2) for day in range(1, 32)]
        if not_start_year or not_end_year:
            truncate_later = True
    elif start_time.month != end_time.month:
        months = [
            f"{month:02d}" for month in range(start_time.month, end_time.month + 1)
        ]
        days = [f"{day:02d}" for day in range(1, 32)]
        if not_start_year or not_end_year:
            truncate_later = True
    else:
        months = [f"{start_time.month:02d}"]
        days = [f"{day:02d}" for day in range(start_time.day, end_time.day + 1)]

    return years, months, days, truncate_later


def _download_sub_tp_data(
    date_range: Tuple[datetime, datetime],
    range_idx: int,
    area: List[float],
    out_dir: Path,
    file_name: str,
    file_ext: str,
    ds_name: str,
    var_name: str,
    coord_name: str,
    data_format: str,
) -> Path:
    """Download total precipitation data for a given sub date range.

    Args:
        date_range (Tuple[datetime, datetime]): Tuple of start and end datetime.
        range_idx (int): Index of the current sub-range.
        area (List[float] | None): Geographical area [North, West, South, East].
        out_dir (Path): Output directory to save the downloaded file.
        file_name (str): Name of the final file to be saved.
        file_ext (str): File extension (e.g. "nc", "grib").
        ds_name (str): Dataset name.
        var_name (str): Variable name.
        coord_name (str): Name of the time coordinate in the dataset.
        data_format (str): Data format (e.g. "netcdf", "grib").

    Returns:
        Path: The path to the temporary file where data is saved.
    """
    start_time = date_range[0]
    end_time = date_range[1]

    # get data for CDS request
    years, months, days, truncate_later = _extract_years_months_days_from_range(
        start_time, end_time
    )

    # build CDS request
    request = {
        "variable": [var_name],
        "year": years,
        "month": months,
        "day": days,
        "time": ["00:00"],
        "data_format": data_format,
        "download_format": "unarchived",
    }

    if area is not None:
        request["area"] = area

    # download data
    tmp_file_path = (
        out_dir / f"{file_name}_tmp_"
        f"{start_time.strftime('%Y-%m-%d')}-{end_time.strftime('%Y-%m-%d')}"
        f".{file_ext}"
    )
    if tmp_file_path.exists():
        print(f"File {tmp_file_path} already exists. Skipping download.")
        return tmp_file_path
    else:
        print(f"Downloading data to temporary file {tmp_file_path} ...")
        download_data(tmp_file_path, ds_name, request)

    if truncate_later:
        tmp_out = tmp_file_path.with_suffix(".truncated.nc")
        with xr.open_dataset(tmp_file_path, chunks={coord_name: "auto"}) as ds:
            print(f"Truncating data of range {range_idx} ...")
            ds = preprocess.truncate_data_by_time(
                ds,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d"),
                var_name=coord_name,
            )

            # write to a another temporary file
            print(f"Writing truncated data to {tmp_out} ...")
            encoding = {
                var: {
                    "zlib": True,
                    "complevel": 4,
                }
                for var in ds.data_vars
            }
            delayed = ds.to_netcdf(
                tmp_out, mode="w", format="NETCDF4", encoding=encoding, compute=False
            )
            with ProgressBar():  # parallel writing with dask
                delayed.compute()
        # replace the original temporary file
        print(f"Replacing {tmp_file_path} with truncated data ...")
        tmp_out.replace(tmp_file_path)

    return tmp_file_path


def download_total_precipitation_from_hourly_era5_land(
    start_date: str,
    end_date: str,
    area: List[float] | None = None,
    out_dir: Path = Path("."),
    base_name: str = "era5_data",
    data_format: str = "netcdf",
    ds_name: str = "reanalysis-era5-land",
    coord_name: str = "valid_time",
    var_name: str = "total_precipitation",
    clean_tmp_files: bool = False,
) -> str:
    """Download total precipitation data from hourly ERA5-Land dataset.
    Due to the nature of this dataset, value at 00:00 is total precipitation
    of the previous day. Therefore, to get total precipitation for the given range,
    We need to download data for the given range shifted by 1 day forward,
    then shift the time value back by 1 day after downloading.

    Args:
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        area (List[float] | None): Geographical area [North, West, South, East].
            Default is None (global).
        out_dir (Path): Output directory to save the downloaded file.
            Default is current directory.
        base_name (str): Base name for the file.
            Default is "era5_data".
        data_format (str): Data format (e.g., "netcdf", "grib").
            Default is "netcdf".
        ds_name (str): Dataset name.
            Default is "reanalysis-era5-land".
            Only modify this if CDS changes the name of the dataset.
        coord_name (str): Name of the time coordinate in the dataset.
            Default is "valid_time".
            Only modify this if CDS changes the name of the coordinate.
        var_name (str): Name of the data variable.
            Default is "total_precipitation".
            Only modify this if CDS changes the name of the variable.
        clean_tmp_files (bool): Flag to indicate if temporary files should be deleted
            after processing. Default is False.

    Returns:
        str: The path to the downloaded file.
    """
    try:
        start_time = datetime.strptime(start_date, "%Y-%m-%d")
        end_time = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format.")

    if start_time > end_time:
        raise ValueError("start_date must be before or equal to end_date.")

    # prepare file name
    has_area = area is not None
    file_ext = _file_extension(data_format)
    area_str = _add_prefix_if_not_empty(_area_format(has_area))
    file_name = f"{base_name}_{start_date}-{end_date}_midnight_tp_daily{area_str}_raw"
    output_file_name = f"{file_name}.{file_ext}"
    output_file_path = out_dir / output_file_name

    # shift 1 day forward
    start_time_shifted = start_time + timedelta(days=1)
    end_time_shifted = end_time + timedelta(days=1)

    # split the range into smaller ranges
    ranges = _split_date_range_by_full_years(start_time_shifted, end_time_shifted)

    # download data
    if output_file_path.exists():
        print(f"File {output_file_path} already exists. Skipping download.")
        return str(output_file_path)

    tmp_files = []
    for i, range in enumerate(ranges):
        print(f"Downloading data for range {i}: from {range[0]} to {range[1]} ...")
        tmp_file_path = _download_sub_tp_data(
            date_range=range,
            range_idx=i,
            area=area,
            out_dir=out_dir,
            file_name=file_name,
            file_ext=file_ext,
            ds_name=ds_name,
            var_name=var_name,
            coord_name=coord_name,
            data_format=data_format,
        )
        tmp_files.append(tmp_file_path)

    assert len(ranges) == len(tmp_files)

    # combine all sub-datasets
    print("Combining all sub-datasets ...")
    combined_ds = xr.open_mfdataset(
        tmp_files, chunks={coord_name: "auto"}, combine="by_coords"
    )
    combined_ds = combined_ds.sortby(coord_name)

    # shift time back by 1 day
    print("Shifting time back by 1 day ...")
    preprocess.shift_time(
        combined_ds,
        offset=-1,
        time_unit="D",
        var_name=coord_name,
    )

    # save the processed data
    print(f"Saving the combined and shifted data to {output_file_path} ...")
    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
        }
        for var in combined_ds.data_vars
    }
    delayed = combined_ds.to_netcdf(
        output_file_path,
        mode="w",
        format="NETCDF4",
        encoding=encoding,
        compute=False,
    )
    with ProgressBar():  # parallel writing with dask
        delayed.compute()

    if clean_tmp_files:
        print("Cleaning up temporary files ...")
        # clean up temporary files
        for tmp_file in tmp_files:
            tmp_file.unlink()

    print(
        "Total precipitation data from hourly dataset saved to {}".format(
            output_file_path
        )
    )

    return str(output_file_path)
