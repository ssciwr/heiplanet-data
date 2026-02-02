from pathlib import Path
from importlib import resources
import json
import jsonschema
import warnings
from typing import Dict, Any, Tuple
from datetime import datetime
import socket
from typing import Optional, List
from datetime import timedelta


pkg = resources.files("heiplanet_data")
DEFAULT_SETTINGS_FILE = {
    "era5": Path(pkg / "era5_settings.json"),
    "isimip": Path(pkg / "isimip_settings.json"),
}


def is_non_empty_file(file_path: Path) -> bool:
    """Check if a file exists and is not empty.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file exists and is not empty, False otherwise.
    """
    invalid_file = (
        not file_path or not file_path.exists() or file_path.stat().st_size == 0
    )
    if invalid_file:
        return False

    return True


def is_valid_settings(settings: dict) -> bool:
    """Check if the settings are valid.
    Args:
        settings (dict): The settings.

    Returns:
        bool: True if the settings are valid, False otherwise.
    """
    pkg = resources.files("heiplanet_data")
    setting_schema_path = Path(pkg / "setting_schema.json")
    setting_schema = json.load(open(setting_schema_path, "r", encoding="utf-8"))

    try:
        jsonschema.validate(instance=settings, schema=setting_schema)
        return True
    except jsonschema.ValidationError as e:
        print(e)
        return False


def _update_new_settings(settings: dict, new_settings: dict) -> bool:
    """Update the settings directly with the new settings.

    Args:
        settings (dict): The settings.
        new_settings (dict): The new settings.

    Returns:
        bool: True if the settings are updated, False otherwise.
    """
    updated = False
    if not settings:
        raise ValueError("Current settings are empty")

    for key, new_value in new_settings.items():
        # check if the new value is different from the old value
        # if the setting schema has more nested structures, deepdiff should be used
        # here just simple check
        updatable = key in settings and settings[key] != new_value
        if key not in settings:
            warnings.warn(
                "Key {} not found in the settings and will be skipped.".format(key),
                UserWarning,
            )
        if updatable:
            old_value = settings[key]
            settings[key] = new_value
            if is_valid_settings(settings):
                updated = True
            else:
                warnings.warn(
                    "The new value for key {} is not valid in the settings. "
                    "Reverting to the old value: {}".format(key, old_value),
                    UserWarning,
                )
                settings[key] = old_value

    return updated


def save_settings_to_file(
    settings: dict,
    dir_path: Optional[str] = None,
    file_name: str = "updated_settings.json",
) -> None:
    """Save the settings to a file.
    If dir_path is None, save to the current directory.

    Args:
        settings (dict): The settings.
        dir_path (str, optional): The path to save the settings file.
            Defaults to None.
        file_name (str, optional): The name for the settings file.
            Defaults to "updated_settings.json".
    """
    file_path = ""

    if dir_path is None:
        file_path = Path.cwd() / file_name
    else:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            file_path = Path(dir_path) / file_name
        except FileExistsError:
            raise ValueError(
                "The path {} already exists and is not a directory".format(dir_path)
            )

    # save the settings to a file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

    print("The settings have been saved to {}".format(file_path))


def load_settings(
    source: str = "era5",
    setting_path: Path | str = "default",
    new_settings: dict | None = None,
) -> Tuple[Dict[str, Any], str]:
    """Get the settings for preprocessing steps.
    If the setting path is "default", return the default settings of the source.
    If the setting path is not default, read the settings from the file.
    If the new settings are provided, overwrite the default/loaded settings.

    Args:
        source (str): Source of the data to get corresponding settings.
        setting_path (Path | str): Path to the settings file.
            Defaults to "default".
        new_settings (dict | None): New settings to overwrite the existing settings.
            Defaults to {}.

    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing the settings dictionary
            and the name of the settings file.
    """
    settings = {}
    settings_fname = ""
    default_setting_path = DEFAULT_SETTINGS_FILE.get(source)

    if not default_setting_path or not is_non_empty_file(default_setting_path):
        raise ValueError(
            f"Default settings file for source {source} not found or is empty."
        )

    def load_json(file_path: Path) -> Tuple[Dict[str, Any], str]:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file), file_path.stem

    try:
        settings, settings_fname = (
            load_json(default_setting_path)
            if setting_path == "default"
            else load_json(Path(setting_path))
        )
        if setting_path != "default" and not is_valid_settings(settings):
            warnings.warn(
                "Invalid settings file. Using default settings instead.",
                UserWarning,
            )
            settings, settings_fname = load_json(default_setting_path)
    except Exception:
        warnings.warn(
            "Error in loading the settings file. Using default settings instead.",
            UserWarning,
        )
        settings, settings_fname = load_json(default_setting_path)

    # update the settings with the new settings
    if new_settings and isinstance(new_settings, dict):
        _update_new_settings(settings, new_settings)

    return settings, settings_fname


def generate_unique_tag() -> str:
    """Generate a unique tag based on the current timestamp and hostname.

    Returns:
        str: A unique tag in the format "YYYYMMDD-HHMMSS_hostname".
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    return f"ts{timestamp}_h{hostname}"


def split_date_range_by_full_years(
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


def extract_years_months_days_from_range(
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
        of different years. Use function split_date_range_by_full_years()
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
