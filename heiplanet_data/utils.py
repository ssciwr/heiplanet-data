import json
import socket
import warnings
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

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
                f"Key {key} not found in the settings and will be skipped.",
                UserWarning,
            )
        if updatable:
            old_value = settings[key]
            settings[key] = new_value
            if is_valid_settings(settings):
                updated = True
            else:
                warnings.warn(
                    f"The new value for key {key} is not valid in the settings. "
                    f"Reverting to the old value: {old_value}",
                    UserWarning,
                )
                settings[key] = old_value

    return updated


def save_settings_to_file(
    settings: dict,
    dir_path: str | None = None,
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
                f"The path {dir_path} already exists and is not a directory"
            )

    # save the settings to a file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

    print(f"The settings have been saved to {file_path}")


def load_settings(
    source: str = "era5",
    setting_path: Path | str = "default",
    new_settings: dict | None = None,
) -> tuple[dict[str, Any], str]:
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

    def load_json(file_path: Path) -> tuple[dict[str, Any], str]:
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
