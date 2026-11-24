"""TEMPORARY: combine yearly P-Model gold files into one (time, lat, lon) file.

Workaround for https://github.com/ssciwr/heiplanet-models/issues/84 :
the P-Model currently writes
  * one file per year (``Mosquito_abundance_Global_<year>.nc``) next to a
    ``*_final_state.nc`` restart file per year, and
  * variables with dimension order ``(longitude, latitude, time)``.

This script concatenates the yearly monthly-mean files (restart files are
skipped) along time, transposes all variables to ``(time, latitude,
longitude)`` and writes a single file
``Mosquito_abundance_Global_<first>-<last>.nc`` that can be used for NUTS
aggregation like the J-Model output.

Delete this script once heiplanet-models#84 is resolved.

Usage:
    python scripts/pmodel_combine_tmp.py <gold_dir> <first_year> <last_year> [<out_file>]
"""

import re
import sys
from pathlib import Path

import xarray as xr

PREFIX = "Mosquito_abundance_Global_"
YEAR_FILE = re.compile(rf"^{PREFIX}(\d{{4}})\.nc$")  # excludes *_final_state.nc
DIMS = ("time", "latitude", "longitude")
USAGE = "usage: pmodel_combine_tmp.py <gold_dir> <first_year> <last_year> [<out_file>]"


def find_year_files(gold_dir: Path, first: int, last: int) -> list[Path]:
    """Return the yearly P-Model output files for first..last, sorted by year."""
    found = {
        int(m.group(1)): p for p in gold_dir.iterdir() if (m := YEAR_FILE.match(p.name))
    }
    years = list(range(first, last + 1))
    missing = [y for y in years if y not in found]
    if missing:
        raise FileNotFoundError(f"Missing {PREFIX}<year>.nc in {gold_dir}: {missing}")
    return [found[y] for y in years]


def combine(files: list[Path], out_file: Path) -> None:
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        data_vars="minimal",
        coords="minimal",  # type: ignore[arg-type]
        compat="override",
        join="exact",  # all years must share the same grid
        chunks={},  # native on-disk chunks
    )
    ds = ds.transpose(*DIMS)

    times = ds["time"].to_index()
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Combined time axis is not strictly increasing.")

    ds.attrs["note"] = (
        f"Combined from {len(files)} yearly P-Model files and transposed to "
        f"{DIMS} (temporary workaround for heiplanet-models#84)."
    )
    ds.attrs["source_files"] = ", ".join(f.name for f in files)

    n_lat, n_lon = ds.sizes["latitude"], ds.sizes["longitude"]
    encoding = {
        v: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            # one time step, ~1/5 x 1/5 of the globe per chunk
            "chunksizes": (1, -(-n_lat // 5), -(-n_lon // 5)),
        }
        for v in ds.data_vars
    }
    ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)


def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(USAGE)
    gold_dir = Path(sys.argv[1])
    first, last = int(sys.argv[2]), int(sys.argv[3])
    files = find_year_files(gold_dir, first, last)
    out_file = (
        Path(sys.argv[4])
        if len(sys.argv) == 5
        else gold_dir / f"{PREFIX}{first}-{last}.nc"
    )
    if out_file.exists():
        sys.exit(f"Refusing to overwrite existing file: {out_file}")

    print(f"Combining {len(files)} files ({first}-{last}) -> {out_file}")
    combine(files, out_file)

    with xr.open_dataset(out_file) as out:
        print(out)
        for v in out.data_vars:
            assert out[v].dims == DIMS, (v, out[v].dims)


if __name__ == "__main__":
    main()
