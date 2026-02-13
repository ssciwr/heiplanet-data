from tinydb import TinyDB, Query
import hashlib
from typing import Any, Dict, List, Tuple, Literal
from datetime import datetime
from heiplanet_data import utils


def get_db_fpath(db: TinyDB) -> str:
    """Get the file path of the TinyDB database.

    Args:
        db (TinyDB): TinyDB instance.

    Returns:
        str: File path of the JSON of TinyDB database.
    """
    return db.storage._handle.name


def _convert_to_canonicalized_str(obj: Any) -> str:
    """Convert an object to a canonicalized string for hashing.
        - dict -> string of values of sorted keys, separated by "|"
            key value are separated by "-"
        - list -> string of sorted canonicalized items, separated by "|"
        - basic types -> string
    Args:
        obj (Any): Input object (dict, list, or basic type).

    Returns:
        str: Canonicalized string.
    """
    if isinstance(obj, dict):
        items = []
        for key in sorted(obj.keys()):
            items.append(
                f"{_convert_to_canonicalized_str(key)}-{_convert_to_canonicalized_str(obj[key])}"
            )
        return "|".join(items)
    elif isinstance(obj, list):
        items = [_convert_to_canonicalized_str(x) for x in obj]
        return "|".join(sorted(items))
    else:
        return str(obj)


def _compute_hash_value(signature_data: Dict[str, Any]) -> str:
    """Compute a hash value based on the signature data.

    Args:
        signature_data (dict): Dictionary containing signature data.
            - ds_name (str): Name of the dataset.
            - product_type (str): Type of the product, if applicable.
            - data_var (str): Data variable name, only one variable is allowed.

    Returns:
        str: Computed hash value as a string.
    """

    signature_str = _convert_to_canonicalized_str(signature_data)
    hash_value = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()

    return hash_value


def _create_single_signature(
    source_dataset: str,
    product_type: str,
    data_var: str,
) -> Dict[str, Any]:
    """Create a signature dictionary for a single data variable
    directly from source dataset, product type, and data variable name.

    Args:
        source_dataset (str): Name of the source dataset.
        product_type (str): Product type.
        data_var (str): Data variable.

    Returns:
        Dict[str, Any]: Signature dictionary.
    """
    signature_var = {
        "ds_name": source_dataset,
        "product_type": product_type,
        "data_var": data_var,
    }
    return signature_var


def _create_signatures(
    source_dataset: str,
    request: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create signature dictionaries,
    one for each data variable in the request.

    Args:
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.

    Returns:
        List[Dict[str, Any]]: List of signature dictionaries.
    """
    signatures = []
    for data_var in request.get("variable", []):
        signature_var = _create_single_signature(
            source_dataset,
            request.get("product_type", ""),
            data_var,
        )
        signatures.append(signature_var)
    return signatures


def _construct_item(
    signature_var: Dict[str, Any],
    request: Dict[str, Any],
    downloaded_fpath: str,
    downloaded_at: str,
    status: str = "active",
) -> Dict[str, Any]:
    """Construct a TinyDB item dictionary.

    Args:
        signature_var (Dict[str, Any]): Signature dictionary.
        request (Dict[str, Any]): Request dictionary used to download the data.
        downloaded_fpath (str): File path where the data is saved.
        downloaded_at (str): Timestamp when the data was downloaded.
        status (str, optional): Status of the document.
            Defaults to "active".

    Returns:
        Dict[str, Any]: Constructed TinyDB item dictionary.
    """
    hash_value = _compute_hash_value(signature_var)
    item = {
        "hash": hash_value,
        "signature": signature_var,
        "year": request.get("year", []),
        "month": request.get("month", []),
        "day": request.get("day", []),
        "time": request.get("time", []),
        "file_path": downloaded_fpath,
        "data_format": request.get("data_format", ""),
        "download_format": request.get("download_format", ""),
        "downloaded_at": downloaded_at,
        "status": status,
    }
    return item


def add_new_documents(
    db_fpath: str,
    source_dataset: str,
    request: Dict[str, Any],
    downloaded_fpath: str,
    downloaded_at: str,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Add a new document to the TinyDB database.
    This function should not be used directly, but rather
    be called by the data downloading function in inout.py
    after the data is downloaded.

    Args:
        db_fpath (str): File path of the TinyDB database.
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.
        downloaded_fpath (str): File path where the data is saved.
        downloaded_at (str): Timestamp when the data was downloaded.

    Returns:
        Tuple[List[int], List[Dict[str, Any]]]: A tuple containing:
            - A list of IDs of the inserted documents in the TinyDB database.
            - A list of the inserted document dictionaries.
    """
    # compute signatures
    signatures = _create_signatures(source_dataset, request)

    # construct items
    items = []
    for signature_var in signatures:
        item = _construct_item(
            signature_var,
            request,
            downloaded_fpath,
            downloaded_at,
            status="active",
        )
        items.append(item)

    # insert items into TinyDB
    with TinyDB(db_fpath) as db:
        doc_ids = db.insert_multiple(items)

    return doc_ids, items


def update_document_status(
    db_fpath: str,
    file_path: str,
    new_status: Literal["deleted", "active"],
) -> List[int]:
    """Update the status of a document in the TinyDB database.
    This function can be used to mark a document as "deleted".

    Args:
        db_fpath (str): File path of the TinyDB database.
        file_path (str): File path of the document to be updated.
        new_status (Literal["deleted", "active"]): New status to be set.

    Returns:
        List[int]: IDs of the updated documents in the TinyDB database.
    """
    query = Query()
    with TinyDB(db_fpath) as db:
        updated_ids = db.update(
            {"status": new_status},
            query.file_path == file_path,
        )
    return updated_ids


def _find_existing_docs_by_var_request(
    db: TinyDB,
    query: Query,
    source_dataset: str,
    request: Dict[str, Any],
    data_var: str,
) -> List[Dict[str, Any]]:
    """Find existing documents in the TinyDB database,
    based on the source dataset, data variable, and request dictionary.

    Search strategy:
    - Retrieve all documents with the same signature,
        i.e. source_dataset, product_type, and data variable
    - Get documents with overlapping year, month, day, and time

    Args:
        db (TinyDB): TinyDB instance.
        query (Query): Query instance for querying the database.
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.
        data_var (str): Data variable in the request.

    Returns:
        List[Dict[str, Any]]: List of documents that match the criteria.
    """
    if data_var not in request.get("variable", []):
        raise ValueError(f"Data variable {data_var} not found in request variables.")

    # create signature
    signature_var = _create_single_signature(
        source_dataset,
        request.get("product_type", ""),
        data_var,
    )

    # find existing documents with same signatures
    # and overlapping year, month, day, time
    req_years = set(request.get("year", []))
    req_months = set(request.get("month", []))
    req_days = set(request.get("day", []))
    req_times = set(request.get("time", []))
    hash_value = _compute_hash_value(signature_var)

    results = db.search(
        (query.hash == hash_value)
        & query.year.any(req_years)
        & query.month.any(req_months)
        & query.day.any(req_days)
        & query.time.any(req_times)
    )

    return results


def find_existing_docs_by_request(
    db_fpath: str,
    source_dataset: str,
    request: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Find existing documents in the TinyDB database,
    based on the source dataset and request dictionary.

    The resulting documents match the request's hash signatures
    for each data variable and have **overlapping** year, month, day, and time.

    Args:
        db_fpath (str): File path of the TinyDB database.
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of documents that match the criteria,
            keyed by data variable.
    """
    filtered_docs = {}
    query = Query()
    with TinyDB(db_fpath) as db:
        for data_var in request.get("variable", []):
            docs_var = _find_existing_docs_by_var_request(
                db,
                query,
                source_dataset,
                request,
                data_var,
            )
            if docs_var:
                filtered_docs[data_var] = docs_var

    return filtered_docs


def find_exsiting_docs_by_var_time(
    db_fpath: str,
    ds_name: str,
    product_type: str,
    data_var: str,
    start_time: str,
    end_time: str,
) -> Tuple[
    Dict[Tuple[datetime, datetime], List[Dict[str, Any]]],
    List[Tuple[datetime, datetime]],
]:
    """Find all documents that contain data for a specific data variable,
    from a dataset with a specific product type,
    and cover a continuous time range from start_time to end_time.

    The range from start_time to end_time will be split into smaller ranges
    that each covers full years. For each smaller range,
    documents with same hashed signature and
    **overlapping** year, month, day, and time will be retrieved.

    Args:
        db_fpath (str): File path of the TinyDB database.
        ds_name (str): Dataset name.
        product_type (str): Product type.
        data_var (str): Data variable name.
        start_time (str): Start time in "%Y-%m-%d-%H:%M" format.
        end_time (str): End time in "%Y-%m-%d-%H:%M" format.

    Returns:
        Tuple[Dict[Tuple[datetime, datetime], Dict[str, Any]],
              List[Tuple[datetime, datetime]]]:
            A tuple containing:
            - A dictionary where keys are tuples of (start_datetime, end_datetime)
              representing the time ranges with existing documents,
              and values are the corresponding documents.
            - A list of tuples of (start_datetime, end_datetime)
              representing the time ranges that are not covered
              by any existing documents in the data lake.
    """
    # TODO: how about data_format, download_format, and area?
    # create signature
    signature_var = _create_single_signature(
        ds_name,
        product_type,
        data_var,
    )
    hash_value = _compute_hash_value(signature_var)

    # get time ranges from the start and end time
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d-%H:%M")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d-%H:%M")
    except ValueError:
        raise ValueError("start_time and end_time must be in '%Y-%m-%d-%H:%M' format.")

    ranges = utils.split_date_range_by_full_years(start_dt, end_dt)

    # find exsiting documents with same signatures
    # and overlapping year, month, day, time
    results = {}
    missing_ranges = []
    query = Query()
    with TinyDB(db_fpath) as db:
        for date_range in ranges:
            years, months, days, times, _ = utils.extract_years_months_days_from_range(
                date_range[0], date_range[1]
            )  # TODO: update extract_years_months_days_from_range to return times as well

            docs = db.search(
                (query.hash == hash_value)
                & query.year.any(years)
                & query.month.any(months)
                & query.day.any(days)
                & query.time.any(times)
            )
            if docs:
                results[(date_range[0], date_range[1])] = docs
            else:
                missing_ranges.append((date_range[0], date_range[1]))

    return results, missing_ranges
