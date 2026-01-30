from tinydb import TinyDB, Query
import hashlib
from typing import Any, Dict, List, Tuple, Literal


def load_db(db_fpath: str) -> Tuple[TinyDB, Query]:
    """Load data file into TinyDB instance.

    Args:
        db_fpath (str): Path to the JSON data file.
    Returns:
        Tuple[TinyDB, Query]: TinyDB instance and Query class.
    """
    return TinyDB(db_fpath), Query()


def _convert_to_canonicalized_str(obj: Any) -> str:
    """Convert an object to a canonicalized string for hashing.
        - dict -> string of values of sorted keys
        - list -> list of sorted canonicalized items
        - basic types -> string
    Args:
        obj (Any): Input object (dict, list, or basic type).

    Returns:
        str: Canonicalized string.
    """
    if isinstance(obj, dict):
        items = []
        for key in sorted(obj.keys()):
            items.append(_convert_to_canonicalized_str(obj[key]))
        return "|".join(items)
    elif isinstance(obj, list):
        items = [_convert_to_canonicalized_str(x) for x in sorted(obj)]
        return "|".join(items)
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

    singature_str = _convert_to_canonicalized_str(signature_data)
    hash_value = hashlib.sha256(singature_str.encode("utf-8")).hexdigest

    return hash_value


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
        signature_var = {
            "ds_name": source_dataset,
            "product_type": request.get("product_type", ""),
            "data_var": data_var,
        }
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
        "time": request.get("time", []),
        "file_path": downloaded_fpath,
        "data_format": request.get("data_format", ""),
        "download_format": request.get("download_format", ""),
        "downloaded_at": downloaded_at,
        "status": status,
    }
    return item


def add_new_document(
    db: TinyDB,
    source_dataset: str,
    request: Dict[str, Any],
    downloaded_fpath: str,
    downloaded_at: str,
) -> List[int]:
    """Add a new document to the TinyDB database.

    Args:
        db (TinyDB): TinyDB instance.
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.
        downloaded_fpath (str): File path where the data is saved.
        downloaded_at (str): Timestamp when the data was downloaded.

    Returns:
        List[int]: IDs of the newly added documents in the TinyDB database.
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
        )
        items.append(item)

    # insert items into TinyDB
    doc_ids = db.insert_multiple(items)

    return doc_ids


def update_document_status(
    db: TinyDB,
    query: Query,
    file_path: str,
    new_status: Literal["deleted", "active"],
) -> List[int]:
    """Update the status of a document in the TinyDB database.
    This function can be used to mark a document as "deleted".

    Args:
        db (TinyDB): TinyDB instance.
        query (Query): Query instance for querying the database.
        file_path (str): File path of the document to be updated.
        new_status (Literal["deleted", "active"]): New status to be set.

    Returns:
        List[int]: IDs of the updated documents in the TinyDB database.
    """
    updated_ids = db.update(
        {"status": new_status},
        query.file_path == file_path,
    )
    return updated_ids


def find_existing_docs_by_request(
    db: TinyDB,
    query: Query,
    source_dataset: str,
    request: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find existing documents in the TinyDB database,
    based on the source dataset and request dictionary.

    Args:
        db (TinyDB): TinyDB instance.
        query (Query): Query instance for querying the database.
        source_dataset (str): Name of the source dataset.
        request (Dict[str, Any]): Request dictionary used to download the data.
    """
    # TODO: redesign thisfunction
    return


def find_docs_var_continues_time(
    db: TinyDB,
    query: Query,
    ds_name: str,
    product_type: str,
    data_var: str,
    start_time: str,
    end_time: str,
) -> List[Dict[str, Any]]:
    """Find all documents that contain data for a specific data variable,
    from a dataset with a specific product type,
    and cover a continuous time range from start_time to end_time.

    Args:
        db (TinyDB): TinyDB instance.
        query (Query): Query instance for querying the database.
        ds_name (str): Dataset name.
        product_type (str): Product type.
        data_var (str): Data variable name.
        start_time (str): Start time in "%Y-%m-%d-%H:%M" format.
        end_time (str): End time in "%Y-%m-%d-%H:%M" format.

    Returns:
        List[Dict[str, Any]]: List of documents that match the criteria.
    """
    # TODO: move functions handling time range from inout to utils.py
    pass
