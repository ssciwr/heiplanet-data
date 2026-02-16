from heiplanet_data import data_lake
import hashlib


def test_get_db_fpath(get_db_query):
    db, _ = get_db_query
    db_fpath = data_lake.get_db_fpath(db)
    assert db_fpath is not None
    assert str(db_fpath).endswith("test_db.json")


def test_convert_to_canonicalized_str():
    # built-in types
    assert data_lake._convert_to_canonicalized_str(42) == "42"
    assert data_lake._convert_to_canonicalized_str("hello") == "hello"
    assert data_lake._convert_to_canonicalized_str(3.14) == "3.14"
    assert data_lake._convert_to_canonicalized_str(True) == "True"
    assert data_lake._convert_to_canonicalized_str(None) == "None"
    assert data_lake._convert_to_canonicalized_str(b"byte") == "b'byte'"

    # flat list
    assert data_lake._convert_to_canonicalized_str([3, 1, 2]) == "1|2|3"

    # nested list
    assert data_lake._convert_to_canonicalized_str([3, [2, 1], 4]) == "1|2|3|4"

    # flat dict
    assert data_lake._convert_to_canonicalized_str({"b": 2, "a": 1}) == "a-1|b-2"

    # nested dict
    assert (
        data_lake._convert_to_canonicalized_str(
            {"b": {"y": 20, "x": 10}, "a": {1: "one", 2: "two"}}
        )
        == "a-1-one|2-two|b-x-10|y-20"
    )
    assert (
        data_lake._convert_to_canonicalized_str({"a": {"x": 10, "y": 20}, "b": [3, 4]})
        == "a-x-10|y-20|b-3|4"
    )

    # list of dicts
    assert (
        data_lake._convert_to_canonicalized_str([{"d": 4, "c": 3}, {"b": 2, "a": 1}])
        == "a-1|b-2|c-3|d-4"
    )


def test_compute_hash_value():
    # same input should give same hash
    input1 = {"a": 1, "b": [3, 2]}
    input2 = {"b": [2, 3], "a": 1}
    assert data_lake._compute_hash_value(input1) == data_lake._compute_hash_value(
        input2
    )

    # different input should give different hash
    input3 = {"a": 1, "b": [3, 2, 4]}
    assert data_lake._compute_hash_value(input1) != data_lake._compute_hash_value(
        input3
    )

    # different structure
    input4 = {"a": 1, "b": {"x": 10, "y": 20}}
    assert data_lake._compute_hash_value(input1) != data_lake._compute_hash_value(
        input4
    )
    input5 = {"a": 1, "b": {"c": [3, 2]}}
    assert data_lake._compute_hash_value(input1) != data_lake._compute_hash_value(
        input5
    )

    # complex nested structure
    input6 = {"a": 1, "b": [{"x": 10}, {"y": 20}]}
    input7 = {"b": [{"y": 20}, {"x": 10}], "a": 1}
    assert data_lake._compute_hash_value(input6) == data_lake._compute_hash_value(
        input7
    )


def test_create_single_signature():
    assert data_lake._create_single_signature(
        source_dataset="era5-land",
        product_type="reanalysis",
        data_var="t2m",
    ) == {
        "ds_name": "era5-land",
        "product_type": "reanalysis",
        "data_var": "t2m",
    }


def test_create_signatures(get_mock_data):
    ds_source, request, expected_signatures, _ = get_mock_data
    # with product_type
    signatures = data_lake._create_signatures(
        source_dataset=ds_source,
        request=request,
    )
    assert signatures == expected_signatures

    # without product_type
    request = request.copy()
    request.pop("product_type")
    signatures_wo = data_lake._create_signatures(
        source_dataset=ds_source,
        request=request,
    )
    expected_signatures_wo = expected_signatures.copy()
    for sig in expected_signatures_wo:
        sig["product_type"] = ""
    assert signatures_wo == expected_signatures_wo


def test_construct_item(get_mock_data):
    _, request, signatures, signature_strs = get_mock_data
    signature = signatures[0]
    signature_str = signature_strs[0]

    item = data_lake._construct_item(
        signature,
        request,
        downloaded_fpath="test.nc",
        downloaded_at="2026-02-13",
        status="active",
    )

    hash_value = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()
    expected_item = {
        "hash": hash_value,
        "signature": signature,
        "year": request.get("year", []),
        "month": request.get("month", []),
        "day": request.get("day", []),
        "time": request.get("time", []),
        "file_path": "test.nc",
        "data_format": request.get("data_format", ""),
        "download_format": request.get("download_format", ""),
        "downloaded_at": "2026-02-13",
        "status": "active",
    }
    assert item == expected_item


def test_add_new_documents(get_db_query, get_mock_data):
    db, _ = get_db_query
    ds_source, request, _, _ = get_mock_data

    downloaded_fpath = "test.nc"
    downloaded_at = "2026-02-13"
    db_fpath = db.storage._handle.name
    inserted_ids, inserted_items = data_lake.add_new_documents(
        db_fpath,
        ds_source,
        request,
        downloaded_fpath,
        downloaded_at,
    )

    assert len(inserted_ids) == 2
    assert len(inserted_items) == 2

    # check if the items are correctly inserted in the db
    for item_id, item in zip(inserted_ids, inserted_items):
        db_item = db.all()[item_id - 1]  # TinyDB ids start from 1
        assert db_item == item


def test_update_document_status(get_db_query):
    db, query = get_db_query
    db_fpath = db.storage._handle.name

    # insert a sample document
    fpath = "tmp/test.nc"
    item_id = db.insert({"file_path": fpath, "status": "active"})

    # update the document status
    new_status = "deleted"
    updated_ids = data_lake.update_document_status(db_fpath, fpath, new_status)

    # check if the status is updated in the db
    db_item = db.search(query.file_path == fpath)[0]
    assert db_item["status"] == new_status
    assert updated_ids == [item_id]
