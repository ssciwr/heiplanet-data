from heiplanet_data import data_lake
import pytest
from tinydb import Query


def test_get_db_fpath(get_empty_db_query):
    db, _ = get_empty_db_query
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
    assert data_lake._convert_to_canonicalized_str("") == ""
    assert data_lake._convert_to_canonicalized_str([]) == ""
    assert data_lake._convert_to_canonicalized_str({}) == ""

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
        product_type=["reanalysis"],
        data_var="t2m",
    ) == {
        "ds_name": "era5-land",
        "product_type": ["reanalysis"],
        "data_var": "t2m",
    }

    with pytest.raises(ValueError):
        data_lake._create_single_signature(
            source_dataset="era5-land",
            product_type=["reanalysis"],
            data_var="",
        )

    with pytest.raises(ValueError):
        data_lake._create_single_signature(
            source_dataset="",
            product_type=["reanalysis"],
            data_var="t2m",
        )


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
        sig["product_type"] = []
    assert signatures_wo == expected_signatures_wo


def test_construct_item(get_mock_data, get_mock_items, get_mock_download_info):
    _, request, signatures, _ = get_mock_data
    downloaded_fpath, downloaded_at, status = get_mock_download_info

    for signature, expected_item in zip(signatures, get_mock_items):
        item = data_lake._construct_item(
            signature,
            request,
            downloaded_fpath=downloaded_fpath,
            downloaded_at=downloaded_at,
            status=status,
        )
        assert item == expected_item


def test_add_new_documents(
    get_empty_db_query,
    get_mock_data,
    get_mock_items,
    get_mock_download_info,
    get_db_with_mock_data,
):
    db, _ = get_empty_db_query
    ds_source, request, _, _ = get_mock_data
    downloaded_fpath, downloaded_at, _ = get_mock_download_info

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
    for item_id, inserted_item, mock_item in zip(
        inserted_ids, inserted_items, get_mock_items
    ):
        db_item = db.all()[item_id - 1]  # TinyDB ids start from 1
        mock_db_item = get_db_with_mock_data.all()[item_id - 1]
        assert db_item == inserted_item
        assert db_item == mock_item
        assert db_item == mock_db_item  # maybe just use one of the last two assertions


def test_update_document_status(get_db_with_mock_data):
    db_fpath = get_db_with_mock_data.storage._handle.name

    fpath = get_db_with_mock_data.all()[0]["file_path"]

    # update the document status
    new_status = "deleted"
    updated_ids = data_lake.update_document_status(db_fpath, fpath, new_status)

    # check if the status is updated in the db
    query = Query()
    db_items = get_db_with_mock_data.search(query.file_path == fpath)

    assert len(db_items) == len(updated_ids) == 2
    for db_item in db_items:
        assert db_item["status"] == new_status


def test_find_existing_docs_by_var_request_invalid(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, _, _ = get_mock_data

    with pytest.raises(ValueError):
        data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source,
            request=request,
            data_var="invalid_var",
        )


def test_find_existing_docs_by_var_request_exact_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        existing_docs = data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source,
            request=request,
            data_var=data_var,
        )
        assert len(existing_docs) == 1
        assert existing_docs[0]["signature"]["data_var"] == data_var
        assert (
            existing_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_var_request_no_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    # no match due to source dataset
    for signature in signatures:
        data_var = signature["data_var"]
        existing_docs = data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source + "_different",
            request=request,
            data_var=data_var,
        )
        assert len(existing_docs) == 0

    # no match due to request
    modified_request = request.copy()
    modified_request["month"] = ["02"]  # different month
    for signature in signatures:
        data_var = signature["data_var"]
        existing_docs = data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source,
            request=modified_request,
            data_var=data_var,
        )
        assert len(existing_docs) == 0


def test_find_existing_docs_by_var_request_partial_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    modified_request = request.copy()
    modified_request["year"] = ["2017", "2018"]  # overlap with original years

    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        existing_docs = data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source,
            request=modified_request,
            data_var=data_var,
        )
        assert len(existing_docs) == 1
        assert (
            existing_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_var_request_full_ymdt(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    # data in the db has year 2016-2017, month 01, time 00:00
    # meaning all days in Jan 2016 and Jan 2017 at 00:00 are covered

    modified_request = request.copy()
    modified_request["day"] = ["11", "12"]

    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        existing_docs = data_lake._find_existing_docs_by_var_request(
            db=get_db_with_mock_data,
            query=Query(),
            source_dataset=ds_source,
            request=modified_request,
            data_var=data_var,
        )
        assert len(existing_docs) == 1
        assert (
            existing_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_request_exact_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    filtered_docs = data_lake.find_existing_docs_by_request(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        request=request,
    )

    assert len(filtered_docs) == 2
    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        matched_docs = filtered_docs[data_var]
        assert len(matched_docs) == 1
        assert (
            matched_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_request_no_match(get_db_with_mock_data, get_mock_data):
    ds_source, request, _, _ = get_mock_data

    # no match due to source dataset
    filtered_docs = data_lake.find_existing_docs_by_request(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source + "_different",
        request=request,
    )

    assert len(filtered_docs) == 0

    # no match due to request
    modified_request = request.copy()
    modified_request["month"] = ["02"]  # different month

    filtered_docs = data_lake.find_existing_docs_by_request(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        request=modified_request,
    )

    assert len(filtered_docs) == 0


def test_find_existing_docs_by_request_partial_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    modified_request = request.copy()
    modified_request["month"] = ["01", "02"]  # overlap with original months

    filtered_docs = data_lake.find_existing_docs_by_request(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        request=modified_request,
    )

    assert len(filtered_docs) == 2
    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        matched_docs = filtered_docs[data_var]
        assert len(matched_docs) == 1
        assert (
            matched_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_request_full_ymdt(get_db_with_mock_data, get_mock_data):
    ds_source, request, signatures, _ = get_mock_data

    modified_request = request.copy()
    modified_request["day"] = ["21", "25"]  # all days in Jan are covered

    filtered_docs = data_lake.find_existing_docs_by_request(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        request=modified_request,
    )

    assert len(filtered_docs) == 2
    for idx in range(len(signatures)):
        data_var = signatures[idx]["data_var"]
        matched_docs = filtered_docs[data_var]
        assert len(matched_docs) == 1
        assert (
            matched_docs[0]["file_path"]
            == get_db_with_mock_data.all()[idx]["file_path"]
        )


def test_find_existing_docs_by_var_time_invalid():
    with pytest.raises(ValueError):
        data_lake.find_existing_docs_by_var_time(
            db_fpath="test_db.json",
            source_dataset="era5-land",
            product_type="reanalysis",
            data_var="t2m",
            start_time="something_wrong",  # invalid format
            end_time=123456789,  # invalid format
        )


def test_find_existing_docs_by_var_time_no_match(get_db_with_mock_data, get_mock_data):
    ds_source, request, signatures, _ = get_mock_data

    all_days = [f"{day:02d}" for day in range(1, 32)]
    all_months = [f"{month:02d}" for month in range(1, 13)]

    product_type = request["product_type"][0]

    # no match due to non-overlapping time range
    for signature in signatures:
        data_var = signature["data_var"]
        results, missing_requests = data_lake.find_existing_docs_by_var_time(
            db_fpath=get_db_with_mock_data.storage._handle.name,
            source_dataset=ds_source,
            product_type=product_type,
            data_var=data_var,
            start_time="2018-01-01",
            end_time="2019-12-31",
        )
        assert len(results) == 0
        assert len(missing_requests) == 1
        assert missing_requests[0]["variable"] == [data_var]
        assert missing_requests[0]["year"] == ["2018", "2019"]
        assert missing_requests[0]["month"] == all_months
        assert missing_requests[0]["day"] == all_days
        assert missing_requests[0]["time"] == ["00:00"]

    # no match due to different source name, product type, or data variable
    results, missing_requests = data_lake.find_existing_docs_by_var_time(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source + "_different",
        product_type=product_type,
        data_var="t2m",
        start_time="2016-01-01",
        end_time="2016-02-28",
    )
    assert len(results) == 0
    assert len(missing_requests) == 1
    assert missing_requests[0]["variable"] == ["t2m"]
    assert missing_requests[0]["year"] == ["2016"]
    assert missing_requests[0]["month"] == ["01", "02"]
    assert missing_requests[0]["day"] == all_days
    assert missing_requests[0]["time"] == ["00:00"]

    results, _ = data_lake.find_existing_docs_by_var_time(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        product_type=product_type + "_different",
        data_var="t2m",
        start_time="2016-01-01",
        end_time="2016-02-28",
    )
    assert len(results) == 0

    results, _ = data_lake.find_existing_docs_by_var_time(
        db_fpath=get_db_with_mock_data.storage._handle.name,
        source_dataset=ds_source,
        product_type=product_type,
        data_var="different_var",
        start_time="2016-01-01",
        end_time="2017-12-31",
    )
    assert len(results) == 0


def test_find_existing_docs_by_var_time_partial_match(
    get_db_with_mock_data, get_mock_data
):
    ds_source, request, signatures, _ = get_mock_data

    product_type = request["product_type"][0]

    # time at midnight
    for signature in signatures:
        data_var = signature["data_var"]
        results, missing_requests = data_lake.find_existing_docs_by_var_time(
            db_fpath=get_db_with_mock_data.storage._handle.name,
            source_dataset=ds_source,
            product_type=product_type,
            data_var=data_var,
            start_time="2017-01-01",
            end_time="2018-12-31",
        )
        assert len(results) == 1
        # TODO: check missing requests
