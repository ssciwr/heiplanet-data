"""Tests for the backward-compatible ``preprocess`` facade.

The implementation tests live in ``test_converters.py``, ``test_regrid.py``,
``test_temporal.py``, ``test_pipeline.py``, and ``test_nuts_aggregation.py``.
"""

from heiplanet_data import (
    converters,
    nuts_aggregation,
    pipeline,
    preprocess,
    regrid,
    temporal,
)


def test_facade_reexports_all_names():
    for name in preprocess.__all__:
        assert hasattr(preprocess, name)


def test_facade_names_are_the_implementation_objects():
    assert preprocess.convert_to_celsius is converters.convert_to_celsius
    assert preprocess.resample_resolution is regrid.resample_resolution
    assert preprocess.truncate_data_by_time is temporal.truncate_data_by_time
    assert preprocess.preprocess_data_file is pipeline.preprocess_data_file
    assert preprocess.aggregate_data_by_nuts is nuts_aggregation.aggregate_data_by_nuts


def test_facade_shares_step_registry():
    # registering a step through the facade must be visible to the pipeline
    assert preprocess._STEP_REGISTRY is pipeline._STEP_REGISTRY
    assert preprocess.register_step is pipeline.register_step
