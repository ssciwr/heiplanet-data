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

SUBMODULES = (converters, regrid, temporal, pipeline, nuts_aggregation)


def _defined_public_names(module):
    """Public names whose objects are defined in the given module itself
    (not imported into it from elsewhere)."""
    return [
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == module.__name__
    ]


def test_facade_reexports_every_public_name():
    # catches facade drift: a public function/class added to a submodule
    # but not re-exported by the facade fails here
    for module in SUBMODULES:
        for name in _defined_public_names(module):
            assert getattr(preprocess, name, None) is getattr(module, name), (
                f"{module.__name__}.{name} is not re-exported by the preprocess facade"
            )


def test_facade_reexports_module_constants():
    # constants have no __module__ attribute, so list them explicitly
    assert preprocess.T is converters.T
    assert preprocess.warn_positive_resolution is regrid.warn_positive_resolution
    assert preprocess.CRS == nuts_aggregation.CRS
    assert preprocess.StepFn == pipeline.StepFn


def test_facade_shares_step_registry():
    # registering a step through the facade must be visible to the pipeline
    assert preprocess._STEP_REGISTRY is pipeline._STEP_REGISTRY
    assert preprocess.register_step is pipeline.register_step
