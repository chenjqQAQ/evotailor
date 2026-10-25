import random

import pytest

from evotailor.spaces.builder import adapt_space
from evotailor.spaces.search_space import Dimension, SearchSpace


def space():
    return SearchSpace(dimensions={
        "memory": Dimension(kind="float", lower=0.3, upper=0.9, step=0.1),
        "batch": Dimension(kind="integer", lower=8, upper=64, step=8),
    }, safe_baseline={"memory": 0.6, "batch": 8})


def test_samples_and_normalization_obey_the_frozen_domain():
    current = space()
    for index in range(200):
        candidate = current.sample(random.Random(index))
        assert not current.validate_candidate(candidate)
        assert candidate["batch"] % 8 == 0
    assert current.normalize({"memory": 0.61, "batch": 9}) == {"memory": 0.6, "batch": 8}


def test_boundary_can_narrow_and_remove_without_mutating_old_space():
    current = space()
    original_hash = current.digest
    evolved = adapt_space(current, {
        "retune": {"memory": {"kind": "float", "lower": 0.7, "upper": 0.9, "step": 0.1}},
        "remove": ["batch"], "evidence": {"memory": "measured.fact", "batch": "measured.fact"},
    }, available_evidence={"measured.fact"})
    assert current.digest == original_hash
    assert "batch" in current.dimensions and "batch" not in evolved.dimensions
    assert evolved.deployment_parameters({"memory": 0.8}) == {"memory": 0.8, "batch": 8}
    assert evolved.dimensions["memory"].values() == [0.7, 0.8, 0.9]


def test_unjustified_changes_are_rejected():
    with pytest.raises(ValueError, match="evidence"):
        adapt_space(space(), {"remove": ["batch"]}, available_evidence=set())


def test_freeze_refuses_redefinition_of_an_active_stage(tmp_path):
    current = space()
    frozen = tmp_path / "frozen.json"
    current.freeze(frozen)
    changed = adapt_space(current, {"remove": ["batch"], "evidence": {"batch": "fact"}},
                          available_evidence={"fact"})
    with pytest.raises(ValueError, match="already frozen"):
        changed.freeze(frozen)
