import random

from evotailor.spaces.base import Dimension, SearchSpace


def _space():
    return SearchSpace.from_dimensions(
        [
            Dimension("runtime.gpu_memory_utilization", "float", (0.3, 0.95)),
            Dimension("runtime.max_num_seqs", "int", (8, 256, 8)),
            Dimension("runtime.enable_prefix_caching", "bool"),
        ],
        constraints=[(lambda c: c["runtime.max_num_seqs"] <= 64
                      or c["runtime.gpu_memory_utilization"] >= 0.6,
                      "many sequences require a larger KV pool")],
        framework="vllm",
    )


def test_samples_are_admissible():
    space = _space()
    rng = random.Random(0)
    for _ in range(200):
        candidate = space.sample(rng)
        ok, _ = space.admit(candidate)
        assert ok


def test_integer_dimension_respects_step():
    dim = Dimension("runtime.max_num_seqs", "int", (8, 256, 8))
    rng = random.Random(1)
    for _ in range(50):
        assert dim.sample(rng) % 8 == 0


def test_evolve_drops_and_narrows():
    space = _space()
    evolved = space.evolve(drop={"runtime.max_num_seqs"},
                           narrow={"runtime.gpu_memory_utilization": (0.6, 0.9)},
                           reason="tighter memory range")
    assert "runtime.max_num_seqs" not in evolved
    assert evolved.dimension("runtime.gpu_memory_utilization").domain == (0.6, 0.9)
    assert evolved.stage_index == space.stage_index + 1
