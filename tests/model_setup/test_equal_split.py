"""
Equal autosplit planner (exllamav3/model/model_equal_split.py): the pure split arithmetic, on synthetic layer
sizes. The module is loaded straight from its file with the package stubbed, so the test needs neither a GPU
nor the compiled extension.
"""

import importlib.util
import itertools
import pathlib
import random
import sys
import types

import pytest

pytestmark = pytest.mark.nogpu

MiB = 1024 ** 2


def _load():
    root = pathlib.Path(__file__).resolve().parents[2] / "exllamav3"
    for name in ("exllamav3", "exllamav3.model"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
    if "exllamav3.constants" not in sys.modules:
        const = types.ModuleType("exllamav3.constants")
        const.PAGE_SIZE = 256
        sys.modules["exllamav3.constants"] = const
    spec = importlib.util.spec_from_file_location(
        "exllamav3.model._equal_split_under_test", root / "model" / "model_equal_split.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


es = _load()


def _margins(assignment, statics, peaks, usable):
    n = len(usable)
    s, p = [0] * n, [0] * n
    for d, gs, gp in zip(assignment, statics, peaks):
        s[d] += gs
        p[d] = max(p[d], gp)
    return {d: usable[d] - s[d] - p[d] for d in set(assignment)}


def _brute_force(statics, peaks, usable):
    """Best max-min margin over every contiguous, in-order split (devices may be skipped)"""
    n_g, n_d = len(statics), len(usable)
    best = None
    for seq in itertools.combinations_with_replacement(range(n_d), n_g):
        m = min(_margins(seq, statics, peaks, usable).values())
        best = m if best is None else max(best, m)
    return best


def test_even_split_on_equal_devices():
    statics = [400 * MiB] * 64
    peaks = [150 * MiB] * 64
    usable = [7600 * MiB] * 8
    assignment, achieved = es.plan_split(statics, peaks, usable)
    counts = [assignment.count(d) for d in range(8)]
    assert counts == [8] * 8
    assert achieved == 7600 * MiB - 8 * 400 * MiB - 150 * MiB
    assert assignment == sorted(assignment)


def test_device_with_draft_fills_down_to_the_same_margin():
    # GPU0 already holds a 4.3 GB draft. With a heavy model it still takes layers, down to the free
    # margin of the others (within one layer)
    statics = [600 * MiB] * 64
    peaks = [150 * MiB] * 64
    usable = [3300 * MiB] + [7600 * MiB] * 7
    assignment, achieved = es.plan_split(statics, peaks, usable)
    margins = _margins(assignment, statics, peaks, usable)
    assert assignment.count(0) > 0
    assert max(margins.values()) - min(margins.values()) <= 600 * MiB
    assert achieved == min(margins.values())


def test_device_with_draft_left_alone_when_already_fuller():
    # With a light model the other GPUs keep more free VRAM than GPU0 has to begin with: putting
    # layers on GPU0 would only make the fullest device fuller
    statics = [400 * MiB] * 64
    peaks = [150 * MiB] * 64
    usable = [3300 * MiB] + [7600 * MiB] * 7
    assignment, achieved = es.plan_split(statics, peaks, usable)
    assert assignment.count(0) == 0
    assert achieved >= 3300 * MiB


def test_peaks_are_a_max_not_a_sum():
    # Ten layers with a large transient each fit on one device: transients don't stack
    statics = [100 * MiB] * 10
    peaks = [2000 * MiB] * 10
    assignment, achieved = es.plan_split(statics, peaks, [4000 * MiB])
    assert achieved == 4000 * MiB - 1000 * MiB - 2000 * MiB


@pytest.mark.parametrize("seed", range(200))
def test_matches_brute_force(seed):
    rng = random.Random(seed)
    n_g = rng.randint(1, 7)
    n_d = rng.randint(1, 3)
    statics = [rng.randint(0, 1000) for _ in range(n_g)]
    peaks = [rng.randint(0, 300) for _ in range(n_g)]
    # A large last group, like an LM head
    if n_g > 1 and rng.random() < 0.3:
        statics[-1] = rng.randint(1000, 3000)
    usable = [rng.randint(0, 4000) for _ in range(n_d)]
    assignment, achieved = es.plan_split(statics, peaks, usable)
    assert assignment == sorted(assignment)
    assert achieved == min(_margins(assignment, statics, peaks, usable).values())
    assert achieved == _brute_force(statics, peaks, usable)


def test_max_context_is_the_largest_fitting_size():
    fixed = [350 * MiB] * 64
    kv = [0 if i % 4 else 256 * 1024 for i in range(64)]   # 16 full-attention layers, 256 KiB/token each
    peaks = [150 * MiB] * 64
    usable = [3300 * MiB] + [7600 * MiB] * 7
    margin = 1024 * MiB
    tokens, achieved = es.max_context_tokens(fixed, kv, peaks, usable, margin, 262144)
    assert tokens % 256 == 0 and 0 < tokens <= 262144
    assert achieved >= margin

    def margin_at(t):
        return es.plan_split([int(f + k * t) for f, k in zip(fixed, kv)], peaks, usable)[1]

    assert margin_at(tokens) >= margin
    if tokens < 262144:
        assert margin_at(tokens + 256) < margin


def test_max_context_capped_and_zero():
    tokens, _ = es.max_context_tokens([MiB], [1.0], [0], [8000 * MiB], 0, 4096)
    assert tokens == 4096
    tokens, _ = es.max_context_tokens([9000 * MiB], [1.0], [0], [8000 * MiB], 0, 4096)
    assert tokens == 0


def test_usable_bytes_proportions():
    gib = 1024 ** 3
    free = [6 * gib, 6 * gib, 6 * gib]
    total = [8 * gib, 8 * gib, 8 * gib]
    assert es.usable_bytes(free, total, [1, 1, 1]) == free
    u = es.usable_bytes(free, total, [2, 1, 0])
    assert u[0] == 6 * gib
    assert u[1] == 2 * gib      # may fill half the card; 2 GiB of it are already in use
    assert u[2] == 0


def test_suggested_context():
    e = es.EqualSplitHeadroomError("x", {0: 300 * MiB, 3: 100 * MiB}, kv_bytes_per_token = 64 * 1024,
                                   num_devices = 8)
    # 300 MiB on the worst device, spread over 8 devices: 2400 MiB of K/V = 38400 tokens
    assert e.suggested_context(262144) == 262144 - 38400
    assert e.suggested_context(262144) % 256 == 0
    e0 = es.EqualSplitHeadroomError("x", {0: 1}, kv_bytes_per_token = 0, num_devices = 8)
    assert e0.suggested_context(4096) == 3840


def test_margin_bytes_number_or_list():
    # One number applies to every active device; a list is indexed by GPU like the proportions
    assert es.margin_bytes(1024, [1, 1, 0, 1], [0, 1, 3]) == [1024 * MiB] * 3
    assert es.margin_bytes(512.5, [1, 1], [0, 1]) == [int(512.5 * MiB)] * 2
    assert es.margin_bytes([100, 200, 300, 400], [1, 1, 0, 1], [0, 1, 3]) == [100 * MiB, 200 * MiB, 400 * MiB]
    with pytest.raises(AssertionError):
        es.margin_bytes([100, 200], [1, 1, 1], [0, 1, 2])


def test_per_device_margin_shifts_layers_off_the_device_with_the_larger_margin():
    # Same usable VRAM everywhere; GPU 1 asks for 2 GiB more margin, so it must end up with about 2 GiB
    # more free VRAM than the others (within one layer), and every device keeps at least its margin
    rng = random.Random(7)
    statics = [rng.randint(300, 420) * MiB for _ in range(40)]
    peaks = [rng.randint(20, 160) * MiB for _ in range(40)]
    usable = [7600 * MiB] * 4
    margin = [512 * MiB, 2560 * MiB, 512 * MiB, 512 * MiB]
    budget = [u - m for u, m in zip(usable, margin)]
    assignment, spare = es.plan_split(statics, peaks, budget)
    free = _margins(assignment, statics, peaks, usable)
    assert spare >= 0
    for d, f in free.items():
        assert f >= margin[d]
    others = [f for d, f in free.items() if d != 1]
    assert free[1] - min(others) >= 2048 * MiB - 420 * MiB
    # Exact on a size brute force can check: no contiguous split leaves more spare beyond the margins
    small = slice(0, 10)
    _, spare_small = es.plan_split(statics[small], peaks[small], budget)
    assert spare_small == _brute_force(statics[small], peaks[small], budget)


def test_per_device_margin_max_context():
    fixed = [350 * MiB] * 32
    kv = [64 * 1024 / 16] * 32
    peaks = [100 * MiB] * 32
    usable = [7600 * MiB] * 4
    margin = [1024 * MiB, 1024 * MiB, 3072 * MiB, 1024 * MiB]
    budget = [u - m for u, m in zip(usable, margin)]
    tokens, spare = es.max_context_tokens(fixed, kv, peaks, budget, 0, 1 << 20)
    tokens_even, _ = es.max_context_tokens(fixed, kv, peaks, [u - 1024 * MiB for u in usable], 0, 1 << 20)
    assert spare >= 0
    assert 0 < tokens < tokens_even
