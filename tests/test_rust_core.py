"""Optional topojson._core (Rust/PyO3) oracle tests — skip when not built."""

import numpy as np
import pytest

from topojson import ops

pytestmark = pytest.mark.skipif(
    not ops.rust_core_available(),
    reason="topojson._core not built (maturin develop -m rust/Cargo.toml)",
)


@pytest.fixture
def core():
    from topojson import _core

    return _core


def test_core_info(core):
    assert "topojson._core" in core.core_info()


@pytest.mark.parametrize("n,iterations", [(2, 2), (3, 1), (3, 2), (8, 3)])
def test_chaikin_matches_numpy(core, n, iterations):
    rng = np.random.default_rng(n * 10 + iterations)
    xy = rng.normal(size=(n, 2))
    got = np.asarray(core.chaikin(xy, iterations), dtype=np.float64)
    # Force NumPy path without disturbing other tests' import cache permanently.
    prev, prev_attempted = ops._RUST, ops._RUST_LOAD_ATTEMPTED
    try:
        ops._RUST = None
        ops._RUST_LOAD_ATTEMPTED = True
        expect = ops.chaikin(xy, iterations)
    finally:
        ops._RUST = prev
        ops._RUST_LOAD_ATTEMPTED = prev_attempted
    assert got.shape == expect.shape
    np.testing.assert_allclose(got, expect, atol=1e-12)


def test_cut_lines_on_grid_parts(core):
    # Two unit squares sharing an edge → one shared border cut at junctions.
    xy = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    li = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    jxy = np.array([[1, 0], [1, 1]], dtype=np.int64)
    is_ring = np.array([True, True])
    parts, offsets, n_parts = core.cut_lines_on_grid(xy, li, jxy, is_ring)
    assert int(n_parts.sum()) == len(offsets) - 1
    assert len(parts) == int(offsets[-1])
