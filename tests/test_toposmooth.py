"""Tests for Topology.toposmooth (arc-wise Chaikin / Catmull-Rom)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import geometry

import topojson
from topojson.ops import catmull_rom, chaikin


def _two_squares():
    """Adjacent unit squares sharing the edge x=1 from y=0 to y=1."""
    return [
        geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        geometry.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]


def test_chaikin_pins_endpoints_and_grows():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    out = chaikin(xy, iterations=2)
    assert out.shape[0] == 4 * len(xy)  # 2x per iteration
    np.testing.assert_allclose(out[0], xy[0])
    np.testing.assert_allclose(out[-1], xy[-1])


def test_chaikin_two_point_arc_unchanged():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = chaikin(xy, iterations=3)
    np.testing.assert_array_equal(out, xy)


def test_segmentize_caps_edge_length_and_pins_ends():
    from topojson.ops import segmentize

    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    out = segmentize(xy, 2.5)
    assert len(out) == 5  # 0, 2.5, 5, 7.5, 10
    np.testing.assert_allclose(out[0], xy[0])
    np.testing.assert_allclose(out[-1], xy[-1])
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert step.max() <= 2.5 + 1e-9


def test_toposmooth_segment_length_densifies_before_chaikin():
    zigzag = geometry.LineString([(0, 0), (10, 0), (20, 1), (30, 0)])
    topo = topojson.Topology([zigzag], prequantize=False)
    plain = topo.toposmooth(iterations=2)
    densified = topo.toposmooth(iterations=2, segment_length=2.0)
    n_plain = sum(len(a) for a in plain.output["arcs"])
    n_dense = sum(len(a) for a in densified.output["arcs"])
    assert n_dense > n_plain
    # Ends of the single open arc stay pinned.
    before = np.asarray(topo.output["arcs"][0], dtype=float)
    after = np.asarray(densified.output["arcs"][0], dtype=float)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[-1], before[-1])


def test_catmull_rom_passes_through_controls():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.5]])
    out = catmull_rom(xy, samples=4)
    # Every original vertex remains (endpoints + interiors at segment ends).
    for p in xy:
        assert np.any(np.all(np.isclose(out, p), axis=1))
    np.testing.assert_allclose(out[0], xy[0])
    np.testing.assert_allclose(out[-1], xy[-1])


def test_toposmooth_shared_edge_stays_coincident():
    topo = topojson.Topology(_two_squares(), prequantize=False)
    # Densify the shared edge so Chaikin has corners to cut.
    arcs = topo.output["arcs"]
    densified = []
    for arc in arcs:
        a = np.asarray(arc, dtype=float)
        if len(a) == 2 and np.isclose(a[0, 0], 1) and np.isclose(a[1, 0], 1):
            ys = np.linspace(a[0, 1], a[1, 1], 5)
            densified.append(np.column_stack([np.full(5, 1.0), ys]).tolist())
        else:
            densified.append(a.tolist())
    topo.output["arcs"] = densified

    smooth = topo.toposmooth("chaikin", iterations=2)
    gdf = smooth.to_gdf()
    shared = gdf.geometry.iloc[0].intersection(gdf.geometry.iloc[1])
    assert shared.length > 0
    # No polygonal overlap / gap: intersection is a line (shared boundary).
    assert shared.geom_type in ("LineString", "MultiLineString", "GeometryCollection")
    assert gdf.geometry.iloc[0].intersects(gdf.geometry.iloc[1])
    assert (
        not gdf.geometry.iloc[0]
        .buffer(-1e-9)
        .intersects(gdf.geometry.iloc[1].buffer(-1e-9))
    )


def test_toposmooth_drops_transform_no_requantize():
    topo = topojson.Topology(_two_squares(), prequantize=True)
    assert "transform" in topo.output
    smooth = topo.toposmooth()
    assert "transform" not in smooth.output
    assert smooth.options.toposmooth == "chaikin"
    # Float arcs (not delta-encoded ints).
    assert isinstance(smooth.output["arcs"][0][0][0], float)


def test_toposmooth_endpoint_pin_on_topology_arcs():
    data = [
        geometry.LineString([(0, 0), (0.5, 0.2), (1, 0), (1.5, -0.2), (2, 0)]),
    ]
    topo = topojson.Topology(data, prequantize=False)
    before = np.asarray(topo.output["arcs"][0], dtype=float)
    after = np.asarray(topo.toposmooth().output["arcs"][0], dtype=float)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[-1], before[-1])


def test_toposmooth_growth_bound_chaikin2():
    zigzag = geometry.LineString([(i, (i % 2) * 0.5) for i in range(20)])
    topo = topojson.Topology([zigzag], prequantize=False)
    n0 = sum(len(a) for a in topo.output["arcs"])
    n1 = sum(len(a) for a in topo.toposmooth(iterations=2).output["arcs"])
    # ~4x for arcs with >=3 vertices; allow small headroom.
    assert n1 <= 4 * n0 + 8
    assert n1 > n0


def test_toposmooth_catmull_rom_chain():
    topo = topojson.Topology(_two_squares(), prequantize=False).toposimplify(0.01)
    smooth = topo.toposmooth("catmull_rom", samples=4)
    assert smooth.options.toposmooth == "catmull_rom"
    assert sum(len(a) for a in smooth.output["arcs"]) >= sum(
        len(a) for a in topo.output["arcs"]
    )


def test_toposmooth_invalid_method_and_caps():
    topo = topojson.Topology(_two_squares(), prequantize=False)
    with pytest.raises(ValueError, match="method"):
        topo.toposmooth("gaussian")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="iterations"):
        topo.toposmooth(iterations=6)
    with pytest.raises(ValueError, match="samples"):
        topo.toposmooth("catmull_rom", samples=-1)
    with pytest.raises(ValueError, match="segment_length"):
        topo.toposmooth(segment_length=0)


def test_toposmooth_inplace_and_incremental_refused():
    topo = topojson.Topology(_two_squares(), prequantize=True)
    topo.toposmooth(inplace=True)
    assert topo.options.toposmooth == "chaikin"
    with pytest.raises(NotImplementedError, match="unsimplified"):
        topo.remove([0])
