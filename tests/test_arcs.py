"""Composable ArcSurface chaining and Topology.simplify / smooth helpers."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import geometry

import topojson
from topojson import ArcSurface


def _zigzag():
    return geometry.LineString([(i, (i % 2) * 0.5) for i in range(20)])


def test_arcs_surface_from_topology():
    topo = topojson.Topology([_zigzag()], prequantize=False)
    surface = topo.arcs()
    assert isinstance(surface, ArcSurface)
    assert len(surface) == len(topo.output["arcs"])


def test_chain_simplify_segmentize_smooth_commit():
    topo = topojson.Topology([_zigzag()], prequantize=True)
    assert "transform" in topo.output
    out = topo.arcs().simplify(0.2).segmentize(0.1).smooth(iterations=2).commit()
    assert "transform" not in out.output
    assert out.options.toposimplify
    assert out.options.toposmooth == "chaikin"
    assert sum(len(a) for a in out.output["arcs"]) > 0
    # Ends of the open arc stay pinned through the chain.
    before = np.asarray(topo.arcs().coordinates[0])
    after = np.asarray(out.arcs().coordinates[0])
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[-1], before[-1])


def test_topology_simplify_smooth_helpers_chain():
    topo = topojson.Topology([_zigzag()], prequantize=False)
    out = topo.simplify(0.1).smooth(iterations=2)
    assert out.options.toposimplify
    assert out.options.toposmooth == "chaikin"
    assert sum(len(a) for a in out.output["arcs"]) > 2


def test_toposmooth_is_alias_of_smooth():
    topo = topojson.Topology([_zigzag()], prequantize=False)
    a = topo.smooth(iterations=2).to_dict()
    b = topo.toposmooth(iterations=2).to_dict()
    assert a["arcs"] == b["arcs"]


def test_map_custom_arc_fn():
    topo = topojson.Topology([_zigzag()], prequantize=False)

    def shift(xy):
        out = xy.copy()
        out[1:-1, 1] += 0.01
        return out

    out = topo.arcs().map(shift).commit()
    before = np.asarray(topo.output["arcs"][0], dtype=float)
    after = np.asarray(out.output["arcs"][0], dtype=float)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[-1], before[-1])
    assert not np.allclose(after[1:-1], before[1:-1])


def test_commit_without_topology_raises():
    surface = ArcSurface([np.array([[0.0, 0.0], [1.0, 0.0]])])
    with pytest.raises(ValueError, match="topology"):
        surface.commit()
