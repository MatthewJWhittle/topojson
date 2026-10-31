"""Smoke tests for the representative benchmarks package (no network)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from benchmarks.catalog import size_steps
from benchmarks.measure import measure_topology, write_run_row
from benchmarks.prepare import cumulative_by_vertices, feature_fraction


FILES = Path(__file__).resolve().parent / "files_shapefile"


def test_size_steps_default_uses_chesapeake_not_landcover_ai():
    steps = size_steps()
    assert any(s.family == "raster_landcover" for s in steps)
    assert all("landcover_ai" not in s.size_id for s in steps)
    assert any("chesapeake" in s.size_id for s in steps)
    ai = size_steps(raster_source="landcover_ai")
    assert any("landcover_ai" in s.size_id for s in ai)


def test_prepare_helpers_on_in_tree_natural_earth():
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    subset = feature_fraction(gdf, 0.25)
    assert 0 < len(subset) < len(gdf)
    cum = cumulative_by_vertices(gdf, 2_000)
    assert len(cum) >= 1
    assert len(cum) <= len(gdf)


def test_measure_topology_row_fields(tmp_path):
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    row = measure_topology(
        gdf,
        family="long_rings",
        dataset="static_natural_earth",
        size_id="smoke",
        prequantize=True,
    )
    assert row["failed"] is False
    assert row["smooth"] == "smooth_not_implemented"
    assert row["prequantize"] is True
    assert row["n_features"] == len(gdf)
    assert row["n_vertices_in"] > 0
    assert row["n_arcs"] > 0
    assert row["wall_s"] >= 0
    assert "t_extract_s" in row and "t_join_s" in row and "t_other_s" in row
    assert row["json_bytes"] > 0
    path = write_run_row(row, cache_dir=tmp_path)
    assert path.exists()
    assert path.parent.name == "runs"


@pytest.mark.parametrize("prequantize", [True, False])
def test_measure_mesh_prequantize_toggle(prequantize):
    gdf = gpd.read_file(
        Path(__file__).resolve().parent / "files_geojson" / "mesh2d.geojson"
    )
    row = measure_topology(
        gdf,
        family="shared_mesh",
        dataset="mesh2d",
        size_id="mesh_smoke",
        prequantize=prequantize,
    )
    assert row["failed"] is False
    assert row["n_junctions"] is not None
    assert row["n_junctions"] >= 0
