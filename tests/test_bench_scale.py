"""Smoke tests for the thin scale-bench scripts (no network)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from tests.bench_prep import cumulative_by_vertices, feature_fraction, size_steps
from tests.bench_scale import FAST_SIZE_IDS, append_jsonl, measure


FILES = Path(__file__).resolve().parent / "files_shapefile"


def test_size_steps_default_chesapeake():
    steps = size_steps()
    assert any(s["family"] == "raster_landcover" for s in steps)
    assert all("landcover_ai" not in s["size_id"] for s in steps)
    assert any("chesapeake" in s["size_id"] for s in steps)
    ai = size_steps(raster_source="landcover_ai")
    assert any("landcover_ai" in s["size_id"] for s in ai)


def test_fast_allowlist_is_subset():
    ids = {s["size_id"] for s in size_steps()}
    assert FAST_SIZE_IDS <= ids


def test_prepare_helpers_on_in_tree_natural_earth():
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    subset = feature_fraction(gdf, 0.25)
    assert 0 < len(subset) < len(gdf)
    cum = cumulative_by_vertices(gdf, 2_000)
    assert 1 <= len(cum) <= len(gdf)


def test_measure_row_fields_jsonl(tmp_path):
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    row = measure(
        gdf,
        family="long_rings",
        dataset="static_natural_earth",
        size_id="smoke",
        prequantize=True,
        profile=False,
    )
    assert row["failed"] is False
    assert row["smooth"] == "smooth_not_implemented"
    assert row["n_arcs"] > 0
    assert row["json_bytes"] > 0
    assert row["t_extract_s"] is None  # profile off
    assert row["n_junctions"] is None  # no monkeypatch
    path = append_jsonl(row, tmp_path)
    assert path.name == "summary.jsonl"
    assert path.read_text().strip()


def test_measure_profile_fills_stages():
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    row = measure(
        gdf,
        family="long_rings",
        dataset="static_natural_earth",
        size_id="smoke_profile",
        prequantize=True,
        profile=True,
    )
    assert row["failed"] is False
    assert row["t_extract_s"] is not None
    assert row["t_join_s"] is not None
    assert row["t_other_s"] is not None


@pytest.mark.parametrize("prequantize", [True, False])
def test_measure_mesh_prequantize_toggle(prequantize):
    gdf = gpd.read_file(
        Path(__file__).resolve().parent / "files_geojson" / "mesh2d.geojson"
    )
    row = measure(
        gdf,
        family="shared_mesh",
        dataset="mesh2d",
        size_id="mesh_smoke",
        prequantize=prequantize,
    )
    assert row["failed"] is False
    assert row["wall_s"] >= 0
