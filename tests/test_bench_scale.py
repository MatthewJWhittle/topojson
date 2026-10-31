"""Smoke tests for the thin scale-bench scripts (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import geopandas as gpd
import pytest

from tests.bench_prep import (
    DEFAULT_SIMPLIFY_M,
    LIVING_ENGLAND_BBOXES,
    cumulative_by_vertices,
    feature_fraction,
    size_steps,
)
from tests.bench_scale import (
    FAST_SIZE_IDS,
    HABITAT_SIZE_IDS,
    MEDIUM_SUBSET_SIZE_IDS,
    append_jsonl,
    measure,
    measure_smooth,
    resolve_simplify_tolerances,
    simplify_epsilon_for_crs,
)

FILES = Path(__file__).resolve().parent / "files_shapefile"


def test_size_steps_default_chesapeake():
    steps = size_steps()
    assert any(s["family"] == "raster_landcover" for s in steps)
    assert all("landcover_ai" not in s["size_id"] for s in steps)
    assert any("chesapeake" in s["size_id"] for s in steps)
    assert any(s["size_id"].startswith("living_england_") for s in steps)
    ai = size_steps(raster_source="landcover_ai")
    assert any("landcover_ai" in s["size_id"] for s in ai)
    tampa = size_steps(raster_source="tampa_ccap")
    assert any(s["size_id"].startswith("tampa_ccap_") for s in tampa)
    assert all("chesapeake" not in s["size_id"] for s in tampa)


def test_fast_allowlist_is_subset():
    ids = {s["size_id"] for s in size_steps()}
    assert ids >= FAST_SIZE_IDS
    assert ids >= MEDIUM_SUBSET_SIZE_IDS
    habitat_ids = {s["size_id"] for s in size_steps(raster_source="tampa_ccap")}
    assert habitat_ids >= HABITAT_SIZE_IDS
    assert set(LIVING_ENGLAND_BBOXES) == {
        "bbox_small",
        "bbox_mid",
        "bbox_wide",
    }


def test_default_simplify_tolerances():
    assert DEFAULT_SIMPLIFY_M == [1.0, 2.0]
    assert resolve_simplify_tolerances({}, simplify=False) is None
    assert resolve_simplify_tolerances({}, simplify=True) == [1.0, 2.0]
    assert resolve_simplify_tolerances({"simplify_m": [2.0]}, simplify=True) == [2.0]


def test_simplify_epsilon_geographic_vs_projected():
    geo = SimpleNamespace(is_geographic=True)
    proj = SimpleNamespace(is_geographic=False)
    assert simplify_epsilon_for_crs(proj, 2.0) == 2.0
    assert simplify_epsilon_for_crs(geo, 2.0) == pytest.approx(2.0 / 111_320.0)


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
    assert row["smooth"] == "ok"
    assert row["smooth_method"] == "chaikin"
    assert row["smooth_iterations"] == 2
    assert row["n_vertices_smoothed"] > 0
    assert row["smooth_growth"] >= 1.0
    assert row["n_arcs"] > 0
    assert row["json_bytes"] > 0
    assert row["t_extract_s"] is None  # profile off
    assert row["n_junctions"] is None  # no monkeypatch
    assert "simplify_wall_s" not in row
    path = append_jsonl(row, tmp_path)
    assert path.name == "summary.jsonl"
    assert path.read_text().strip()


def test_measure_with_toposimplify_records_fields():
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    row = measure(
        gdf,
        family="long_rings",
        dataset="static_natural_earth",
        size_id="smoke_simplify",
        prequantize=True,
        simplify_m=[1.0, 2.0],
    )
    assert row["failed"] is False
    assert row["smooth"] == "ok"
    assert row["smooth_method"] == "chaikin"
    assert row["n_vertices_smoothed"] >= row["n_vertices_simplified"]
    assert row["simplify_tolerance_m"] == 2.0
    assert row["simplify_wall_s"] >= 0
    assert row["n_vertices_simplified"] > 0
    assert row["n_vertices_simplified"] <= row["n_vertices_out"]
    assert len(row["simplify_runs"]) == 2
    assert row["simplify_runs"][0]["simplify_tolerance_m"] == 1.0


def test_measure_smooth_hook_absent_api():
    topo = MagicMock(spec=["output"])
    topo.output = {"arcs": []}
    assert measure_smooth(topo) == {"smooth": "smooth_not_implemented"}


def test_measure_smooth_hook_with_api():
    gdf = gpd.read_file(FILES / "static_natural_earth.gpkg")
    import topojson

    topo = topojson.Topology(gdf, prequantize=True).toposimplify(1.0)
    row = measure_smooth(topo)
    assert row["smooth"] == "ok"
    assert row["smooth_wall_s"] >= 0
    assert row["smooth_growth"] >= 1.0


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
        simplify_m=2.0,
    )
    assert row["failed"] is False
    assert row["wall_s"] >= 0
    assert row["simplify_wall_s"] >= 0
    assert row["n_vertices_simplified"] > 0
