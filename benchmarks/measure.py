"""Time Topology runs and write one JSON row per run beside the cache."""

from __future__ import annotations

import cProfile
import json
import pstats
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

import topojson
from topojson.core.hashmap import Hashmap
from topojson.ops import arc_coordinates

from .cache import ensure_cache_dir, geometry_stats


SMOOTH_STATUS = "smooth_not_implemented"


def _rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _count_arc_vertices(arcs: list) -> int:
    if not arcs:
        return 0
    total = 0
    for arc in arcs:
        arr = np.asarray(arc)
        if arr.ndim == 1:
            total += len(arr)
        else:
            total += len(arr)
    return int(total)


def _profile_buckets(stats: pstats.Stats) -> dict[str, float]:
    """Stage seconds from cumtime of each pipeline entry point.

    Extract runs before Join._joiner (via super()), so those cumtimes do not
    nest. Cut/dedup/hashmap/_topo fill the residual against wall time.
    """
    extract = 0.0
    join = 0.0
    for (_filename, _line, name), (_cc, _nc, _tt, cumtime, _callers) in stats.stats.items():
        if name == "_extractor":
            extract = max(extract, cumtime)
        elif name == "_joiner":
            join = max(join, cumtime)
    return {"t_extract_s": extract, "t_join_s": join}


def _normalize_stage_times(
    buckets: dict[str, float], topology_wall_s: float
) -> dict[str, float]:
    extract = buckets.get("t_extract_s", 0.0)
    join = buckets.get("t_join_s", 0.0)
    tagged = extract + join
    if tagged > topology_wall_s and tagged > 0:
        scale = topology_wall_s / tagged
        extract *= scale
        join *= scale
    other = max(0.0, topology_wall_s - extract - join)
    return {
        "t_extract_s": round(extract, 6),
        "t_join_s": round(join, 6),
        "t_other_s": round(other, 6),
    }


def measure_topology(
    gdf: gpd.GeoDataFrame,
    *,
    family: str,
    dataset: str,
    size_id: str,
    prequantize: bool,
    simplify_with: str | None = None,
    simplify_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Run one Topology (+ optional toposimplify / to_json) and return a result row."""
    input_stats = geometry_stats(gdf)
    crs = input_stats.get("crs")
    row: dict[str, Any] = {
        "family": family,
        "dataset": dataset,
        "size_id": size_id,
        "crs": crs,
        "prequantize": prequantize,
        "n_features": input_stats["n_features"],
        "n_parts": input_stats["n_parts"],
        "n_vertices_in": input_stats["n_vertices"],
        "topojson_version": getattr(topojson, "__version__", None),
        "smooth": SMOOTH_STATUS,
        "failed": False,
        "fail_stage": None,
    }

    captured: dict[str, Any] = {"n_junctions": None}
    original_hashmapper = Hashmap._hashmapper

    def _capturing_hashmapper(self, data):  # noqa: ANN001
        captured["n_junctions"] = len(data.get("junctions") or [])
        return original_hashmapper(self, data)

    Hashmap._hashmapper = _capturing_hashmapper  # type: ignore[method-assign]
    rss_before = _rss_mb()
    profiler = cProfile.Profile()
    topo = None
    try:
        wall0 = time.perf_counter()
        proc0 = time.process_time()
        profiler.enable()
        try:
            topo = topojson.Topology(gdf, prequantize=prequantize)
        except MemoryError:
            profiler.disable()
            row["failed"] = True
            row["fail_stage"] = "memory"
            row["wall_s"] = round(time.perf_counter() - wall0, 6)
            row["process_s"] = round(time.process_time() - proc0, 6)
            row["rss_peak_mb"] = round(_rss_mb(), 3)
            return row
        except Exception:
            profiler.disable()
            row["failed"] = True
            row["fail_stage"] = "extract"
            row["error"] = traceback.format_exc(limit=3)
            row["wall_s"] = round(time.perf_counter() - wall0, 6)
            row["process_s"] = round(time.process_time() - proc0, 6)
            row["rss_peak_mb"] = round(_rss_mb(), 3)
            return row
        profiler.disable()
        wall_s = time.perf_counter() - wall0
        proc_s = time.process_time() - proc0
        rss_after = _rss_mb()

        arcs = topo.output.get("arcs") or []
        try:
            decoded = arc_coordinates(arcs, topo.output.get("transform"))
            n_vertices_out = int(sum(len(np.asarray(a)) for a in decoded))
        except Exception:  # noqa: BLE001
            n_vertices_out = _count_arc_vertices(arcs)

        row.update(
            {
                "n_arcs": len(arcs),
                "n_vertices_out": n_vertices_out,
                "n_junctions": captured["n_junctions"],
                "wall_s": round(wall_s, 6),
                "process_s": round(proc_s, 6),
                "rss_peak_mb": round(max(rss_before, rss_after, _rss_mb()), 3),
                "rss_after_topology_mb": round(rss_after, 3),
            }
        )

        buckets = _profile_buckets(pstats.Stats(profiler))
        row.update(_normalize_stage_times(buckets, wall_s))

        # Separate JSON timing after Topology has returned.
        try:
            j0 = time.perf_counter()
            payload = topo.to_json()
            j_wall = time.perf_counter() - j0
            if isinstance(payload, str):
                raw = payload
            else:
                raw = json.dumps(topo.to_dict())
            row["t_json_s"] = round(j_wall, 6)
            row["json_wall_s"] = row["t_json_s"]
            row["json_bytes"] = len(raw.encode("utf-8"))
            row["rss_after_json_mb"] = round(_rss_mb(), 3)
        except Exception:  # noqa: BLE001
            row["failed"] = True
            row["fail_stage"] = "json"
            row["error"] = traceback.format_exc(limit=3)
            return row

        if simplify_tolerance_m is not None:
            try:
                s0 = time.perf_counter()
                kwargs: dict[str, Any] = {}
                if simplify_with:
                    kwargs["simplify_with"] = simplify_with
                simplified = topo.toposimplify(simplify_tolerance_m, **kwargs)
                s_wall = time.perf_counter() - s0
                s_arcs = simplified.output.get("arcs") or []
                try:
                    s_decoded = arc_coordinates(
                        s_arcs, simplified.output.get("transform")
                    )
                    n_simp = int(sum(len(np.asarray(a)) for a in s_decoded))
                except Exception:  # noqa: BLE001
                    n_simp = _count_arc_vertices(s_arcs)
                row["simplify_with"] = simplify_with or "shapely"
                row["simplify_tolerance_m"] = simplify_tolerance_m
                row["n_vertices_simplified"] = n_simp
                row["simplify_wall_s"] = round(s_wall, 6)
            except Exception:  # noqa: BLE001
                row["simplify_error"] = traceback.format_exc(limit=2)

        # Explicit marker: 2.2 has no toposmooth.
        row["smooth"] = SMOOTH_STATUS
        return row
    finally:
        Hashmap._hashmapper = original_hashmapper  # type: ignore[method-assign]


def write_run_row(row: dict[str, Any], cache_dir: Path | None = None) -> Path:
    """Write one JSON object per run under cache/runs/."""
    cache_dir = ensure_cache_dir(cache_dir)
    runs = cache_dir / "runs"
    runs.mkdir(exist_ok=True)
    pq = "pq1" if row.get("prequantize") else "pq0"
    name = f"{row.get('family')}_{row.get('size_id')}_{pq}"
    if row.get("simplify_tolerance_m") is not None:
        name += f"_simp{row['simplify_tolerance_m']}"
    path = runs / f"{name}.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    return path


def append_summary(row: dict[str, Any], cache_dir: Path | None = None) -> Path:
    """Append the row to cache/runs/summary.jsonl for easy aggregation."""
    cache_dir = ensure_cache_dir(cache_dir)
    path = cache_dir / "runs" / "summary.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path
