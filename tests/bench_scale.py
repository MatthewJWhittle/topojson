"""Scale-ladder Topology benchmarks (fire), beside the in-tree chart harness.

    python tests/bench_scale.py download [--fast] [--raster_source=chesapeake]
    python tests/bench_scale.py run [--fast] [--family=long_rings] [--profile]

Downloads use pooch. Rows append to tests/bench_cache/runs/summary.jsonl.
Smooth is recorded as smooth_not_implemented (absent in 2.2).
Dataset / family specs live in the Project docs, not here.
"""

from __future__ import annotations

import cProfile
import json
import pstats
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import sys

import fire
import geopandas as gpd
import numpy as np

import topojson
from topojson.ops import arc_coordinates

# Allow `python tests/bench_scale.py` to import the sibling module.
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from bench_prep import (  # noqa: E402
    CHESAPEAKE_KEYS,
    CORE_KEYS,
    cache_dir,
    fetch_top10nl,
    prepare_step,
    reproject_for_metres,
    retrieve,
    size_steps,
)

# PR / cloud-agent allowlist (see Project docs/bench-ci-loop.md).
FAST_SIZE_IDS = {
    "ne110m_full",
    "ne10m_longest_rings_80k",
    "rivers_20k",
    "rivers_80k",
    "dc_blocks_quarter",
    "dc_blocks_half",
    "counties_20m",
}

SMOOTH = "smooth_not_implemented"


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _arc_vertices(arcs, transform=None) -> int:
    if not arcs:
        return 0
    try:
        decoded = arc_coordinates(arcs, transform)
        return int(sum(len(np.asarray(a)) for a in decoded))
    except Exception:  # noqa: BLE001
        return int(sum(len(np.asarray(a)) for a in arcs))


def _stage_from_profile(profiler: cProfile.Profile, wall_s: float) -> dict[str, float]:
    stats = pstats.Stats(profiler)
    extract = join = 0.0
    for (_fn, _ln, name), (_cc, _nc, _tt, cum, _c) in stats.stats.items():
        if name == "_extractor":
            extract = max(extract, cum)
        elif name == "_joiner":
            join = max(join, cum)
    tagged = extract + join
    if tagged > wall_s and tagged > 0:
        scale = wall_s / tagged
        extract *= scale
        join *= scale
    return {
        "t_extract_s": round(extract, 6),
        "t_join_s": round(join, 6),
        "t_other_s": round(max(0.0, wall_s - extract - join), 6),
    }


def measure(
    gdf: gpd.GeoDataFrame,
    *,
    family: str,
    dataset: str,
    size_id: str,
    prequantize: bool,
    profile: bool = False,
    simplify_m: float | None = None,
    simplify_with: str = "shapely",
) -> dict[str, Any]:
    """Wall/process/RSS + to_json; optional cProfile stage buckets. No monkeypatch."""
    n_features = int(len(gdf))
    from shapely import get_num_coordinates, get_parts

    n_parts = n_vertices = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        n_parts += len(get_parts(geom))
        n_vertices += int(get_num_coordinates(geom))

    row: dict[str, Any] = {
        "family": family,
        "dataset": dataset,
        "size_id": size_id,
        "crs": gdf.crs.to_string() if gdf.crs is not None else None,
        "prequantize": prequantize,
        "n_features": n_features,
        "n_parts": n_parts,
        "n_vertices_in": n_vertices,
        "n_junctions": None,  # not on Topology output after hashmap
        "topojson_version": getattr(topojson, "__version__", None),
        "smooth": SMOOTH,
        "failed": False,
        "fail_stage": None,
    }

    profiler = cProfile.Profile() if profile else None
    try:
        wall0, proc0 = time.perf_counter(), time.process_time()
        if profiler:
            profiler.enable()
        topo = topojson.Topology(gdf, prequantize=prequantize)
        if profiler:
            profiler.disable()
        wall_s = time.perf_counter() - wall0
        proc_s = time.process_time() - proc0
    except MemoryError:
        row.update(
            failed=True,
            fail_stage="memory",
            wall_s=round(time.perf_counter() - wall0, 6),
            process_s=round(time.process_time() - proc0, 6),
            rss_peak_mb=round(_rss_mb(), 3),
        )
        return row
    except Exception:  # noqa: BLE001
        row.update(
            failed=True,
            fail_stage="extract",
            error=traceback.format_exc(limit=3),
            wall_s=round(time.perf_counter() - wall0, 6),
            process_s=round(time.process_time() - proc0, 6),
            rss_peak_mb=round(_rss_mb(), 3),
        )
        return row

    arcs = topo.output.get("arcs") or []
    row.update(
        {
            "n_arcs": len(arcs),
            "n_vertices_out": _arc_vertices(arcs, topo.output.get("transform")),
            "wall_s": round(wall_s, 6),
            "process_s": round(proc_s, 6),
            "rss_peak_mb": round(_rss_mb(), 3),
            "rss_after_topology_mb": round(_rss_mb(), 3),
        }
    )
    if profiler:
        row.update(_stage_from_profile(profiler, wall_s))
    else:
        row.update(t_extract_s=None, t_join_s=None, t_other_s=None)

    try:
        j0 = time.perf_counter()
        payload = topo.to_json()
        j_wall = time.perf_counter() - j0
        raw = payload if isinstance(payload, str) else json.dumps(topo.to_dict())
        row["json_wall_s"] = round(j_wall, 6)
        row["t_json_s"] = row["json_wall_s"]
        row["json_bytes"] = len(raw.encode("utf-8"))
        row["rss_after_json_mb"] = round(_rss_mb(), 3)
    except Exception:  # noqa: BLE001
        row.update(failed=True, fail_stage="json", error=traceback.format_exc(limit=3))
        return row

    if simplify_m is not None:
        try:
            s0 = time.perf_counter()
            simplified = topo.toposimplify(simplify_m, simplify_with=simplify_with)
            s_wall = time.perf_counter() - s0
            s_arcs = simplified.output.get("arcs") or []
            row.update(
                simplify_with=simplify_with,
                simplify_tolerance_m=simplify_m,
                n_vertices_simplified=_arc_vertices(
                    s_arcs, simplified.output.get("transform")
                ),
                simplify_wall_s=round(s_wall, 6),
            )
        except Exception:  # noqa: BLE001
            row["simplify_error"] = traceback.format_exc(limit=2)

    row["smooth"] = SMOOTH
    return row


def append_jsonl(row: dict[str, Any], dest: Path) -> Path:
    path = dest / "runs" / "summary.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def download(
    cache: str | None = None,
    fast: bool = False,
    raster_source: str = "chesapeake",
    skip_tracts: bool = False,
    skip_rasters: bool = False,
    skip_top10nl: bool = False,
) -> None:
    """Fetch datasets into the pooch cache. --fast: small zips only."""
    dest = cache_dir(cache)
    keys = list(CORE_KEYS)
    if fast or skip_tracts:
        keys = [k for k in keys if k != "cb_2024_us_tract_500k"]
    if fast:
        # counties 500k not in FAST_SIZE_IDS
        keys = [k for k in keys if k != "cb_2024_us_county_500k"]
        skip_rasters = True
        skip_top10nl = True
    for key in keys:
        print(f"pooch {key}")
        retrieve(key, dest)
    if not skip_rasters:
        if raster_source == "landcover_ai":
            print("pooch landcover_ai_v1")
            retrieve("landcover_ai_v1", dest)
        else:
            for key in CHESAPEAKE_KEYS:
                print(f"pooch {key}")
                retrieve(key, dest)
    if not skip_top10nl:
        for bbox in ("bbox_small", "bbox_mid", "bbox_wide"):
            print(f"PDOK terrein_vlak {bbox}")
            fetch_top10nl(dest, "terrein_vlak", bbox)
        print("PDOK gebouw_vlak bbox_wide")
        fetch_top10nl(dest, "gebouw_vlak", "bbox_wide")
    print(f"cache ready under {dest}")


def run(
    cache: str | None = None,
    fast: bool = False,
    family: str | None = None,
    size_id: str | None = None,
    prequantize: str = "both",
    raster_source: str = "chesapeake",
    profile: bool = False,
    simplify: bool = False,
    max_steps: int | None = None,
    download_first: bool = False,
) -> None:
    """Prepare size steps and time Topology; append jsonl rows."""
    dest = cache_dir(cache)
    if download_first:
        download(
            cache=str(dest),
            fast=fast,
            raster_source=raster_source,
            skip_tracts=fast,
            skip_rasters=fast,
            skip_top10nl=fast,
        )

    steps = size_steps(raster_source=raster_source)
    if fast:
        steps = [s for s in steps if s["size_id"] in FAST_SIZE_IDS]
    if family:
        steps = [s for s in steps if s["family"] == family]
    if size_id:
        steps = [s for s in steps if s["size_id"] == size_id]
    if max_steps is not None:
        steps = steps[:max_steps]
    if not steps:
        raise SystemExit("No size steps selected")

    if prequantize == "true":
        pq_values = [True]
    elif prequantize == "false":
        pq_values = [False]
    else:
        pq_values = [True, False]

    n_ok = 0
    for step in steps:
        print(f"prepare {step['size_id']} ({step['family']})")
        try:
            gdf, meta = prepare_step(step, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  prepare failed: {exc}")
            row = {
                "family": step["family"],
                "dataset": step["dataset"],
                "size_id": step["size_id"],
                "failed": True,
                "fail_stage": "prepare",
                "error": str(exc),
                "smooth": SMOOTH,
            }
            append_jsonl(row, dest)
            continue

        gdf = reproject_for_metres(
            gdf, dataset=step["dataset"], family=step["family"]
        )
        print(
            f"  features={meta.get('n_features')} "
            f"vertices={meta.get('n_vertices')} crs={gdf.crs}"
        )
        for pq in pq_values:
            if (not pq) and step.get("skip_pq_false"):
                print(f"  skip prequantize=False for {step['size_id']}")
                continue
            print(f"  Topology prequantize={pq}")
            row = measure(
                gdf,
                family=step["family"],
                dataset=step["dataset"],
                size_id=step["size_id"],
                prequantize=pq,
                profile=profile,
            )
            append_jsonl(row, dest)
            status = "FAIL" if row.get("failed") else "ok"
            print(
                f"    {status} wall={row.get('wall_s')}s arcs={row.get('n_arcs')} "
                f"-> summary.jsonl"
            )
            if not row.get("failed"):
                n_ok += 1
            if simplify and step.get("simplify_m") and not row.get("failed"):
                for tol in step["simplify_m"]:
                    srow = measure(
                        gdf,
                        family=step["family"],
                        dataset=step["dataset"],
                        size_id=step["size_id"],
                        prequantize=pq,
                        profile=profile,
                        simplify_m=tol,
                    )
                    append_jsonl(srow, dest)

    print(f"Finished {n_ok} Topology runs; jsonl under {dest / 'runs'}")


if __name__ == "__main__":
    fire.Fire({"download": download, "run": run})
