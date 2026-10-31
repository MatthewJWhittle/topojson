"""Scale-ladder Topology benchmarks (fire), beside the in-tree chart harness.

    python tests/bench_scale.py download [--fast] [--raster_source=chesapeake]
    python tests/bench_scale.py run [--fast] [--family=long_rings] [--profile]
    python tests/bench_scale.py run --fast --simplify   # default on for --fast

Downloads use pooch. Rows append to tests/bench_cache/runs/summary.jsonl.
Every successful Topology run also times toposimplify when --simplify is on
(default for --fast; default on for non-fast / medium as well).
Smooth is recorded as smooth_not_implemented (absent in 2.2); measure_smooth
is the one-call hook for a future toposmooth API.
Dataset / family specs live in the Project docs, not here.
"""

from __future__ import annotations

import cProfile
import json
import pstats
import resource
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    DEFAULT_SIMPLIFY_M,
    HABITAT_VECTOR_KEYS,
    LIVING_ENGLAND_BBOXES,
    cache_dir,
    fetch_living_england,
    fetch_top10nl,
    prepare_step,
    reproject_for_metres,
    retrieve,
    size_steps,
    tampa_ccap_geotiff,
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

# Representative medium subset for simplify baselines (beside --fast).
MEDIUM_SUBSET_SIZE_IDS = {
    "dc_blocks_full",
    "counties_500k",
    "tracts_states_midatl",
    "top10nl_terrein_bbox_mid",
    "top10nl_gebouw_bbox_wide",
    "rivers_full",
    "ne10m_full",
    "chesapeake_dc_chip_1m",
}

# Living England nested bboxes + Tampa C-CAP chips (habitat stress ladder).
HABITAT_SIZE_IDS = {
    "living_england_bbox_small",
    "living_england_bbox_mid",
    "living_england_bbox_wide",
    "tampa_ccap_chip2048_2m",
    "tampa_ccap_chip4096_2m",
    "tampa_ccap_chip4096_4m",
    "tampa_ccap_chip8192_4m",
}

SMOOTH = "smooth_not_implemented"
# Approximate metres→degrees at the equator for geographic CRS tolerances.
_METRES_PER_DEGREE = 111_320.0


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _arc_vertices(arcs, transform=None) -> int:
    if not arcs:
        return 0
    try:
        decoded = arc_coordinates(arcs, transform)
        return int(sum(len(np.asarray(a)) for a in decoded))
    except Exception:
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


def simplify_epsilon_for_crs(crs, tolerance_m: float) -> float:
    """Map a metre tolerance into CRS units (degrees ≈ m / 111320 when geographic)."""
    if crs is not None and getattr(crs, "is_geographic", False):
        return tolerance_m / _METRES_PER_DEGREE
    return float(tolerance_m)


def resolve_simplify_tolerances(
    step: dict[str, Any], *, simplify: bool
) -> list[float] | None:
    """Tolerances in metres from the step, or the datasets-doc default (1–2 m)."""
    if not simplify:
        return None
    raw = step.get("simplify_m")
    if raw:
        return [float(t) for t in raw]
    return list(DEFAULT_SIMPLIFY_M)


def measure_smooth(topo: Any, *, tolerance: float | None = None) -> dict[str, Any]:
    """Hook for future Topology.toposmooth; one call site when the API lands.

    Until then every row keeps ``smooth: smooth_not_implemented``. Call after
    toposimplify so a future smooth times the simplified arcs.
    """
    smooth_fn = getattr(topo, "toposmooth", None)
    if not callable(smooth_fn):
        return {"smooth": SMOOTH}
    try:
        kwargs: dict[str, Any] = {}
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        t0 = time.perf_counter()
        smoothed = smooth_fn(**kwargs) if kwargs else smooth_fn()
        wall = time.perf_counter() - t0
        out = smoothed if smoothed is not None else topo
        arcs = getattr(out, "output", {}).get("arcs") or []
        transform = getattr(out, "output", {}).get("transform")
        return {
            "smooth": "ok",
            "smooth_wall_s": round(wall, 6),
            "n_vertices_smoothed": _arc_vertices(arcs, transform),
        }
    except TypeError:
        # Signature unknown until the API exists; keep the sentinel.
        return {"smooth": SMOOTH}
    except Exception:
        return {
            "smooth": SMOOTH,
            "smooth_error": traceback.format_exc(limit=2),
        }


def _record_toposimplify(
    row: dict[str, Any],
    topo: Any,
    *,
    tolerances_m: Sequence[float],
    simplify_with: str,
    crs,
) -> Any:
    """Time toposimplify for each metre tolerance on the same Topology object.

    Primary fields (``simplify_wall_s``, ``n_vertices_simplified``, …) come from
    the last tolerance (datasets doc uses 1–2 m; steps usually end on 2 m).
    Extra tolerances land in ``simplify_runs``. Returns the last simplified
    Topology (for the smooth hook) or ``topo`` if every attempt failed.
    """
    runs: list[dict[str, Any]] = []
    last_ok = topo
    for tol_m in tolerances_m:
        epsilon = simplify_epsilon_for_crs(crs, tol_m)
        try:
            s0 = time.perf_counter()
            simplified = topo.toposimplify(epsilon, simplify_with=simplify_with)
            s_wall = time.perf_counter() - s0
            s_arcs = simplified.output.get("arcs") or []
            entry = {
                "simplify_with": simplify_with,
                "simplify_tolerance_m": tol_m,
                "simplify_epsilon": epsilon,
                "n_vertices_simplified": _arc_vertices(
                    s_arcs, simplified.output.get("transform")
                ),
                "simplify_wall_s": round(s_wall, 6),
            }
            runs.append(entry)
            last_ok = simplified
        except Exception:
            runs.append(
                {
                    "simplify_tolerance_m": tol_m,
                    "simplify_epsilon": epsilon,
                    "simplify_error": traceback.format_exc(limit=2),
                }
            )
    ok_runs = [r for r in runs if "simplify_wall_s" in r]
    if ok_runs:
        primary = ok_runs[-1]
        row.update(
            {
                "simplify_with": primary["simplify_with"],
                "simplify_tolerance_m": primary["simplify_tolerance_m"],
                "simplify_epsilon": primary["simplify_epsilon"],
                "n_vertices_simplified": primary["n_vertices_simplified"],
                "simplify_wall_s": primary["simplify_wall_s"],
            }
        )
    elif runs:
        row["simplify_error"] = runs[-1].get("simplify_error")
    if len(runs) > 1:
        row["simplify_runs"] = runs
    return last_ok


def measure(
    gdf: gpd.GeoDataFrame,
    *,
    family: str,
    dataset: str,
    size_id: str,
    prequantize: bool,
    profile: bool = False,
    simplify_m: float | Sequence[float] | None = None,
    simplify_with: str = "shapely",
    smooth: bool = True,
) -> dict[str, Any]:
    """Wall/process/RSS + to_json; optional toposimplify on the same Topology."""
    n_features = len(gdf)
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
    except Exception:
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
    except Exception:
        row.update(failed=True, fail_stage="json", error=traceback.format_exc(limit=3))
        return row

    target = topo
    if simplify_m is not None:
        tols = (
            [float(simplify_m)]
            if isinstance(simplify_m, (int, float))
            else [float(t) for t in simplify_m]
        )
        if tols:
            target = _record_toposimplify(
                row,
                topo,
                tolerances_m=tols,
                simplify_with=simplify_with,
                crs=gdf.crs,
            )

    if smooth:
        row.update(measure_smooth(target))
    else:
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
    skip_habitat: bool = False,
    habitat_only: bool = False,
) -> None:
    """Fetch datasets into the pooch cache. --fast: small zips only."""
    dest = cache_dir(cache)
    if habitat_only:
        print("pooch living_england_2022_2023 (OGL)")
        retrieve("living_england_2022_2023", dest)
        for bbox in LIVING_ENGLAND_BBOXES:
            print(f"Living England clip {bbox}")
            fetch_living_england(dest, bbox)
        print("pooch tampa_ccap_2021 (unzip GeoTIFF)")
        tampa_ccap_geotiff(dest)
        print(f"habitat cache ready under {dest}")
        return
    keys = list(CORE_KEYS)
    if fast or skip_tracts:
        keys = [k for k in keys if k != "cb_2024_us_tract_500k"]
    if fast:
        # counties 500k not in FAST_SIZE_IDS
        keys = [k for k in keys if k != "cb_2024_us_county_500k"]
        skip_rasters = True
        skip_top10nl = True
        skip_habitat = True
    for key in keys:
        print(f"pooch {key}")
        retrieve(key, dest)
    if not skip_rasters:
        if raster_source == "landcover_ai":
            print("pooch landcover_ai_v1")
            retrieve("landcover_ai_v1", dest)
        elif raster_source == "tampa_ccap":
            print("pooch tampa_ccap_2021")
            tampa_ccap_geotiff(dest)
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
    if not skip_habitat:
        for key in HABITAT_VECTOR_KEYS:
            print(f"pooch {key}")
            retrieve(key, dest)
        for bbox in LIVING_ENGLAND_BBOXES:
            print(f"Living England clip {bbox}")
            fetch_living_england(dest, bbox)
    print(f"cache ready under {dest}")


def run(
    cache: str | None = None,
    fast: bool = False,
    medium_subset: bool = False,
    habitat: bool = False,
    family: str | None = None,
    size_id: str | None = None,
    prequantize: str = "both",
    raster_source: str = "chesapeake",
    profile: bool = False,
    simplify: bool = True,
    smooth: bool = True,
    max_steps: int | None = None,
    download_first: bool = False,
) -> None:
    """Prepare size steps and time Topology (+ toposimplify by default).

    ``--simplify`` / ``--smooth`` default on for fast and medium profiles so
    every successful Topology row also records simplify (and the smooth hook).
    Pass ``--nosimplify`` / ``--nosmooth`` to skip. ``--medium_subset`` selects
    the representative medium size_ids beside ``--fast``. ``--habitat`` runs
    Living England nested bboxes + Tampa C-CAP chips (sets raster_source).
    """
    dest = cache_dir(cache)
    if habitat:
        raster_source = "tampa_ccap"
    if download_first:
        download(
            cache=str(dest),
            fast=fast,
            raster_source=raster_source,
            skip_tracts=fast,
            skip_rasters=fast,
            skip_top10nl=fast,
            skip_habitat=fast and not habitat,
            habitat_only=habitat,
        )

    steps = size_steps(raster_source=raster_source)
    if habitat:
        steps = [s for s in steps if s["size_id"] in HABITAT_SIZE_IDS]
    elif fast and medium_subset:
        allow = FAST_SIZE_IDS | MEDIUM_SUBSET_SIZE_IDS
        steps = [s for s in steps if s["size_id"] in allow]
    elif fast:
        steps = [s for s in steps if s["size_id"] in FAST_SIZE_IDS]
    elif medium_subset:
        steps = [s for s in steps if s["size_id"] in MEDIUM_SUBSET_SIZE_IDS]
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
        except Exception as exc:
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

        gdf = reproject_for_metres(gdf, dataset=step["dataset"], family=step["family"])
        tols = resolve_simplify_tolerances(step, simplify=simplify)
        print(
            f"  features={meta.get('n_features')} "
            f"vertices={meta.get('n_vertices')} crs={gdf.crs}"
            + (f" simplify_m={tols}" if tols else "")
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
                simplify_m=tols,
                smooth=smooth,
            )
            append_jsonl(row, dest)
            status = "FAIL" if row.get("failed") else "ok"
            simp = (
                f" simplify={row.get('simplify_wall_s')}s"
                f"/{row.get('n_vertices_simplified')}v"
                if row.get("simplify_wall_s") is not None
                else ""
            )
            print(
                f"    {status} wall={row.get('wall_s')}s arcs={row.get('n_arcs')}"
                f"{simp} -> summary.jsonl"
            )
            if not row.get("failed"):
                n_ok += 1

    print(f"Finished {n_ok} Topology runs; jsonl under {dest / 'runs'}")


if __name__ == "__main__":
    fire.Fire({"download": download, "run": run})
