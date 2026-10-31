"""Prepare size variants: subsets, ring budgets, TOP10NL layers, raster polygonize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry
from shapely import get_num_coordinates

from .cache import ensure_cache_dir, find_vector_path, geometry_stats
from .catalog import (
    REMOTE_FILES,
    TRACT_STATE_GROUPS,
    SizeStep,
)


def _vertex_count(geom: shapely.geometry.base.BaseGeometry) -> int:
    if geom is None or geom.is_empty:
        return 0
    return int(get_num_coordinates(geom))


def load_shapefile_zip(cache_dir: Path, source_key: str) -> gpd.GeoDataFrame:
    path = find_vector_path(cache_dir, source_key)
    return gpd.read_file(path)


def cumulative_by_vertices(
    gdf: gpd.GeoDataFrame, vertex_budget: int
) -> gpd.GeoDataFrame:
    """Keep features in order until the cumulative vertex count hits the budget."""
    counts = [_vertex_count(g) for g in gdf.geometry]
    total = 0
    keep = 0
    for i, n in enumerate(counts):
        total += n
        keep = i + 1
        if total >= vertex_budget:
            break
    return gdf.iloc[:keep].copy()


def longest_rings_budget(
    gdf: gpd.GeoDataFrame, vertex_budget: int
) -> gpd.GeoDataFrame:
    """Keep the longest rings (parts) until the vertex budget is met.

    Natural Earth countries are MultiPolygons; exploding to parts and ranking by
    ring length fills the gap between 110m (~10k verts) and full 10m (~548k).
    """
    from shapely import get_parts as _get_parts

    rings: list[Any] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        for part in _get_parts(geom):
            # Polygon exterior only for the long-coast stress; drop holes.
            if part.geom_type == "Polygon":
                ring = part.exterior
                if ring is not None:
                    rings.append(shapely.geometry.Polygon(ring))
            else:
                rings.append(part)
    ranked = sorted(rings, key=_vertex_count, reverse=True)
    chosen: list[Any] = []
    total = 0
    for geom in ranked:
        n = _vertex_count(geom)
        chosen.append(geom)
        total += n
        if total >= vertex_budget:
            break
    return gpd.GeoDataFrame({"geometry": chosen}, crs=gdf.crs)


def feature_fraction(gdf: gpd.GeoDataFrame, fraction: float) -> gpd.GeoDataFrame:
    n = max(1, int(round(len(gdf) * fraction)))
    return gdf.iloc[:n].copy()


def tract_state_group(gdf: gpd.GeoDataFrame, group: str) -> gpd.GeoDataFrame:
    states = TRACT_STATE_GROUPS[group]
    # Cartographic tract shapefiles expose STATEFP.
    col = "STATEFP" if "STATEFP" in gdf.columns else "STATEFP20"
    if col not in gdf.columns:
        raise KeyError(f"No state FIPS column on tracts; columns={list(gdf.columns)}")
    return gdf[gdf[col].isin(states)].copy()


def load_top10nl(
    cache_dir: Path, layer: str, bbox: str
) -> gpd.GeoDataFrame:
    path = cache_dir / f"top10nl_{layer}_{bbox}.gpkg"
    if not path.exists():
        from .cache import fetch_top10nl_bbox

        fetch_top10nl_bbox(cache_dir, layer, bbox)
    return gpd.read_file(path, layer=layer)


def _polygonize_array(
    data: np.ndarray,
    transform,
    crs,
    nodata: int | float | None = 0,
) -> gpd.GeoDataFrame:
    """Polygonize a 2D categorical array with rasterio shapes."""
    from rasterio import features

    mask = None if nodata is None else data != nodata
    shapes = features.shapes(data, mask=mask, transform=transform)
    records = []
    for geom, value in shapes:
        if nodata is not None and int(value) == int(nodata):
            continue
        records.append({"class": int(value), "geometry": shapely.geometry.shape(geom)})
    if not records:
        return gpd.GeoDataFrame({"class": [], "geometry": []}, crs=crs)
    return gpd.GeoDataFrame(records, crs=crs)


def polygonize_geotiff(
    path: Path,
    *,
    ground_resolution_m: float,
    chip_max_side: int | None = None,
    nodata: int | float | None = 0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Resample/sieve toward ground_resolution_m, then polygonize.

    Leaves the source GeoTIFF untouched; generalization happens here.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from rasterio.transform import Affine

    with rasterio.open(path) as ds:
        native_x = abs(ds.transform.a)
        factor = max(1, int(round(ground_resolution_m / native_x)))
        width = ds.width
        height = ds.height
        window = None
        if chip_max_side is not None:
            side = min(chip_max_side, width, height)
            col_off = max(0, (width - side) // 2)
            row_off = max(0, (height - side) // 2)
            window = Window(col_off, row_off, side, side)
            width = side
            height = side

        out_w = max(1, width // factor)
        out_h = max(1, height // factor)
        data = ds.read(
            1,
            window=window,
            out_shape=(out_h, out_w),
            resampling=Resampling.mode,
        )
        transform = ds.window_transform(window) if window else ds.transform
        if factor > 1:
            transform = Affine(
                transform.a * factor,
                transform.b,
                transform.c,
                transform.d,
                transform.e * factor,
                transform.f,
            )
        src_nodata = ds.nodata if ds.nodata is not None else nodata
        crs = ds.crs
        src_meta = {
            "source_path": str(path),
            "source_width": ds.width,
            "source_height": ds.height,
            "source_pixel_size": native_x,
            "n_pixels_source_window": int(width) * int(height),
            "n_pixels_generalized": int(out_w) * int(out_h),
            "ground_resolution_m": ground_resolution_m,
            "resample_factor": factor,
            "chip_max_side": chip_max_side,
        }

    gdf = _polygonize_array(data, transform, crs, nodata=src_nodata)
    meta = {**src_meta, **geometry_stats(gdf), "crs": crs.to_string() if crs else None}
    return gdf, meta


def _first_landcover_ai_mask(cache_dir: Path) -> Path:
    extract = cache_dir / "landcover_ai_v1"
    masks = sorted(extract.rglob("*.tif"))
    # Prefer files under a masks/ annotations-style path when present.
    preferred = [p for p in masks if "mask" in p.as_posix().lower()]
    candidates = preferred or masks
    if not candidates:
        raise FileNotFoundError(
            f"No GeoTIFF masks found under {extract}; download landcover_ai_v1 first"
        )
    return candidates[0]


def prepare_step(
    step: SizeStep, cache_dir: Path | None = None
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Materialize the GeoDataFrame for one size step, with prepare metadata."""
    cache_dir = ensure_cache_dir(cache_dir)
    prep = step.prep
    args = dict(step.prep_args)
    extra: dict[str, Any] = {
        "size_id": step.size_id,
        "dataset": step.dataset,
        "family": step.family,
        "source_key": step.source_key,
        "prep": prep,
    }

    if prep == "shapefile_zip":
        gdf = load_shapefile_zip(cache_dir, step.source_key)
    elif prep == "cumulative_vertices":
        gdf = cumulative_by_vertices(
            load_shapefile_zip(cache_dir, step.source_key),
            int(args["vertex_budget"]),
        )
    elif prep == "longest_rings":
        gdf = longest_rings_budget(
            load_shapefile_zip(cache_dir, step.source_key),
            int(args["vertex_budget"]),
        )
    elif prep == "feature_fraction":
        gdf = feature_fraction(
            load_shapefile_zip(cache_dir, step.source_key),
            float(args["fraction"]),
        )
    elif prep == "tract_states":
        gdf = tract_state_group(
            load_shapefile_zip(cache_dir, step.source_key),
            str(args["group"]),
        )
    elif prep == "top10nl_gpkg":
        gdf = load_top10nl(cache_dir, str(args["layer"]), str(args["bbox"]))
    elif prep == "raster_polygonize":
        remote = REMOTE_FILES[step.source_key]
        path = cache_dir / remote.filename
        gdf, raster_meta = polygonize_geotiff(
            path,
            ground_resolution_m=float(args["ground_resolution_m"]),
            chip_max_side=args.get("chip_max_side"),
        )
        extra.update(raster_meta)
        _write_prepare_meta(cache_dir, step.size_id, extra)
        return gdf, {**extra, **geometry_stats(gdf)}
    elif prep == "landcover_ai_polygonize":
        path = _first_landcover_ai_mask(cache_dir)
        gdf, raster_meta = polygonize_geotiff(
            path,
            ground_resolution_m=float(args["ground_resolution_m"]),
            chip_max_side=args.get("chip_max_side"),
            nodata=0,
        )
        extra.update(raster_meta)
        _write_prepare_meta(cache_dir, step.size_id, extra)
        return gdf, {**extra, **geometry_stats(gdf)}
    else:
        raise ValueError(f"Unknown prep kind: {prep}")

    stats = geometry_stats(gdf)
    extra.update(stats)
    _write_prepare_meta(cache_dir, step.size_id, extra)
    return gdf, extra


def _write_prepare_meta(cache_dir: Path, size_id: str, meta: dict[str, Any]) -> Path:
    path = cache_dir / "prepared" / f"{size_id}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    return path


def reproject_for_metres(
    gdf: gpd.GeoDataFrame, *, dataset: str, family: str
) -> gpd.GeoDataFrame:
    """Reproject geographic US inputs to EPSG:5070 before metre simplify tolerances.

    Leaves projected CRS (TOP10NL 28992, LandCover.ai 2180) and global Natural
    Earth geographic layers unchanged.
    """
    if gdf.crs is None or gdf.crs.is_projected:
        return gdf
    us_markers = (
        "tl_",
        "cb_",
        "wash_",
        "loud_",
        "tabblock",
        "county",
        "tract",
        "chesapeake",
    )
    if family in {"shared_mesh", "raster_landcover"} or any(
        m in dataset for m in us_markers
    ):
        return gdf.to_crs("EPSG:5070")
    return gdf
