"""Download and cache benchmark datasets; record per-file metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
from shapely import get_num_coordinates, get_parts

from .catalog import (
    CORE_DOWNLOAD_KEYS,
    CHESAPEAKE_DOWNLOAD_KEYS,
    DEFAULT_RASTER_SOURCE,
    REMOTE_FILES,
    TOP10NL_API,
    TOP10NL_BBOXES,
    TOP10NL_LAYERS,
    RemoteFile,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
METADATA_NAME = "metadata.json"


def default_cache_dir() -> Path:
    return CACHE_DIR


def ensure_cache_dir(cache_dir: Path | None = None) -> Path:
    path = Path(cache_dir) if cache_dir else default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(exist_ok=True)
    (path / "prepared").mkdir(exist_ok=True)
    return path


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "-o",
        str(partial),
        url,
    ]
    subprocess.run(cmd, check=True)
    partial.replace(dest)


def download_file(remote: RemoteFile, cache_dir: Path, force: bool = False) -> Path:
    dest = cache_dir / remote.filename
    if dest.exists() and not force:
        return dest
    print(f"Downloading {remote.key} -> {dest.name}")
    _curl(remote.url, dest)
    if remote.kind == "zip":
        extract_dir = cache_dir / remote.key
        if extract_dir.exists() and force:
            shutil.rmtree(extract_dir)
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(extract_dir)
    return dest


def find_vector_path(cache_dir: Path, key: str) -> Path:
    """Locate a .shp/.gpkg inside an extracted zip directory."""
    extract_dir = cache_dir / key
    if not extract_dir.exists():
        remote = REMOTE_FILES[key]
        download_file(remote, cache_dir)
    for pattern in ("*.shp", "*.gpkg", "*.geojson"):
        matches = sorted(extract_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No vector file found under {extract_dir}")


def geometry_stats(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    geoms = gdf.geometry
    n_features = int(len(gdf))
    n_parts = 0
    n_vertices = 0
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        n_parts += len(get_parts(geom))
        n_vertices += int(get_num_coordinates(geom))
    crs = None
    if gdf.crs is not None:
        crs = gdf.crs.to_string()
    return {
        "n_features": n_features,
        "n_parts": n_parts,
        "n_vertices": n_vertices,
        "crs": crs,
    }


def raster_stats(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as ds:
        transform = ds.transform
        crs = ds.crs.to_string() if ds.crs else None
        return {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": str(ds.dtypes[0]),
            "pixel_size_x": float(transform.a),
            "pixel_size_y": float(abs(transform.e)),
            "crs": crs,
            "n_pixels": int(ds.width) * int(ds.height),
        }


def record_vector_metadata(
    cache_dir: Path, key: str, path: Path | None = None
) -> dict[str, Any]:
    vector_path = path or find_vector_path(cache_dir, key)
    gdf = gpd.read_file(vector_path)
    stats = geometry_stats(gdf)
    archive = cache_dir / REMOTE_FILES[key].filename if key in REMOTE_FILES else None
    meta = {
        "key": key,
        "path": str(vector_path),
        "bytes": archive.stat().st_size if archive and archive.exists() else vector_path.stat().st_size,
        **stats,
    }
    return meta


def record_raster_metadata(cache_dir: Path, key: str) -> dict[str, Any]:
    remote = REMOTE_FILES[key]
    path = cache_dir / remote.filename
    meta = {
        "key": key,
        "path": str(path),
        "bytes": path.stat().st_size,
        "kind": "geotiff",
        **raster_stats(path),
    }
    return meta


def _http_get_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/geo+json, application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_top10nl_bbox(
    cache_dir: Path,
    layer: str,
    bbox_id: str,
    force: bool = False,
) -> Path:
    """Page PDOK OGC API Features for one layer/bbox into a GeoPackage."""
    if layer not in TOP10NL_LAYERS:
        raise ValueError(f"Unsupported TOP10NL layer: {layer}")
    if bbox_id not in TOP10NL_BBOXES:
        raise ValueError(f"Unknown bbox id: {bbox_id}")

    out = cache_dir / f"top10nl_{layer}_{bbox_id}.gpkg"
    if out.exists() and not force:
        return out

    bbox = TOP10NL_BBOXES[bbox_id]
    base = TOP10NL_API.format(layer=layer)
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "limit": "1000",
        "f": "json",
    }
    url: str | None = f"{base}?{urlencode(params)}"
    frames: list[gpd.GeoDataFrame] = []
    while url:
        print(f"PDOK {layer} {bbox_id}: {url}")
        payload = _http_get_json(url)
        features = payload.get("features") or []
        if features:
            frames.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))
        next_url = None
        for link in payload.get("links") or []:
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break
        url = next_url

    if not frames:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        # PDOK returns WGS84; TOP10NL map CRS is EPSG:28992.
        gdf = gdf.to_crs("EPSG:28992")

    if out.exists():
        out.unlink()
    gdf.to_file(out, layer=layer, driver="GPKG")
    return out


def download_top10nl(
    cache_dir: Path, force: bool = False, bboxes: list[str] | None = None
) -> list[Path]:
    paths: list[Path] = []
    bbox_ids = bboxes or list(TOP10NL_BBOXES)
    for bbox_id in bbox_ids:
        for layer in TOP10NL_LAYERS:
            # Nested terrain series needs every bbox; buildings only on widest.
            if layer == "gebouw_vlak" and bbox_id != "bbox_wide":
                continue
            paths.append(fetch_top10nl_bbox(cache_dir, layer, bbox_id, force=force))
    return paths


def load_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / METADATA_NAME
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text())


def save_metadata(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    path = cache_dir / METADATA_NAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def refresh_metadata(
    cache_dir: Path,
    keys: list[str] | None = None,
    include_top10nl: bool = True,
    include_rasters: bool = True,
) -> dict[str, Any]:
    metadata = load_metadata(cache_dir)
    files = metadata.setdefault("files", {})

    for key in keys or list(REMOTE_FILES):
        remote = REMOTE_FILES[key]
        path = cache_dir / remote.filename
        if not path.exists():
            continue
        if remote.kind == "geotiff":
            files[key] = record_raster_metadata(cache_dir, key)
        elif remote.kind == "zip":
            try:
                files[key] = record_vector_metadata(cache_dir, key)
            except Exception as exc:  # noqa: BLE001 — record failure, keep going
                files[key] = {
                    "key": key,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "error": str(exc),
                }

    if include_top10nl:
        for gpkg in cache_dir.glob("top10nl_*.gpkg"):
            key = gpkg.stem
            gdf = gpd.read_file(gpkg)
            files[key] = {
                "key": key,
                "path": str(gpkg),
                "bytes": gpkg.stat().st_size,
                **geometry_stats(gdf),
            }

    if include_rasters:
        for prepared in (cache_dir / "prepared").glob("*.meta.json"):
            files[prepared.stem] = json.loads(prepared.read_text())

    metadata["updated"] = _utcnow()
    save_metadata(cache_dir, metadata)
    return metadata


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def download_all(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    raster_source: str = DEFAULT_RASTER_SOURCE,
    skip_tracts: bool = False,
    skip_rasters: bool = False,
    skip_top10nl: bool = False,
) -> dict[str, Any]:
    """Fetch the six scripted datasets (and companions) into the local cache."""
    cache_dir = ensure_cache_dir(cache_dir)
    keys = list(CORE_DOWNLOAD_KEYS)
    if skip_tracts:
        keys = [k for k in keys if k != "cb_2024_us_tract_500k"]

    for key in keys:
        download_file(REMOTE_FILES[key], cache_dir, force=force)

    if not skip_rasters:
        if raster_source == "landcover_ai":
            download_file(REMOTE_FILES["landcover_ai_v1"], cache_dir, force=force)
        else:
            for key in CHESAPEAKE_DOWNLOAD_KEYS:
                download_file(REMOTE_FILES[key], cache_dir, force=force)

    if not skip_top10nl:
        download_top10nl(cache_dir, force=force)

    return refresh_metadata(
        cache_dir,
        keys=keys
        + (
            ["landcover_ai_v1"]
            if raster_source == "landcover_ai" and not skip_rasters
            else (CHESAPEAKE_DOWNLOAD_KEYS if not skip_rasters else [])
        ),
        include_top10nl=not skip_top10nl,
    )

