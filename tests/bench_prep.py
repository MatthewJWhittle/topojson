"""Size-ladder prep for the scale benchmark: subsets, PDOK, raster polygonize.

Kept beside tests/bench_scale.py so the driver stays short. Spec for the matrix
lives in the Project datasets doc; this file is only the runnable steps.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
import pooch
import shapely.geometry
from shapely import get_num_coordinates, get_parts

# ---------------------------------------------------------------------------
# URLs (pooch). Natural Earth countries are not in geodatasets; use pooch.
# geodatasets covers other NE layers (land/lakes/cities) if needed later.
# ---------------------------------------------------------------------------

URLS: dict[str, str] = {
    "ne_110m_admin_0_countries": (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    ),
    "ne_10m_admin_0_countries": (
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    ),
    "ne_10m_rivers_lake_centerlines": (
        "https://naciscdn.org/naturalearth/10m/physical/"
        "ne_10m_rivers_lake_centerlines.zip"
    ),
    "tl_2024_11_tabblock20": (
        "https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/"
        "tl_2024_11_tabblock20.zip"
    ),
    "cb_2024_us_county_20m": (
        "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_20m.zip"
    ),
    "cb_2024_us_county_500k": (
        "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"
    ),
    "cb_2024_us_tract_500k": (
        "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_tract_500k.zip"
    ),
    "wash_11001_lulc_2021": (
        "https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/"
        "68011bd6d4be0263cab0ff29/wash_11001_lulc_2021_2024-Edition.tif"
    ),
    "loud_51107_lulc_2021": (
        "https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/"
        "68011c1fd4be0263cab100e6/loud_51107_lulc_2021_2024-Edition.tif"
    ),
    "landcover_ai_v1": (
        "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
    ),
}

CORE_KEYS = [
    "ne_110m_admin_0_countries",
    "ne_10m_admin_0_countries",
    "ne_10m_rivers_lake_centerlines",
    "tl_2024_11_tabblock20",
    "cb_2024_us_county_20m",
    "cb_2024_us_county_500k",
    "cb_2024_us_tract_500k",
]
CHESAPEAKE_KEYS = ["wash_11001_lulc_2021", "loud_51107_lulc_2021"]
RASTER_KEYS = set(CHESAPEAKE_KEYS) | {"landcover_ai_v1"}

TOP10NL_API = (
    "https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1/collections/{layer}/items"
)
TOP10NL_BBOXES = {
    "bbox_small": (5.40, 52.05, 5.60, 52.07),
    "bbox_mid": (5.20, 52.05, 5.80, 52.07),
    "bbox_wide": (5.00, 52.05, 6.02, 52.07),
}
TRACT_STATE_GROUPS = {
    "states_dc": ["11"],
    "states_midatl": ["11", "10", "24", "51"],
    "states_northeast": [
        "09",
        "10",
        "11",
        "23",
        "24",
        "25",
        "33",
        "34",
        "36",
        "42",
        "44",
        "50",
        "51",
    ],
}

DEFAULT_CACHE = Path(__file__).resolve().parent / "bench_cache"
# Datasets doc: time toposimplify at one or two metre tolerances (metre CRS).
DEFAULT_SIMPLIFY_M = [1.0, 2.0]


def cache_dir(path: str | Path | None = None) -> Path:
    root = Path(path) if path else DEFAULT_CACHE
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    return root


def retrieve(key: str, dest: Path) -> Path:
    """Fetch one URL into dest via pooch (hash skipped; public evolving files)."""
    url = URLS[key]
    fname = url.rsplit("/", 1)[-1]
    path = Path(
        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=fname,
            path=dest,
            progressbar=True,
        )
    )
    if key in RASTER_KEYS and path.suffix.lower() in {".tif", ".tiff"}:
        return path
    if path.suffix.lower() == ".zip":
        extract_dir = dest / key
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(extract_dir)
        return extract_dir
    return path


def vector_path(dest: Path, key: str) -> Path:
    extract_dir = dest / key
    if not extract_dir.exists():
        retrieve(key, dest)
    for pattern in ("*.shp", "*.gpkg", "*.geojson"):
        matches = sorted(extract_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No vector under {extract_dir}")


def geometry_stats(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    n_parts = n_vertices = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        n_parts += len(get_parts(geom))
        n_vertices += int(get_num_coordinates(geom))
    return {
        "n_features": len(gdf),
        "n_parts": n_parts,
        "n_vertices": n_vertices,
        "crs": gdf.crs.to_string() if gdf.crs is not None else None,
    }


def _vertex_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    return int(get_num_coordinates(geom))


def cumulative_by_vertices(gdf: gpd.GeoDataFrame, budget: int) -> gpd.GeoDataFrame:
    total = keep = 0
    for i, geom in enumerate(gdf.geometry):
        total += _vertex_count(geom)
        keep = i + 1
        if total >= budget:
            break
    return gdf.iloc[:keep].copy()


def longest_rings_budget(gdf: gpd.GeoDataFrame, budget: int) -> gpd.GeoDataFrame:
    rings = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        for part in get_parts(geom):
            if part.geom_type == "Polygon" and part.exterior is not None:
                rings.append(shapely.geometry.Polygon(part.exterior))
            else:
                rings.append(part)
    chosen, total = [], 0
    for geom in sorted(rings, key=_vertex_count, reverse=True):
        chosen.append(geom)
        total += _vertex_count(geom)
        if total >= budget:
            break
    return gpd.GeoDataFrame({"geometry": chosen}, crs=gdf.crs)


def feature_fraction(gdf: gpd.GeoDataFrame, fraction: float) -> gpd.GeoDataFrame:
    return gdf.iloc[: max(1, round(len(gdf) * fraction))].copy()


def tract_state_group(gdf: gpd.GeoDataFrame, group: str) -> gpd.GeoDataFrame:
    states = TRACT_STATE_GROUPS[group]
    col = "STATEFP" if "STATEFP" in gdf.columns else "STATEFP20"
    return gdf[gdf[col].isin(states)].copy()


def fetch_top10nl(dest: Path, layer: str, bbox_id: str) -> Path:
    out = dest / f"top10nl_{layer}_{bbox_id}.gpkg"
    if out.exists():
        return out
    bbox = TOP10NL_BBOXES[bbox_id]
    url: str | None = (
        f"{TOP10NL_API.format(layer=layer)}?"
        f"{urlencode({'bbox': ','.join(map(str, bbox)), 'limit': '1000', 'f': 'json'})}"
    )
    frames = []
    while url:
        req = Request(url, headers={"Accept": "application/geo+json, application/json"})
        with urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        feats = payload.get("features") or []
        if feats:
            frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        url = next(
            (
                link["href"]
                for link in payload.get("links") or []
                if link.get("rel") == "next" and link.get("href")
            ),
            None,
        )
    if frames:
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:28992")
    else:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:28992")
    gdf.to_file(out, layer=layer, driver="GPKG")
    return out


def polygonize_geotiff(
    path: Path,
    *,
    ground_resolution_m: float,
    chip_max_side: int | None = None,
    nodata: int | float | None = 0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    import rasterio
    from rasterio import features
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        native_x = abs(ds.transform.a)
        factor = max(1, round(ground_resolution_m / native_x))
        width, height = ds.width, ds.height
        window = None
        if chip_max_side is not None:
            side = min(chip_max_side, width, height)
            window = Window(
                max(0, (width - side) // 2),
                max(0, (height - side) // 2),
                side,
                side,
            )
            width = height = side
        out_w, out_h = max(1, width // factor), max(1, height // factor)
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
        meta = {
            "source_path": str(path),
            "source_width": ds.width,
            "source_height": ds.height,
            "source_pixel_size": native_x,
            "n_pixels_source_window": width * height,
            "n_pixels_generalized": out_w * out_h,
            "ground_resolution_m": ground_resolution_m,
            "chip_max_side": chip_max_side,
        }

    mask = None if src_nodata is None else data != src_nodata
    records = []
    for geom, value in features.shapes(data, mask=mask, transform=transform):
        if src_nodata is not None and int(value) == int(src_nodata):
            continue
        records.append({"class": int(value), "geometry": shapely.geometry.shape(geom)})
    gdf = (
        gpd.GeoDataFrame(records, crs=crs)
        if records
        else gpd.GeoDataFrame({"class": [], "geometry": []}, crs=crs)
    )
    meta.update(geometry_stats(gdf))
    meta["crs"] = crs.to_string() if crs else None
    return gdf, meta


def first_landcover_ai_mask(dest: Path) -> Path:
    extract = dest / "landcover_ai_v1"
    if not extract.exists():
        retrieve("landcover_ai_v1", dest)
    masks = sorted(extract.rglob("*.tif"))
    preferred = [p for p in masks if "mask" in p.as_posix().lower()]
    candidates = preferred or masks
    if not candidates:
        raise FileNotFoundError(f"No masks under {extract}")
    return candidates[0]


def reproject_for_metres(
    gdf: gpd.GeoDataFrame, *, dataset: str, family: str
) -> gpd.GeoDataFrame:
    if gdf.crs is None or gdf.crs.is_projected:
        return gdf
    us = ("tl_", "cb_", "wash_", "loud_", "tabblock", "county", "tract", "chesapeake")
    if family in {"shared_mesh", "raster_landcover"} or any(m in dataset for m in us):
        return gdf.to_crs("EPSG:5070")
    return gdf


def size_steps(raster_source: str = "chesapeake") -> list[dict[str, Any]]:
    """Compact step table: family / size_id / source / prep / args."""
    steps: list[dict[str, Any]] = [
        {
            "size_id": "ne110m_full",
            "source": "ne_110m_admin_0_countries",
            "family": "long_rings",
            "dataset": "ne_110m_admin_0_countries",
            "prep": "zip",
        },
        {
            "size_id": "ne10m_longest_rings_80k",
            "source": "ne_10m_admin_0_countries",
            "family": "long_rings",
            "dataset": "ne_10m_admin_0_countries",
            "prep": "longest_rings",
            "args": {"budget": 80_000},
        },
        {
            "size_id": "ne10m_full",
            "source": "ne_10m_admin_0_countries",
            "family": "long_rings",
            "dataset": "ne_10m_admin_0_countries",
            "prep": "zip",
        },
        {
            "size_id": "rivers_20k",
            "source": "ne_10m_rivers_lake_centerlines",
            "family": "open_lines",
            "dataset": "ne_10m_rivers_lake_centerlines",
            "prep": "cumulative",
            "args": {"budget": 20_000},
        },
        {
            "size_id": "rivers_80k",
            "source": "ne_10m_rivers_lake_centerlines",
            "family": "open_lines",
            "dataset": "ne_10m_rivers_lake_centerlines",
            "prep": "cumulative",
            "args": {"budget": 80_000},
        },
        {
            "size_id": "rivers_full",
            "source": "ne_10m_rivers_lake_centerlines",
            "family": "open_lines",
            "dataset": "ne_10m_rivers_lake_centerlines",
            "prep": "zip",
        },
        {
            "size_id": "dc_blocks_quarter",
            "source": "tl_2024_11_tabblock20",
            "family": "shared_mesh",
            "dataset": "tl_2024_11_tabblock20",
            "prep": "fraction",
            "args": {"fraction": 0.25},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "dc_blocks_half",
            "source": "tl_2024_11_tabblock20",
            "family": "shared_mesh",
            "dataset": "tl_2024_11_tabblock20",
            "prep": "fraction",
            "args": {"fraction": 0.5},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "dc_blocks_full",
            "source": "tl_2024_11_tabblock20",
            "family": "shared_mesh",
            "dataset": "tl_2024_11_tabblock20",
            "prep": "zip",
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "counties_20m",
            "source": "cb_2024_us_county_20m",
            "family": "shared_mesh",
            "dataset": "cb_2024_us_county_20m",
            "prep": "zip",
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "counties_500k",
            "source": "cb_2024_us_county_500k",
            "family": "shared_mesh",
            "dataset": "cb_2024_us_county_500k",
            "prep": "zip",
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "tracts_states_dc",
            "source": "cb_2024_us_tract_500k",
            "family": "shared_mesh",
            "dataset": "cb_2024_us_tract_500k",
            "prep": "tract_states",
            "args": {"group": "states_dc"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "tracts_states_midatl",
            "source": "cb_2024_us_tract_500k",
            "family": "shared_mesh",
            "dataset": "cb_2024_us_tract_500k",
            "prep": "tract_states",
            "args": {"group": "states_midatl"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "tracts_states_northeast",
            "source": "cb_2024_us_tract_500k",
            "family": "shared_mesh",
            "dataset": "cb_2024_us_tract_500k",
            "prep": "tract_states",
            "args": {"group": "states_northeast"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "top10nl_terrein_bbox_small",
            "source": "top10nl",
            "family": "shared_landcover",
            "dataset": "top10nl_terrein_vlak",
            "prep": "top10nl",
            "args": {"layer": "terrein_vlak", "bbox": "bbox_small"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "top10nl_terrein_bbox_mid",
            "source": "top10nl",
            "family": "shared_landcover",
            "dataset": "top10nl_terrein_vlak",
            "prep": "top10nl",
            "args": {"layer": "terrein_vlak", "bbox": "bbox_mid"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "top10nl_terrein_bbox_wide",
            "source": "top10nl",
            "family": "shared_landcover",
            "dataset": "top10nl_terrein_vlak",
            "prep": "top10nl",
            "args": {"layer": "terrein_vlak", "bbox": "bbox_wide"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
        {
            "size_id": "top10nl_gebouw_bbox_wide",
            "source": "top10nl",
            "family": "shared_landcover",
            "dataset": "top10nl_gebouw_vlak",
            "prep": "top10nl",
            "args": {"layer": "gebouw_vlak", "bbox": "bbox_wide"},
            "simplify_m": list(DEFAULT_SIMPLIFY_M),
        },
    ]
    if raster_source == "landcover_ai":
        steps.extend(
            [
                {
                    "size_id": "landcover_ai_chip_native",
                    "source": "landcover_ai_v1",
                    "family": "raster_landcover",
                    "dataset": "landcover_ai_v1",
                    "prep": "landcover_ai",
                    "args": {"ground_resolution_m": 0.25, "chip_max_side": 1024},
                    "simplify_m": [1.0],
                },
                {
                    "size_id": "landcover_ai_tile_1m",
                    "source": "landcover_ai_v1",
                    "family": "raster_landcover",
                    "dataset": "landcover_ai_v1",
                    "prep": "landcover_ai",
                    "args": {"ground_resolution_m": 1.0, "chip_max_side": None},
                    "simplify_m": list(DEFAULT_SIMPLIFY_M),
                },
                {
                    "size_id": "landcover_ai_tile_2m",
                    "source": "landcover_ai_v1",
                    "family": "raster_landcover",
                    "dataset": "landcover_ai_v1",
                    "prep": "landcover_ai",
                    "args": {"ground_resolution_m": 2.0, "chip_max_side": None},
                    "skip_pq_false": True,
                    "simplify_m": list(DEFAULT_SIMPLIFY_M),
                },
            ]
        )
    else:
        for key, area in (
            ("wash_11001_lulc_2021", "dc"),
            ("loud_51107_lulc_2021", "loudoun"),
        ):
            for metres, skip_false in ((1.0, False), (2.0, False), (4.0, True)):
                steps.append(
                    {
                        "size_id": f"chesapeake_{area}_{metres:g}m",
                        "source": key,
                        "family": "raster_landcover",
                        "dataset": key,
                        "prep": "raster",
                        "args": {
                            "ground_resolution_m": metres,
                            "chip_max_side": (
                                None if area == "dc" or metres >= 2 else 4096
                            ),
                        },
                        "skip_pq_false": skip_false or area == "loudoun",
                        "simplify_m": list(DEFAULT_SIMPLIFY_M),
                    }
                )
            steps.append(
                {
                    "size_id": f"chesapeake_{area}_chip_1m",
                    "source": key,
                    "family": "raster_landcover",
                    "dataset": key,
                    "prep": "raster",
                    "args": {"ground_resolution_m": 1.0, "chip_max_side": 1024},
                    "simplify_m": list(DEFAULT_SIMPLIFY_M),
                }
            )
    return steps


def prepare_step(
    step: dict[str, Any], dest: Path
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    prep, args = step["prep"], dict(step.get("args") or {})
    extra = {
        "size_id": step["size_id"],
        "dataset": step["dataset"],
        "family": step["family"],
        "source": step["source"],
    }
    if prep == "zip":
        gdf = gpd.read_file(vector_path(dest, step["source"]))
    elif prep == "cumulative":
        gdf = cumulative_by_vertices(
            gpd.read_file(vector_path(dest, step["source"])), int(args["budget"])
        )
    elif prep == "longest_rings":
        gdf = longest_rings_budget(
            gpd.read_file(vector_path(dest, step["source"])), int(args["budget"])
        )
    elif prep == "fraction":
        gdf = feature_fraction(
            gpd.read_file(vector_path(dest, step["source"])), float(args["fraction"])
        )
    elif prep == "tract_states":
        gdf = tract_state_group(
            gpd.read_file(vector_path(dest, step["source"])), str(args["group"])
        )
    elif prep == "top10nl":
        path = fetch_top10nl(dest, args["layer"], args["bbox"])
        gdf = gpd.read_file(path, layer=args["layer"])
    elif prep == "raster":
        path = retrieve(step["source"], dest)
        gdf, raster_meta = polygonize_geotiff(
            path,
            ground_resolution_m=float(args["ground_resolution_m"]),
            chip_max_side=args.get("chip_max_side"),
        )
        extra.update(raster_meta)
        return gdf, {**extra, **geometry_stats(gdf)}
    elif prep == "landcover_ai":
        path = first_landcover_ai_mask(dest)
        gdf, raster_meta = polygonize_geotiff(
            path,
            ground_resolution_m=float(args["ground_resolution_m"]),
            chip_max_side=args.get("chip_max_side"),
            nodata=0,
        )
        extra.update(raster_meta)
        return gdf, {**extra, **geometry_stats(gdf)}
    else:
        raise ValueError(f"Unknown prep {prep}")
    extra.update(geometry_stats(gdf))
    return gdf, extra
