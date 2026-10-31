"""Dataset catalog for representative topojson benchmarks.

Durable dataset specs live in the Project docs (benchmark-datasets.md).
This module is the scripted mirror: URLs, cost families, and size steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Family = Literal[
    "long_rings",
    "open_lines",
    "shared_mesh",
    "shared_landcover",
    "raster_landcover",
]

# Default raster family uses Chesapeake Bay CC0 COGs. LandCover.ai is CC BY-NC-SA
# and remains available via --raster-source landcover_ai.
DEFAULT_RASTER_SOURCE = "chesapeake"


@dataclass(frozen=True)
class RemoteFile:
    """A curl-able artifact cached under benchmarks/cache/."""

    key: str
    url: str
    filename: str
    kind: Literal["zip", "geotiff", "gpkg"] = "zip"
    description: str = ""
    license_note: str = ""


@dataclass(frozen=True)
class SizeStep:
    """One runnable size inside a cost family."""

    size_id: str
    source_key: str
    family: Family
    dataset: str
    # How to derive the GeoDataFrame from the cached file.
    prep: str
    prep_args: dict[str, Any] = field(default_factory=dict)
    # When True, skip prequantize=False (huge generalized rasters).
    skip_prequantize_false: bool = False
    # Optional toposimplify metres (metre CRS inputs only).
    simplify_metres: list[float] = field(default_factory=list)


NATURAL_EARTH_110M = RemoteFile(
    key="ne_110m_admin_0_countries",
    url="https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    filename="ne_110m_admin_0_countries.zip",
    description="Natural Earth 1:110m admin-0 countries",
    license_note="Public domain (Natural Earth terms).",
)

NATURAL_EARTH_10M_COUNTRIES = RemoteFile(
    key="ne_10m_admin_0_countries",
    url="https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    filename="ne_10m_admin_0_countries.zip",
    description="Natural Earth 1:10m admin-0 countries (scale companion)",
    license_note="Public domain (Natural Earth terms).",
)

NATURAL_EARTH_RIVERS = RemoteFile(
    key="ne_10m_rivers_lake_centerlines",
    url=(
        "https://naciscdn.org/naturalearth/10m/physical/"
        "ne_10m_rivers_lake_centerlines.zip"
    ),
    filename="ne_10m_rivers_lake_centerlines.zip",
    description="Natural Earth 1:10m rivers and lake centerlines",
    license_note="Public domain (Natural Earth terms).",
)

TIGER_DC_BLOCKS = RemoteFile(
    key="tl_2024_11_tabblock20",
    url=(
        "https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/"
        "tl_2024_11_tabblock20.zip"
    ),
    filename="tl_2024_11_tabblock20.zip",
    description="TIGER/Line 2024 tabulation blocks, District of Columbia",
    license_note="US Census Bureau, public domain.",
)

CB_COUNTY_20M = RemoteFile(
    key="cb_2024_us_county_20m",
    url="https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_20m.zip",
    filename="cb_2024_us_county_20m.zip",
    description="Census cartographic counties 1:20,000,000 (scale companion)",
    license_note="US Census Bureau, public domain.",
)

CB_COUNTY_500K = RemoteFile(
    key="cb_2024_us_county_500k",
    url="https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip",
    filename="cb_2024_us_county_500k.zip",
    description="Census cartographic counties 1:500,000 (scale companion)",
    license_note="US Census Bureau, public domain.",
)

CB_TRACT_500K = RemoteFile(
    key="cb_2024_us_tract_500k",
    url="https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_tract_500k.zip",
    filename="cb_2024_us_tract_500k.zip",
    description="Census cartographic tracts 1:500,000, national",
    license_note="US Census Bureau, public domain.",
)

CHESAPEAKE_DC = RemoteFile(
    key="wash_11001_lulc_2021",
    url=(
        "https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/"
        "68011bd6d4be0263cab0ff29/wash_11001_lulc_2021_2024-Edition.tif"
    ),
    filename="wash_11001_lulc_2021_2024-Edition.tif",
    kind="geotiff",
    description="Chesapeake Bay 1 m LULC 2024 edition, District of Columbia",
    license_note="CC0 1.0 (Claggett and others, 2025, doi:10.5066/P14BEBRC).",
)

CHESAPEAKE_LOUDOUN = RemoteFile(
    key="loud_51107_lulc_2021",
    url=(
        "https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/"
        "68011c1fd4be0263cab100e6/loud_51107_lulc_2021_2024-Edition.tif"
    ),
    filename="loud_51107_lulc_2021_2024-Edition.tif",
    kind="geotiff",
    description="Chesapeake Bay 1 m LULC 2024 edition, Loudoun County VA",
    license_note="CC0 1.0 (Claggett and others, 2025, doi:10.5066/P14BEBRC).",
)

LANDCOVER_AI = RemoteFile(
    key="landcover_ai_v1",
    url="https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip",
    filename="landcover.ai.v1.zip",
    description="LandCover.ai v1 masks (optional; CC BY-NC-SA 4.0)",
    license_note=(
        "CC BY-NC-SA 4.0. Non-commercial. Cite Boguszewski et al., CVPR Workshops 2021. "
        "Default raster family uses Chesapeake Bay CC0 instead."
    ),
)

REMOTE_FILES: dict[str, RemoteFile] = {
    f.key: f
    for f in (
        NATURAL_EARTH_110M,
        NATURAL_EARTH_10M_COUNTRIES,
        NATURAL_EARTH_RIVERS,
        TIGER_DC_BLOCKS,
        CB_COUNTY_20M,
        CB_COUNTY_500K,
        CB_TRACT_500K,
        CHESAPEAKE_DC,
        CHESAPEAKE_LOUDOUN,
        LANDCOVER_AI,
    )
}

# Curl/zip downloads for the six scripted datasets plus scale companions.
# TOP10NL is fetched via PDOK OGC API (see TOP10NL_BBOXES), not a national zip.
CORE_DOWNLOAD_KEYS = [
    "ne_110m_admin_0_countries",
    "ne_10m_admin_0_countries",
    "ne_10m_rivers_lake_centerlines",
    "tl_2024_11_tabblock20",
    "cb_2024_us_county_20m",
    "cb_2024_us_county_500k",
    "cb_2024_us_tract_500k",
]

CHESAPEAKE_DOWNLOAD_KEYS = [
    "wash_11001_lulc_2021",
    "loud_51107_lulc_2021",
]

# Nested WGS84 bboxes for TOP10NL terrain (lon_min, lat_min, lon_max, lat_max).
# Innermost matches the measured window in the datasets doc; outer boxes grow area.
TOP10NL_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "bbox_small": (5.40, 52.05, 5.60, 52.07),
    "bbox_mid": (5.20, 52.05, 5.80, 52.07),
    "bbox_wide": (5.00, 52.05, 6.02, 52.07),
}

TOP10NL_LAYERS = ("terrein_vlak", "gebouw_vlak")
TOP10NL_API = (
    "https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1/collections/{layer}/items"
)

# State FIPS groups for nested tract series (many-cell mesh). Nested: each
# group includes the previous.
TRACT_STATE_GROUPS: dict[str, list[str]] = {
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


def size_steps(raster_source: str = DEFAULT_RASTER_SOURCE) -> list[SizeStep]:
    """Return the default size matrix for all cost families."""
    steps: list[SizeStep] = [
        # Family 1: long rings, few junctions
        SizeStep(
            size_id="ne110m_full",
            source_key="ne_110m_admin_0_countries",
            family="long_rings",
            dataset="ne_110m_admin_0_countries",
            prep="shapefile_zip",
        ),
        SizeStep(
            size_id="ne10m_longest_rings_80k",
            source_key="ne_10m_admin_0_countries",
            family="long_rings",
            dataset="ne_10m_admin_0_countries",
            prep="longest_rings",
            prep_args={"vertex_budget": 80_000},
        ),
        SizeStep(
            size_id="ne10m_full",
            source_key="ne_10m_admin_0_countries",
            family="long_rings",
            dataset="ne_10m_admin_0_countries",
            prep="shapefile_zip",
        ),
        # Family 2: open lines, almost no shared path
        SizeStep(
            size_id="rivers_20k",
            source_key="ne_10m_rivers_lake_centerlines",
            family="open_lines",
            dataset="ne_10m_rivers_lake_centerlines",
            prep="cumulative_vertices",
            prep_args={"vertex_budget": 20_000},
        ),
        SizeStep(
            size_id="rivers_80k",
            source_key="ne_10m_rivers_lake_centerlines",
            family="open_lines",
            dataset="ne_10m_rivers_lake_centerlines",
            prep="cumulative_vertices",
            prep_args={"vertex_budget": 80_000},
        ),
        SizeStep(
            size_id="rivers_full",
            source_key="ne_10m_rivers_lake_centerlines",
            family="open_lines",
            dataset="ne_10m_rivers_lake_centerlines",
            prep="shapefile_zip",
        ),
        # Family 3a: DC blocks nested subsets
        SizeStep(
            size_id="dc_blocks_quarter",
            source_key="tl_2024_11_tabblock20",
            family="shared_mesh",
            dataset="tl_2024_11_tabblock20",
            prep="feature_fraction",
            prep_args={"fraction": 0.25},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="dc_blocks_half",
            source_key="tl_2024_11_tabblock20",
            family="shared_mesh",
            dataset="tl_2024_11_tabblock20",
            prep="feature_fraction",
            prep_args={"fraction": 0.5},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="dc_blocks_full",
            source_key="tl_2024_11_tabblock20",
            family="shared_mesh",
            dataset="tl_2024_11_tabblock20",
            prep="shapefile_zip",
            simplify_metres=[2.0],
        ),
        # Family 3b: counties fixed-polygon, vertex density
        SizeStep(
            size_id="counties_20m",
            source_key="cb_2024_us_county_20m",
            family="shared_mesh",
            dataset="cb_2024_us_county_20m",
            prep="shapefile_zip",
        ),
        SizeStep(
            size_id="counties_500k",
            source_key="cb_2024_us_county_500k",
            family="shared_mesh",
            dataset="cb_2024_us_county_500k",
            prep="shapefile_zip",
        ),
        # Family 3c: tracts many-cell nested states
        SizeStep(
            size_id="tracts_states_dc",
            source_key="cb_2024_us_tract_500k",
            family="shared_mesh",
            dataset="cb_2024_us_tract_500k",
            prep="tract_states",
            prep_args={"group": "states_dc"},
        ),
        SizeStep(
            size_id="tracts_states_midatl",
            source_key="cb_2024_us_tract_500k",
            family="shared_mesh",
            dataset="cb_2024_us_tract_500k",
            prep="tract_states",
            prep_args={"group": "states_midatl"},
        ),
        SizeStep(
            size_id="tracts_states_northeast",
            source_key="cb_2024_us_tract_500k",
            family="shared_mesh",
            dataset="cb_2024_us_tract_500k",
            prep="tract_states",
            prep_args={"group": "states_northeast"},
        ),
        # Family 5: TOP10NL shared land cover (PDOK bbox)
        SizeStep(
            size_id="top10nl_terrein_bbox_small",
            source_key="top10nl_terrein_vlak_bbox_small",
            family="shared_landcover",
            dataset="top10nl_terrein_vlak",
            prep="top10nl_gpkg",
            prep_args={"layer": "terrein_vlak", "bbox": "bbox_small"},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="top10nl_terrein_bbox_mid",
            source_key="top10nl_terrein_vlak_bbox_mid",
            family="shared_landcover",
            dataset="top10nl_terrein_vlak",
            prep="top10nl_gpkg",
            prep_args={"layer": "terrein_vlak", "bbox": "bbox_mid"},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="top10nl_terrein_bbox_wide",
            source_key="top10nl_terrein_vlak_bbox_wide",
            family="shared_landcover",
            dataset="top10nl_terrein_vlak",
            prep="top10nl_gpkg",
            prep_args={"layer": "terrein_vlak", "bbox": "bbox_wide"},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="top10nl_gebouw_bbox_wide",
            source_key="top10nl_gebouw_vlak_bbox_wide",
            family="shared_landcover",
            dataset="top10nl_gebouw_vlak",
            prep="top10nl_gpkg",
            prep_args={"layer": "gebouw_vlak", "bbox": "bbox_wide"},
            simplify_metres=[2.0],
        ),
    ]

    if raster_source == "landcover_ai":
        steps.extend(_landcover_ai_steps())
    else:
        steps.extend(_chesapeake_steps())
    return steps


def _chesapeake_steps() -> list[SizeStep]:
    """Raster land-cover family using Chesapeake Bay CC0 COGs (script default)."""
    out: list[SizeStep] = []
    for key, area in (
        ("wash_11001_lulc_2021", "dc"),
        ("loud_51107_lulc_2021", "loudoun"),
    ):
        for metres, skip_false in ((1.0, False), (2.0, False), (4.0, True)):
            out.append(
                SizeStep(
                    size_id=f"chesapeake_{area}_{metres:g}m",
                    source_key=key,
                    family="raster_landcover",
                    dataset=key,
                    prep="raster_polygonize",
                    prep_args={
                        "ground_resolution_m": metres,
                        "chip_max_side": None if area == "dc" or metres >= 2 else 4096,
                    },
                    skip_prequantize_false=skip_false or area == "loudoun",
                    simplify_metres=[2.0],
                )
            )
        # Chip at native 1 m for a small end of the area axis
        out.append(
            SizeStep(
                size_id=f"chesapeake_{area}_chip_1m",
                source_key=key,
                family="raster_landcover",
                dataset=key,
                prep="raster_polygonize",
                prep_args={"ground_resolution_m": 1.0, "chip_max_side": 1024},
                simplify_metres=[2.0],
            )
        )
    return out


def _landcover_ai_steps() -> list[SizeStep]:
    """Optional LandCover.ai steps (CC BY-NC-SA); not the script default."""
    return [
        SizeStep(
            size_id="landcover_ai_chip_native",
            source_key="landcover_ai_v1",
            family="raster_landcover",
            dataset="landcover_ai_v1",
            prep="landcover_ai_polygonize",
            prep_args={"ground_resolution_m": 0.25, "chip_max_side": 1024},
            simplify_metres=[1.0],
        ),
        SizeStep(
            size_id="landcover_ai_tile_1m",
            source_key="landcover_ai_v1",
            family="raster_landcover",
            dataset="landcover_ai_v1",
            prep="landcover_ai_polygonize",
            prep_args={"ground_resolution_m": 1.0, "chip_max_side": None},
            simplify_metres=[2.0],
        ),
        SizeStep(
            size_id="landcover_ai_tile_2m",
            source_key="landcover_ai_v1",
            family="raster_landcover",
            dataset="landcover_ai_v1",
            prep="landcover_ai_polygonize",
            prep_args={"ground_resolution_m": 2.0, "chip_max_side": None},
            skip_prequantize_false=True,
            simplify_metres=[2.0],
        ),
    ]
