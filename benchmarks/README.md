# Representative topojson benchmarks

Harness for timing `Topology` on the six scripted datasets (and scale companions)
described in the project datasets doc. Rows land as JSON beside a local cache.
Smooth is recorded as `smooth_not_implemented` on every row (absent in 2.2).

## Setup

```bash
pip install -e '.[dev]'
pip install rasterio tqdm   # raster polygonize + optional progress
```

Cache default: `benchmarks/cache/` (gitignored). Override with `--cache-dir`.

## Download

Fetches Natural Earth, Census TIGER/GENZ zips, PDOK TOP10NL bbox extracts, and
the default raster family.

```bash
python -m benchmarks download
```

**Raster default:** Chesapeake Bay 1 m LULC COGs (CC0). LandCover.ai is
CC BY-NC-SA; use it only when that license fits:

```bash
python -m benchmarks download --raster-source landcover_ai
```

Useful flags: `--skip-tracts`, `--skip-rasters`, `--skip-top10nl`, `--force`.

Metadata (bytes, features, parts, vertices, CRS; width/height/pixel size for
rasters) is written to `benchmarks/cache/metadata.json`.

## Run

```bash
# Smoke: one family, prequantize on only, few steps
python -m benchmarks run --family long_rings --prequantize true --max-steps 2

# Full matrix for one family (on/off prequantize)
python -m benchmarks run --family shared_mesh

# Download then run everything selected
python -m benchmarks run --download-first --family open_lines
```

Each run writes `benchmarks/cache/runs/<family>_<size_id>_pq{0|1}.json` and
appends `benchmarks/cache/runs/summary.jsonl`.

Row fields include family, size, `prequantize`, input counts, arc/junction
counts, wall/process seconds, RSS, stage buckets (`t_extract_s`, `t_join_s`,
`t_other_s`), separate `json_wall_s` / `json_bytes`, optional simplify fields,
and `smooth: "smooth_not_implemented"`.

Optional simplify timing (metre CRS; US inputs are reprojected to EPSG:5070):

```bash
python -m benchmarks run --family shared_landcover --simplify
```

## Cost families

| Family | Datasets | Size variation |
| --- | --- | --- |
| `long_rings` | NE 110m + 10m countries | Full 110m; longest-ring budget on 10m; full 10m |
| `open_lines` | NE 10m rivers | Cumulative ~20k / ~80k / full vertices |
| `shared_mesh` | DC blocks; counties 20m/500k; tracts by state groups | Nested fractions / scales / states |
| `shared_landcover` | TOP10NL `terrein_vlak` (+ `gebouw_vlak`) | Nested PDOK bboxes |
| `raster_landcover` | Chesapeake COGs (default) or LandCover.ai | Area × ground resolution after polygonize |

Pool fits **inside** one family. Topology x-axis is input vertices.

## Legacy CI-style timings

`tests/benchmark_compute.py` still times the in-tree sample files for the
existing chart workflow. Prefer this package for representative public data.
