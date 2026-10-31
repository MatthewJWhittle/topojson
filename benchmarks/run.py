"""CLI for downloading datasets and running representative Topology benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cache import default_cache_dir, download_all, ensure_cache_dir, refresh_metadata
from .catalog import DEFAULT_RASTER_SOURCE, size_steps
from .measure import append_summary, measure_topology, write_run_row
from .prepare import prepare_step, reproject_for_metres


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Download representative topojson benchmark datasets and time Topology "
            "across cost families (prequantize on/off, stage timing, memory)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Cache directory (default: {default_cache_dir()})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download/cache datasets and write metadata")
    p_dl.add_argument("--force", action="store_true")
    p_dl.add_argument(
        "--raster-source",
        choices=["chesapeake", "landcover_ai"],
        default=DEFAULT_RASTER_SOURCE,
        help=(
            "Raster land-cover source. Default chesapeake (CC0). "
            "landcover_ai is CC BY-NC-SA and optional."
        ),
    )
    p_dl.add_argument("--skip-tracts", action="store_true")
    p_dl.add_argument("--skip-rasters", action="store_true")
    p_dl.add_argument("--skip-top10nl", action="store_true")

    p_run = sub.add_parser("run", help="Prepare sizes and time Topology")
    p_run.add_argument(
        "--raster-source",
        choices=["chesapeake", "landcover_ai"],
        default=DEFAULT_RASTER_SOURCE,
    )
    p_run.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Limit to one or more cost families (repeatable)",
    )
    p_run.add_argument(
        "--size-id",
        action="append",
        dest="size_ids",
        help="Limit to specific size_id values (repeatable)",
    )
    p_run.add_argument(
        "--prequantize",
        choices=["both", "true", "false"],
        default="both",
        help="Which prequantize settings to time (default both)",
    )
    p_run.add_argument(
        "--simplify",
        action="store_true",
        help="Also time toposimplify at the metres listed on each size step",
    )
    p_run.add_argument(
        "--simplify-with",
        default="shapely",
        help="Package for toposimplify (shapely or simplification)",
    )
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on number of size steps (smoke testing)",
    )
    p_run.add_argument(
        "--download-first",
        action="store_true",
        help="Call download before running",
    )

    p_meta = sub.add_parser("metadata", help="Refresh metadata.json from the cache")
    return parser


def cmd_download(args: argparse.Namespace) -> int:
    cache_dir = ensure_cache_dir(args.cache_dir)
    meta = download_all(
        cache_dir,
        force=args.force,
        raster_source=args.raster_source,
        skip_tracts=args.skip_tracts,
        skip_rasters=args.skip_rasters,
        skip_top10nl=args.skip_top10nl,
    )
    print(f"Wrote {cache_dir / 'metadata.json'} ({len(meta.get('files', {}))} files)")
    return 0


def cmd_metadata(args: argparse.Namespace) -> int:
    cache_dir = ensure_cache_dir(args.cache_dir)
    meta = refresh_metadata(cache_dir)
    print(f"Wrote {cache_dir / 'metadata.json'} ({len(meta.get('files', {}))} files)")
    return 0


def _prequantize_values(mode: str) -> list[bool]:
    if mode == "true":
        return [True]
    if mode == "false":
        return [False]
    return [True, False]


def cmd_run(args: argparse.Namespace) -> int:
    cache_dir = ensure_cache_dir(args.cache_dir)
    if args.download_first:
        download_all(
            cache_dir,
            raster_source=args.raster_source,
            skip_tracts=False,
            skip_rasters=False,
            skip_top10nl=False,
        )

    steps = size_steps(raster_source=args.raster_source)
    if args.families:
        wanted = set(args.families)
        steps = [s for s in steps if s.family in wanted]
    if args.size_ids:
        wanted_ids = set(args.size_ids)
        steps = [s for s in steps if s.size_id in wanted_ids]
    if args.max_steps is not None:
        steps = steps[: args.max_steps]

    if not steps:
        print("No size steps selected.", file=sys.stderr)
        return 1

    pq_values = _prequantize_values(args.prequantize)
    n_ok = 0
    for step in steps:
        print(f"Preparing {step.size_id} ({step.family})…")
        try:
            gdf, prep_meta = prepare_step(step, cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  prepare failed: {exc}", file=sys.stderr)
            row = {
                "family": step.family,
                "dataset": step.dataset,
                "size_id": step.size_id,
                "failed": True,
                "fail_stage": "prepare",
                "error": str(exc),
                "smooth": "smooth_not_implemented",
            }
            write_run_row(row, cache_dir)
            append_summary(row, cache_dir)
            continue

        # Metre CRS for US geographic inputs when simplifying in metres.
        gdf_m = reproject_for_metres(gdf, dataset=step.dataset, family=step.family)
        print(
            f"  features={prep_meta.get('n_features')} "
            f"vertices={prep_meta.get('n_vertices')} crs={gdf_m.crs}"
        )

        for pq in pq_values:
            if (not pq) and step.skip_prequantize_false:
                print(f"  skip prequantize=False for {step.size_id}")
                continue
            print(f"  Topology prequantize={pq}…")
            row = measure_topology(
                gdf_m,
                family=step.family,
                dataset=step.dataset,
                size_id=step.size_id,
                prequantize=pq,
            )
            path = write_run_row(row, cache_dir)
            append_summary(row, cache_dir)
            status = "FAIL" if row.get("failed") else "ok"
            print(
                f"    {status} wall={row.get('wall_s')}s "
                f"arcs={row.get('n_arcs')} junctions={row.get('n_junctions')} "
                f"-> {path.name}"
            )
            if not row.get("failed"):
                n_ok += 1

            if args.simplify and step.simplify_metres and not row.get("failed"):
                for tol in step.simplify_metres:
                    print(f"  Topology+simplify {tol}m prequantize={pq}…")
                    srow = measure_topology(
                        gdf_m,
                        family=step.family,
                        dataset=step.dataset,
                        size_id=step.size_id,
                        prequantize=pq,
                        simplify_with=args.simplify_with,
                        simplify_tolerance_m=tol,
                    )
                    write_run_row(srow, cache_dir)
                    append_summary(srow, cache_dir)

    print(f"Finished {n_ok} successful Topology runs; rows under {cache_dir / 'runs'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "metadata":
        return cmd_metadata(args)
    if args.command == "run":
        return cmd_run(args)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
