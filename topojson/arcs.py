"""Composable float-arc surfaces for post-topology edits.

Decode once with ``Topology.arcs()``, chain ``simplify`` / ``segmentize`` /
``smooth`` / ``map``, then ``commit()`` back onto a Topology. Shared borders stay
one arc; indices are unchanged. Endpoints are pinned by the built-in smoothers
and by ``segmentize``.

Prefer ``topo.arcs().simplify(...).smooth(...).commit()`` over
``topo.simplify(...).smooth(...)`` when you care about decode cost: the high-level
helpers commit (and list-materialize) between steps unless a float-arc cache is
already warm. After ``commit``, ``Topology`` keeps decoded float arrays so a
follow-up ``arcs()`` / ``smooth`` does not re-decode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np

from .core import incremental
from .ops import (
    arc_coordinates,
    bounds,
    catmull_rom,
    chaikin_many,
    compare_bounds,
    restore_collapsed_rings,
    segmentize_many,
    simplify,
    simplify_keep,
)

if TYPE_CHECKING:
    from ._types import Algorithm, Package, SmoothMethod
    from .core.topology import Topology


class ArcSurface:
    """Decoded float arcs of a Topology, with chainable edits.

    Parameters
    ----------
    coordinates :
        One ``(n, 2)`` array per arc (float space).
    topology :
        Topology these arcs came from (needed for ``commit`` and for
        topology-preserving simplify ring repair).
    tags :
        Labels recorded for options when committing (``simplified``, ``smoothed``).
    """

    __slots__ = ("_coords", "_tags", "_topology")

    def __init__(
        self,
        coordinates: list[np.ndarray],
        *,
        topology: Topology | None = None,
        tags: frozenset[str] = frozenset(),
    ) -> None:
        self._coords = [np.asarray(c, dtype=np.float64) for c in coordinates]
        self._topology = topology
        self._tags = frozenset(tags)

    @classmethod
    def from_topology(cls, topology: Topology) -> Self:
        """Decode a Topology's arcs (and transform, if any) into float space."""
        cached = getattr(topology, "_float_arcs", None)
        if cached is not None and topology.output.get("transform") is None:
            return cls(cached, topology=topology)
        arcs = topology.output.get("arcs") or []
        coords = arc_coordinates(arcs, topology.output.get("transform"))
        return cls(coords, topology=topology)

    def __len__(self) -> int:
        return len(self._coords)

    def __iter__(self):
        return iter(self._coords)

    @property
    def coordinates(self) -> list[np.ndarray]:
        """Copy of each arc's coordinates."""
        return [c.copy() for c in self._coords]

    def as_lists(self) -> list[list[list[float]]]:
        """Nested Python lists suitable for TopoJSON float arcs."""
        return [c.tolist() for c in self._coords]

    def _spawn(self, coordinates: list[np.ndarray], *extra_tags: str) -> ArcSurface:
        return ArcSurface(
            coordinates,
            topology=self._topology,
            tags=self._tags.union(extra_tags),
        )

    def map(self, fn: Callable[[np.ndarray], np.ndarray]) -> ArcSurface:
        """Apply ``fn`` to each arc. Caller is responsible for pinning endpoints."""
        return self._spawn(
            [np.asarray(fn(c.copy()), dtype=np.float64) for c in self._coords]
        )

    def segmentize(self, max_segment_length: float) -> ArcSurface:
        """Densify so no edge exceeds ``max_segment_length`` (endpoints fixed)."""
        if max_segment_length <= 0:
            raise ValueError(
                f"max_segment_length must be > 0, got: {max_segment_length!r}"
            )
        return self._spawn(segmentize_many(self._coords, max_segment_length))

    def simplify(
        self,
        epsilon: float | None = None,
        *,
        keep: float | None = None,
        simplify_algorithm: Algorithm | None = None,
        simplify_with: Package | None = None,
        prevent_oversimplify: bool | None = None,
    ) -> ArcSurface:
        """Douglas-Peucker / Visvalingam-Whyatt on each arc (float path)."""
        if (epsilon is None) == (keep is None):
            raise ValueError("give either epsilon or keep")
        if keep is not None and not 0 <= keep <= 1:
            raise ValueError(f"keep is a share between 0 and 1, got: {keep!r}")

        topo = self._topology
        algorithm: Algorithm = "dp"
        package: Package = "shapely"
        keep_rings = True
        if topo is not None:
            algorithm = topo.options.simplify_algorithm
            package = topo.options.simplify_with
            keep_rings = topo.options.prevent_oversimplify
        if simplify_algorithm in ("dp", "vw"):
            algorithm = simplify_algorithm
        if simplify_with in ("shapely", "simplification"):
            package = simplify_with
        if "geos" in (simplify_with, package):
            raise ValueError(
                "simplify_with='geos' applies to presimplify coverage only; "
                "arc simplify uses 'shapely' or 'simplification'"
            )
        if isinstance(prevent_oversimplify, bool):
            keep_rings = prevent_oversimplify

        if keep is not None:
            arcs, _ = simplify_keep(self._coords, keep, algorithm)
        else:
            arcs = simplify(
                self._coords,
                epsilon,
                algorithm=algorithm,
                package=package,
                input_as="array",
                prevent_oversimplify=keep_rings,
            )
        coords = [np.asarray(a, dtype=np.float64) for a in arcs]
        if keep_rings and topo is not None:
            sequences = incremental._all_sequences(topo.output)
            rings = [s for s, ring in sequences if ring]
            restored = restore_collapsed_rings(
                [c.tolist() for c in coords], self._coords, rings
            )
            coords = [np.asarray(a, dtype=np.float64) for a in restored]
        return self._spawn(coords, "simplified")

    def smooth(
        self,
        method: SmoothMethod = "chaikin",
        *,
        iterations: int = 2,
        samples: int = 6,
        segment_length: float | None = None,
    ) -> ArcSurface:
        """Chaikin or Catmull-Rom on each arc; endpoints fixed."""
        if method not in ("chaikin", "catmull_rom"):
            raise ValueError(
                f"method must be 'chaikin' or 'catmull_rom', got: {method!r}"
            )
        if method == "chaikin":
            if not isinstance(iterations, int) or isinstance(iterations, bool):
                raise ValueError(f"iterations must be an int, got: {iterations!r}")
            if iterations < 0 or iterations > 5:
                raise ValueError(
                    f"iterations must be between 0 and 5 inclusive, got: {iterations}"
                )
        else:
            if not isinstance(samples, int) or isinstance(samples, bool):
                raise ValueError(f"samples must be an int, got: {samples!r}")
            if samples < 0:
                raise ValueError(f"samples must be >= 0, got: {samples}")
        if segment_length is not None and segment_length <= 0:
            raise ValueError(
                f"segment_length must be > 0 when set, got: {segment_length!r}"
            )

        coords = self._coords
        if segment_length is not None:
            coords = segmentize_many(coords, segment_length)
        if method == "chaikin":
            out = chaikin_many(coords, iterations=iterations)
        else:
            out = [catmull_rom(arc, samples=samples) for arc in coords]
        return self._spawn(out, "smoothed")

    @overload
    def commit(
        self,
        topology: Topology | None = None,
        *,
        inplace: Literal[False] = False,
        smooth_method: str | bool = "chaikin",
    ) -> Topology: ...

    @overload
    def commit(
        self,
        topology: Topology | None = None,
        *,
        inplace: Literal[True],
        smooth_method: str | bool = "chaikin",
    ) -> None: ...

    def commit(
        self,
        topology: Topology | None = None,
        *,
        inplace: bool = False,
        smooth_method: str | bool = "chaikin",
    ) -> Topology | None:
        """Write these float arcs onto a Topology; drop construction-grid transform.

        Stores ``float64`` arrays in ``output["arcs"]`` (no intermediate Python
        nested lists). ``to_json`` / ``NpEncoder`` materialize lists on export.
        Uses a light arc-edit copy so ``simplify().smooth()`` does not
        ``deepcopy`` every feature object.
        """
        topo = topology if topology is not None else self._topology
        if topo is None:
            raise ValueError(
                "pass topology=... or build the surface with Topology.arcs()"
            )
        result = topo if inplace else topo._copy_for_arc_edit()
        # Keep arrays in-place; share the list with the float-arc cache.
        result.output["arcs"] = self._coords
        result.output.pop("transform", None)
        result._float_arcs = self._coords
        lsbs = bounds(self._coords)
        ptbs = bounds(result.output.get("coordinates") or [])
        result.output["bbox"] = compare_bounds(lsbs, ptbs)
        if "simplified" in self._tags and not result.options.toposimplify:
            result.options.toposimplify = True
        if "smoothed" in self._tags:
            result.options.toposmooth = smooth_method
        if not inplace:
            return result
        topo.output["arcs"] = self._coords
        topo.output.pop("transform", None)
        topo._float_arcs = self._coords
        if "bbox" in result.output:
            topo.output["bbox"] = result.output["bbox"]
        topo.options.toposimplify = result.options.toposimplify
        topo.options.toposmooth = result.options.toposmooth
        return None
