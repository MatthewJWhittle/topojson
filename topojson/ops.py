import itertools
import logging
import os
import pprint

import numpy as np
import shapely
from shapely import geometry, wkt
from shapely.ops import linemerge
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)

# Optional Rust coordinate core (maturin). Absent → pure NumPy / Shapely path.
_RUST = None
_RUST_LOAD_ATTEMPTED = False


def _rust_core():
    """Return the ``topojson._core`` extension module, or ``None`` if unavailable."""
    global _RUST, _RUST_LOAD_ATTEMPTED
    if not _RUST_LOAD_ATTEMPTED:
        _RUST_LOAD_ATTEMPTED = True
        try:
            from . import _core as rust_mod

            _RUST = rust_mod
        except ImportError:
            _RUST = None
    return _RUST


def rust_core_available() -> bool:
    """True when the optional PyO3 ``topojson._core`` extension is importable."""
    return _rust_core() is not None


def _rust_join_enabled() -> bool:
    """Opt-in for the Rust grid join (default off; NumPy is faster at LE scale)."""
    return os.environ.get("TOPOJSON_RUST_JOIN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def asvoid(arr):
    """
    Utility function to create a 1-dimensional numpy void object (bytes)
    of a 2-dimensional array. This is useful for the function numpy.in1d(),
    since it only accepts 1-dimensional objects.

    Parameters
    ----------
    arr : numpy.array
        2-dimensional numpy array

    Returns
    -------
    numpy.void
        1-dimensional numpy void object
    """

    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.0
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def strtree_query_geoms(tree, arc):
    # get junctions that contain within bbox line
    if hasattr(tree, "geometries"):
        # shapely version >= 1.8.3
        tree_index = tree.query(arc)
        tree_geom = tree.geometries.take(tree_index)
    elif hasattr(tree, "query_geoms"):
        # 1.8.0 <= shapely version < 1.8.3
        tree_geom = tree.query_geoms(arc)
    else:
        # shapely version < 1.8.0
        tree_geom = tree.query(arc)
    return tree_geom  # pts_within_bbox


def strtree_query_index(tree, arc, geoms):
    if hasattr(tree, "geometries"):
        # shapely version >= 1.8.3
        tree_index = tree.query(arc)
    elif hasattr(tree, "query_items"):
        # 1.8.0 <= shapely version < 1.8.3
        tree_index = tree.query_items(arc)
    else:
        # shapely version < 1.8.0
        tree_geom = tree.query(arc)
        if isinstance(tree_geom, shapely.geometry.base.BaseGeometry):
            tree_geom = [tree_geom]
        tree_index = [geoms.index(tree_g) for tree_g in tree_geom]
        return tree_index
        # raise AttributeError("This version is not supported within topojson")
    return tree_index


def explode(segments):
    if not isinstance(segments, list):
        segments = [segments]
    list_explode = [
        list(geom.geoms) if hasattr(geom, "geoms") else [geom] for geom in segments
    ]
    return list(itertools.chain.from_iterable(list_explode))


def linemerge_ext(geom: geometry.base.BaseGeometry) -> geometry.base.BaseGeometry:
    line = extract_lines(geom)
    if isinstance(line, geometry.MultiLineString):
        return linemerge(line)
    else:
        return line


def extract_lines(geom: geometry.base.BaseGeometry) -> geometry.base.BaseGeometry:
    if isinstance(geom, (geometry.LineString, geometry.MultiLineString)):
        return geom
    elif isinstance(geom, geometry.GeometryCollection):
        geoms = [
            geom
            for geom in geom.geoms
            if (
                not geom.is_empty
                and isinstance(geom, (geometry.LineString, geometry.MultiLineString))
            )
        ]
        if len(geoms) == 0:
            return geometry.LineString()
        elif len(geoms) == 1:
            return geoms[0]
        else:
            return geometry.MultiLineString(geoms)
    return geometry.LineString()


def np_array_bbox_points_line(line, tree_splitter):
    """
    Get junctions within bbox of line and return both as numpy array

    Parameters
    ----------
    line : numpy.array
        numpy array with coordinates representing a line segment
    tree_splitter : STRtree
        a STRtree splitter object

    Returns
    -------
    numpy.array
        `ls_xy`, numpy array from coordinates, if any, representing a line segment
    numpy.array
        `pts_xy_bbox`, numpy array with coordinates that near or on the line
    """

    # get junctions that contain within bbox line
    pts_within_bbox = strtree_query_geoms(tree_splitter, line)

    if len(pts_within_bbox) == 0:
        # no point near bbox, nothing to insert, nothing to split
        return None, None
    # convert shapely linestring and multipoint to np.array if there are points on line
    ls_xy = np.array(line.coords)
    pts_xy_bbox = np.array([x for pt in pts_within_bbox for x in pt.coords])

    return ls_xy, pts_xy_bbox


def insert_coords_in_line(line, tree_splitter):
    """
    Insert coordinates that are on the line, but where no vertices exists

    Parameters
    ----------
    line : numpy.array
        numpy array with coordinates representing a line segment
    tree_splitter : STRtree
        a STRtree splitter object

    Returns
    -------
    (numpy.array)
        `new_ls_xy` is an array with inserted coordinates, if any, representing a line
        segment
    (numpy.array)
        `pts_xy_on_line` is an array with coordinates that are on the line
    """

    # get junctions that contain within bbox line
    pts_within_bbox = strtree_query_geoms(tree_splitter, line)

    # select junctions that are within tolerance of line
    tol_dist = 1e-8
    pts_on_line = list(
        itertools.compress(
            pts_within_bbox, [line.distance(pt) < tol_dist for pt in pts_within_bbox]
        )
    )

    if len(pts_on_line) == 0:
        # no point on line, nothing to insert, nothing to split
        return None, None
    # convert shapely linestring and multipoint to np.array if there are points on line
    ls_xy = np.array(line.coords)
    pts_xy_on_line = np.array([x for pt in pts_on_line for x in pt.coords])

    # select junctions having non existing vertices in linestring
    tol_float_prc = 1e8
    pts_xy_nonexst = pts_xy_on_line[
        ~np.isin(
            asvoid(np.around(pts_xy_on_line * tol_float_prc).astype(np.int64))[:, 0],
            asvoid(np.around(ls_xy * tol_float_prc).astype(np.int64))[:, 0],
        )
    ]
    if pts_xy_nonexst.size == 0:
        return ls_xy, pts_xy_on_line
    # compute the distance from the beginning of the linestring for each junction on line
    splitter_dist = np.array(
        [line.project(pt) for pt in geometry.MultiPoint(pts_xy_nonexst).geoms]
    )
    splitter_dist = splitter_dist[splitter_dist > 0]

    # sort distance of non-existing junctions and apply sorting to the splitter distance
    # and corresponding junction coordinates to be inserted.
    sort_idx = splitter_dist.argsort()
    splitter_dist = splitter_dist[sort_idx]
    pts_xy_nonexst = pts_xy_nonexst[sort_idx]

    # get euclidean distance of all coords of line
    ls_xy_roll = np.roll(ls_xy, 1, axis=0)
    roll_min_ls = ls_xy_roll - ls_xy
    eucl_dist = np.sqrt(np.einsum("ij,ij->i", roll_min_ls, roll_min_ls))

    # the first distance is computed from the first point to the last point, set to 0
    eucl_dist[0] = 0
    ls_cumsum = eucl_dist.cumsum()

    # include junctions in linestring
    insert_idx = np.searchsorted(ls_cumsum, splitter_dist)
    new_ls_xy = np.insert(ls_xy, insert_idx, pts_xy_nonexst, axis=0)

    return new_ls_xy, pts_xy_on_line


def shared_path_ends(pairs):
    """
    Start and end points of the paths that each pair of lines shares. These are the
    junctions of the topology. Lines that only touch or cross share no path, and equal
    lines are skipped.

    Parameters
    ----------
    pairs : iterable of (LineString, LineString)
        Pairs of lines whose envelopes intersect

    Returns
    -------
    set of tuple
        Junction coordinates
    """
    paths = [linemerge_ext(a.intersection(b)) for a, b in pairs if not a.equals(b)]
    paths = explode([path for path in paths if not path.is_empty])
    return {xy for path in paths for xy in (path.coords[0], path.coords[-1])}


def _coordinates_index(lines):
    """Packed ``(V, 2)`` coordinates and per-vertex line index for geom or array arcs."""
    if not len(lines):
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.intp)
    if hasattr(lines[0], "coords"):
        return shapely.get_coordinates(lines, return_index=True)
    arrays = [np.asarray(ls, dtype=np.float64).reshape(-1, 2) for ls in lines]
    xy = np.concatenate(arrays) if arrays else np.empty((0, 2), dtype=np.float64)
    li = np.repeat(np.arange(len(arrays), dtype=np.intp), [len(a) for a in arrays])
    return xy, li


def _as_linestring_list(lines):
    """Ensure a list of Shapely LineStrings (batch-build from coordinate arrays)."""
    if not len(lines) or hasattr(lines[0], "coords"):
        return lines
    xy, li = _coordinates_index(lines)
    if len(xy) == 0:
        return [geometry.LineString() for _ in lines]
    # linestrings() needs ≥2 vertices per part; fall back per-arc when any are short.
    counts = np.bincount(li, minlength=len(lines))
    if int(counts.min()) >= 2:
        return list(shapely.linestrings(xy, indices=li))
    out = []
    for i, n in enumerate(counts.tolist()):
        if n < 2:
            out.append(geometry.LineString())
        else:
            out.append(geometry.LineString(np.asarray(lines[i], dtype=float)))
    return out


def shared_path_ends_on_grid(linestrings):
    """
    Junctions of quantized lines: the start and end points of the paths that each
    pair of lines shares, as `shared_path_ends` returns them for all pairs, but
    computed for all lines at once on the integer grid.

    On the grid, a shared path is a run of segments that both lines contain. A
    vertex that lies inside a collinear segment of another line is first inserted
    in that line. A vertex is then a junction when the other lines on its incoming
    segment differ from those on its outgoing segment. Equal lines count once, as
    equal pairs share no path. Lines that pass a vertex twice, or that are closed
    and lie wholly on another line, are left to `shared_path_ends` together with
    the lines they share segments with. Segments and sets of lines are compared by
    64-bit hashes.

    Parameters
    ----------
    linestrings : list of LineString or (n, 2) arrays
        Lines with integer coordinates; a closed line continues from its end into
        its start

    Returns
    -------
    set of tuple
        Junction coordinates
    """
    if len(linestrings) == 0:
        return set()
    xy, li = _coordinates_index(linestrings)
    xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
    li = np.asarray(li, dtype=np.int32)

    if hasattr(linestrings[0], "coords"):
        closed = shapely.is_closed(np.asarray(linestrings, dtype=object))
    else:
        closed = np.fromiter(
            (
                len(a) > 2 and bool(np.array_equal(np.asarray(a[0]), np.asarray(a[-1])))
                for a in linestrings
            ),
            dtype=bool,
            count=len(linestrings),
        )

    # Rust join exists on topojson._core but is not the default: the NumPy
    # vectorized path is faster on Living England-scale meshes. Opt in with
    # TOPOJSON_RUST_JOIN=1 once the Rust port matches NumPy throughput.
    rust = _rust_core()
    if rust is not None and _rust_join_enabled():
        try:
            points, pairs = rust.shared_path_ends_on_grid(xy, li, closed)
            points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
            if len(pairs):
                geoms = _as_linestring_list(linestrings)
                ends = shared_path_ends(
                    [(geoms[int(i)], geoms[int(j)]) for i, j in pairs]
                )
            else:
                ends = set()
            return ends | {tuple(p) for p in points.tolist()}
        except Exception:
            logger.debug(
                "rust shared_path_ends_on_grid failed; using NumPy path",
                exc_info=True,
            )

    s, h, vertex = _grid_segments(xy, li)
    at, pts = _t_vertices(xy, li, s, h)
    if len(at):
        o = np.lexsort([np.abs(pts - xy[at]).sum(1), at])
        xy = np.insert(xy, at[o] + 1, pts[o], axis=0)
        li = np.insert(li, at[o] + 1, li[at[o]])
        s, h, vertex = _grid_segments(xy, li)
    points, pairs = _junction_vertices(xy, li, s, h, vertex, closed)
    if len(pairs):
        geoms = _as_linestring_list(linestrings)
        ends = shared_path_ends([(geoms[i], geoms[j]) for i, j in pairs])
    else:
        ends = set()
    return ends | set(map(tuple, points.tolist()))


def _mix(x):
    """splitmix64 finalizer."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _hash(*cols):
    """64-bit hash of the rows of integer columns."""
    h = np.full(len(cols[0]), 0x9E3779B97F4A7C15, np.uint64)
    for col in cols:
        h = _mix(h + np.asarray(col, np.int64).view(np.uint64))
    return h.view(np.int64)


def _repeated(h):
    """True where a hash may occur more than once (bucket counts, no sort)."""
    size = 1 << max(int(8 * len(h)).bit_length(), 4)
    bucket = (h & (size - 1)).astype(np.intp)
    return np.bincount(bucket, minlength=size)[bucket] > 1


def _changes(*cols):
    """True at the first row and where any column differs from the row before."""
    new = np.ones(len(cols[0]), bool)
    new[1:] = np.any([col[1:] != col[:-1] for col in cols], axis=0)
    return new


def _ranges(starts, counts):
    """Concatenated ranges [starts[i], starts[i] + counts[i])."""
    offset = np.repeat(starts - np.cumsum(counts) + counts, counts)
    return offset + np.arange(counts.sum())


def _grid_segments(xy, li):
    """First vertex of each segment, a hash of the undirected segment, and a hash
    of each vertex."""
    vertex = _hash(*xy.T)
    s = np.flatnonzero(li[1:] == li[:-1])
    a, b = vertex[s], vertex[s + 1]
    return s, _hash(a ^ b, a + b), vertex


def _t_vertices(xy, li, s, h):
    """Vertices that lie strictly inside a collinear segment: the first vertex of
    each segment that holds one, and the point."""
    a, b = xy[s], xy[s + 1]
    d = b - a
    d //= np.maximum(np.gcd(*d.T), 1)[:, None]
    d[(d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0))] *= -1
    c = d[:, 1] * a[:, 0] - d[:, 0] * a[:, 1]
    lo, hi = np.sort(np.einsum("ij,kij->ki", d, np.stack([a, b])), 0)
    carrier = _hash(*d.T, c)  # the line through the segment

    # straight stretches of the lines, equal ones (shared borders) once; only
    # carriers on which stretches overlap can hold such a vertex
    r = np.flatnonzero(_changes(li[s], carrier))
    key, start, end = carrier[r], np.minimum.reduceat(lo, r), np.maximum.reduceat(hi, r)
    u = np.flatnonzero(_repeated(key))
    stretch = _hash(key[u], start[u], end[u], np.bitwise_xor.reduceat(h, r)[u])
    u = u[np.unique(stretch, return_index=True)[1]]
    u = u[np.lexsort([start[u], key[u]])]
    overlap = (key[u][1:] == key[u][:-1]) & (start[u][1:] < end[u][:-1])
    k = np.flatnonzero(np.isin(carrier, key[u][1:][overlap]))

    # rank the segment ends along their exact carrier; the ends ranked strictly
    # between those of a segment lie inside it
    cols = [np.tile(x[k], 2) for x in (d[:, 0], d[:, 1], c)] + [np.r_[lo[k], hi[k]]]
    o = np.lexsort(cols[::-1])
    new = _changes(*(col[o] for col in cols))
    rank = np.empty(len(o), np.intp)
    rank[o] = np.cumsum(new) - 1
    count = np.maximum(rank[len(k) :] - rank[: len(k)] - 1, 0)
    seg = np.repeat(k, count)
    t = cols[3][o][new][_ranges(rank[: len(k)] + 1, count)]
    ds, origin = d[seg], a[seg]
    step = (t - np.einsum("ij,ij->i", ds, origin)) // np.einsum("ij,ij->i", ds, ds)
    return s[seg], origin + ds * step[:, None]


def _junction_vertices(xy, li, s, h, vertex, closed):
    """Vertices where the other lines on the incoming segment differ from those on
    the outgoing one, and the pairs of lines left to the pairwise method."""
    n, sl = len(closed), li[s]
    key = _hash(np.arange(n))
    count = np.bincount(li, minlength=n)
    last = np.cumsum(count) - 1
    ring = np.flatnonzero(closed & (count > 1))

    # shared segments, grouped with the lines in order; each line once
    o = np.flatnonzero(_repeated(h))
    o = o[np.argsort(h[o], kind="stable")]
    group = np.cumsum(_changes(h[o])) - 1
    shared = np.zeros(len(s), bool)
    shared[o] = np.bincount(group)[group] > 1
    o = o[shared[o]]
    o = o[_changes(h[o], sl[o])]

    # equal lines (the same segments, all shared) count once
    only = np.bincount(sl, minlength=n) > 0
    only[sl[~shared]] = False
    m = o[only[sl[o]]]
    m = m[np.argsort(sl[m], kind="stable")]
    start = np.flatnonzero(_changes(sl[m]))
    lines = key.copy()
    lines[sl[m][start]] = np.bitwise_xor.reduceat(h[m], start)
    keep = np.zeros(n, bool)
    keep[np.unique(lines, return_index=True)[1]] = True
    o = o[keep[sl[o]]]

    # lines that pass a vertex twice, and closed lines that lie wholly on another
    # line (a shared loop has no end), go to the pairwise method
    visit = np.ones(len(xy), bool)
    visit[last[ring]] = False
    v = np.flatnonzero(visit)
    at = vertex[v] ^ key[li[v]]
    v, at = v[_repeated(at)], at[_repeated(at)]
    _, inv, times = np.unique(at, return_inverse=True, return_counts=True)
    odd = np.zeros(n, bool)
    odd[li[v][times[inv.ravel()] > 1]] = True
    group = np.cumsum(_changes(h[o])) - 1
    a, b, times = _pairs(sl[o], group, keep & (odd | (only & closed)))
    loop = only[a] & closed[a] & (times == np.bincount(sl[o], minlength=n)[a])
    odd[a[loop]] = True
    odd &= keep
    a, b = a[odd[a]], b[odd[a]]
    pairs = set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))

    # the other lines on each segment of the remaining lines, in and out of each
    # vertex; a closed line continues from its end into its start
    regular = (keep & ~odd)[sl]
    o = o[regular[o]]
    group = np.cumsum(_changes(h[o])) - 1
    other = np.zeros(len(s), np.int64)
    first = np.flatnonzero(_changes(group))
    other[o] = np.bitwise_xor.reduceat(key[sl[o]], first)[group] ^ key[sl[o]]
    hin, hout = np.zeros(len(xy), np.int64), np.zeros(len(xy), np.int64)
    hout[s[regular]], hin[s[regular] + 1] = other[regular], other[regular]
    begin = last[ring] - count[ring] + 1
    hin[begin], hout[last[ring]] = hin[last[ring]], hout[begin]
    return xy[hin != hout], sorted(pairs)


def _pairs(line, group, candidate):
    """Pairs of lines on a common segment, the first a candidate, with the number of
    segments they share. The rows are sorted by segment, `group` numbers them."""
    first = np.flatnonzero(_changes(group))
    size = np.diff(np.r_[first, len(group)])
    rows = np.flatnonzero(candidate[line])
    count = size[group[rows]] - 1
    i = np.repeat(rows, count)
    j = _ranges(first[group[rows]], count)
    j += j >= i  # skip the row itself
    n = len(candidate)
    pair, times = np.unique(line[i] * n + line[j], return_counts=True)
    return pair // n, pair % n, times


def cut_line(line, tree_splitter, is_ring, shared_coords=False):
    """
    Cut a line at the junctions it passes through. A ring is first rotated to start
    at a junction. Collinear points are removed from each part.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Line to cut
    tree_splitter : STRtree or None
        Spatial index on the junction points; None if there are no junctions
    is_ring : bool
        True if the line is the ring of a polygon
    shared_coords : bool
        True to only cut at junctions that are vertices of the line

    Returns
    -------
    list of numpy.ndarray
        Coordinates of the parts
    """
    splitter = None
    if tree_splitter is not None:
        locate = np_array_bbox_points_line if shared_coords else insert_coords_in_line
        coords, splitter = locate(line, tree_splitter)
    if splitter is None:
        return [remove_collinear_points(np.array(line.coords))]
    return [
        remove_collinear_points(part) for part in fast_split(coords, splitter, is_ring)
    ]


def cut_lines_on_grid(linestrings, junctions, is_ring):
    """
    Cut quantized lines at the junctions they pass through, as `cut_line` does for
    one line, for all lines at once. A ring is first rotated to start at its first
    junction; collinear points are removed from each part.

    Parameters
    ----------
    linestrings : list of LineString or (n, 2) arrays
        Lines with integer coordinates
    junctions : list of Point or (j, 2) array
        Junctions with integer coordinates
    is_ring : numpy.ndarray of bool
        True for the lines that are the ring of a polygon

    Returns
    -------
    list of numpy.ndarray
        Coordinates of the parts, line after line
    numpy.ndarray
        Number of parts of each line
    """
    rust = _rust_core()
    n_lines = len(linestrings)
    if n_lines == 0:
        return [], np.zeros(0, dtype=np.int64)

    xy, li = _coordinates_index(linestrings)
    xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
    li = np.asarray(li, dtype=np.int32)
    if len(junctions) == 0:
        parts = [
            remove_collinear_points(np.asarray(ls, dtype=np.int64).reshape(-1, 2))
            if not hasattr(ls, "coords")
            else remove_collinear_points(np.asarray(ls.coords, dtype=np.int64))
            for ls in linestrings
        ]
        return parts, np.ones(n_lines, dtype=np.int64)

    if hasattr(junctions[0], "coords"):
        jxy = shapely.get_coordinates(junctions).astype(np.int64)
        j_geoms = junctions
    else:
        jxy = np.asarray(junctions, dtype=np.int64).reshape(-1, 2)
        j_geoms = list(shapely.points(jxy))

    # Per-line insert detection (a junction may be a vertex of line A but need
    # inserting into collinear line B). STRtree needs geometries.
    lines = np.asarray(_as_linestring_list(linestrings), dtype=object)
    tree = shapely.STRtree(j_geoms)
    line, j = tree.query(lines, predicate="intersects")
    on_vertex = np.isin(_hash(line, *jxy[j].T), _hash(li, *xy.T))
    insert = np.unique(line[~on_vertex])
    if len(insert):
        coords = np.split(xy, np.flatnonzero(np.diff(li)) + 1)
        for i in insert.tolist():
            coords[i] = insert_coords_in_line(lines[i], tree)[0]
        li = np.repeat(np.arange(len(lines), dtype=np.int32), [len(c) for c in coords])
        xy = np.concatenate(coords).astype(np.int64)

    junction_hashes = _hash(*jxy.T)

    # After any inserts, Rust can cut on packed arrays (no further Shapely).
    if rust is not None and len(insert) == 0:
        try:
            parts_xy, offsets, n_parts = rust.cut_lines_on_grid(
                xy, li, jxy, np.asarray(is_ring, dtype=bool)
            )
            parts = [
                parts_xy[int(offsets[i]) : int(offsets[i + 1])]
                for i in range(len(offsets) - 1)
            ]
            return parts, np.asarray(n_parts, dtype=np.int64)
        except Exception:
            logger.debug(
                "rust cut_lines_on_grid failed; using NumPy path", exc_info=True
            )

    # rotate a ring to start at its first junction
    split = np.isin(_hash(*xy.T), junction_hashes)
    count = np.bincount(li, minlength=n_lines)
    pos = np.arange(len(xy)) - (np.cumsum(count) - count)[li]
    first = np.zeros(n_lines, np.intp)
    first[li[split][::-1]] = pos[split][::-1]
    shift, n = (first * is_ring)[li], count[li] - 1
    rotated = np.where(pos == n, shift, (pos + shift) % np.maximum(n, 1))
    order = np.arange(len(xy)) + np.where(shift > 0, rotated - pos, 0)
    xy, split = xy[order], split[order]

    # split at the junctions inside each line: such a junction ends one part and
    # starts the next
    take = np.repeat(np.arange(len(xy)), 1 + (split & (pos > 0) & (pos < n)))
    start = np.r_[True, take[1:] != take[:-1] + 1]
    start |= np.r_[True, li[take][1:] != li[take][:-1]]
    xy = xy[take]

    # remove collinear points inside each part
    mid = np.flatnonzero(~start[1:-1] & ~start[2:]) + 1
    u, w = xy[mid] - xy[mid - 1], xy[mid + 1] - xy[mid - 1]
    cross = u[:, 0] * w[:, 1] - w[:, 0] * u[:, 1]
    keep = np.ones(len(xy), bool)
    keep[mid[cross == 0]] = False
    parts = _split(xy[keep], np.flatnonzero(start[keep])[1:])
    return parts, np.bincount(li[take][start], minlength=n_lines)


def fast_split(line, splitter, is_ring):
    """
    Split a LineString (numpy.array) with a Point or MultiPoint.
    This function is a replacement for the shapely.ops.split function, but faster.

    Parameters
    ----------
    line : numpy.array
        numpy array with coordinates that you like to be split
    splitter : numpy.array
        numpy array with coordinates on which the line should be tried splitting
    is_ring : bool
        True if the line represents a ring. In this case, for the first point found, the
        linestring will be rotated rather than split. For consecutive points it will be
        split.

    Returns
    -------
    list of numpy.array
        If more than 1 item, the line was split. Each item in the list is an
        array of coordinates.
    """

    # previously did convert geometries of coordinates from LineString and (Multi)Point
    # to numpy arrays. This function now expect this as input to save time.
    # line = np.array(line.coords)
    # splitter = np.array([x for pt in splitter for x in pt.coords])

    # locate index of splitter coordinates in linestring
    tol = 1e8
    splitter_indices = np.flatnonzero(
        np.isin(
            asvoid(np.around(line * tol).astype(np.int64))[:, 0],
            asvoid(np.around(splitter * tol).astype(np.int64))[:, 0],
        )
    )

    # For a ring, rotate rather than split for the first splitter_index
    # Remark: the start and end coordinate of a ring are the same, so keep it that way
    if is_ring and len(splitter_indices) > 0 and splitter_indices[0] != 0:
        first_index = splitter_indices[0]
        line = line[:-1]
        line = np.roll(line, -first_index, axis=0)
        line = np.append(line, [line[0]], axis=0)
        splitter_indices = splitter_indices[1:]
        splitter_indices = splitter_indices - first_index
    # compute the indices on which to split the line
    # cannot split on first or last index of linestring
    splitter_indices = splitter_indices[
        (splitter_indices < (line.shape[0] - 1)) & (splitter_indices > 0)
    ]

    # split the linestring where each sub-array includes the split-point
    # create a new array with the index elements repeated
    tmp_indices = np.zeros(line.shape[0], dtype=np.intp)
    tmp_indices[splitter_indices] = 1
    tmp_indices += 1
    ls_xy = np.repeat(line, tmp_indices, axis=0)

    # update indices to account for the changed array
    splitter_indices = splitter_indices + np.arange(1, len(splitter_indices) + 1)

    # split using the indices as usual
    slines = np.split(ls_xy, splitter_indices, axis=0)

    return slines


def signed_area(ring):
    """
    Compute the signed area of a ring (polygon)

    Note: implementation is numpy variant of shapely's version:
    https://github.com/Toblerity/Shapely/blob/master/shapely/algorithms/cga.py

    Parameters
    ----------
    ring : numpy.array
        coordinates representing an exterior or inner ring

    Returns
    -------
    float
        the signed area
    """
    xs, ys = ring.T
    signed_area = (xs * (np.roll(ys, -1) - np.roll(ys, +1))).sum() / 2
    return signed_area


def is_ccw(ring):
    """
    Provide information if a given ring is clockwise or counterclockwise.

    Parameters
    ----------
    ring : numpy.array
        coordinates representing an exterior or inner ring

    Returns
    -------
    boolean
        True if ring is counterclockwise and False if ring is clockwise
    """
    return signed_area(ring) >= 0.0


def properties_foreign(objects):
    """
    Try to parse the object properties as foreign members. Reserved keys are:
    [`type`, `bbox`, `coordinates`, `geometries`, `geometry`, `properties`, `features`]

    If these keys are detected they will not be set as a foreign member and will remain
    nested within properties.

    Only if the

    Parameters
    ----------
    objects : [type]
        [description]
    """
    reserved_keys = [
        "type",
        "bbox",
        "coordinates",
        "geometries",
        "geometry",
        "properties",
        "features",
        "arcs",
    ]
    reserved_keys_used = bool(
        set(objects[0]["properties"].keys()).intersection(reserved_keys)
    )

    for obj in objects:
        reserved_keys = False
        for k, v in list(obj["properties"].items()):
            if not reserved_keys_used or k in reserved_keys:
                obj[k] = v
                obj["properties"].pop(k, None)
        if not reserved_keys_used:
            obj.pop("properties", None)
    return objects


def bounds(arr):
    """
    Returns a (minx, miny, maxx, maxy) tuple (float values) that bounds the object.

    Parameters
    ----------
    arr : np.array
        array to get bounds from

    Returns
    -------
    tuple
        (minx, miny, maxx, maxy)
    """
    if not len(arr):
        return []
    first = arr[0]
    if hasattr(first, "coords"):
        # Shapely 2 vectorized path — avoids per-geom `.coords` Python walks.
        xy = shapely.get_coordinates(arr)
        if xy.size == 0:
            return []
        return (
            float(np.nanmin(xy[:, 0])),
            float(np.nanmin(xy[:, 1])),
            float(np.nanmax(xy[:, 0])),
            float(np.nanmax(xy[:, 1])),
        )
    stacked = np.vstack([np.asarray(a, dtype=np.float64).reshape(-1, 2) for a in arr])
    return (
        float(np.nanmin(stacked[:, 0])),
        float(np.nanmin(stacked[:, 1])),
        float(np.nanmax(stacked[:, 0])),
        float(np.nanmax(stacked[:, 1])),
    )


def compare_bounds(b0, b1):
    """
    Function that compares two bounds with each other. Returns the max bound.

    Parameters
    ----------
    b0 : tuple
        tuple of xmin, ymin, xmax, ymax
    b1 : tuple
        tuple of xmin, ymin, xmax, ymax

    Returns
    -------
    tuple
        min of mins and max of maxs
    """

    if len(b0) and len(b1):
        bounds = (
            min(b0[0], b1[0]),
            min(b0[1], b1[1]),
            max(b0[2], b1[2]),
            max(b0[3], b1[3]),
        )
    elif len(b0) and not len(b1):
        bounds = b0
    elif not len(b0) and len(b1):
        bounds = b1
    else:
        bounds = []
    return bounds


def np_array_from_lists(nested_lists):
    """
    Function to create numpy array from nested lists. The shape of the numpy array
    are the number of nested lists (rows) x the length of the longest nested list
    (columns). Rows that contain less values are filled with np.nan values.

    Parameters
    ----------
    nested_lists : list of lists
        list containing nested lists of different sizes.

    Returns
    -------
    numpy.ndarray
        array created from nested lists, np.nan is used to fill the array
    """

    np_array = np.array(list(itertools.zip_longest(*nested_lists, fillvalue=np.nan))).T
    return np_array


def lists_from_np_array(np_array):
    """
    Function to convert numpy array to list, where elements set as np.nan
    are filtered
    """

    nested_lists = [obj[~np.isnan(obj)].astype(np.int64).tolist() for obj in np_array]
    return nested_lists


def np_array_from_arcs(arcs):
    max_len_arc = len(max(arcs, key=len))
    no_arcs = len(arcs)
    np_array = np.empty((no_arcs, max_len_arc, 2))
    np_array.fill(np.nan)
    for idx in range(no_arcs):
        np_array[idx, 0 : len(arcs[idx])] = arcs[idx]
    return np_array


def arc_coordinates(arcs, transform=None):
    """
    Coordinates of each arc as an array of its own, so that memory follows the number
    of coordinates instead of the number of arcs times the longest arc. With a
    transform, the arcs are delta-encoded integers on its grid; they are decoded and
    scaled back.

    Parameters
    ----------
    arcs : list of lists
        Arcs of a topology
    transform : dict, optional
        TopoJSON transform (`scale` and `translate`) of the arcs

    Returns
    -------
    list of numpy.ndarray
        (n, 2) float coordinates of each arc
    """
    if not len(arcs):
        return []
    if transform is None:
        xy = np.concatenate(arcs).astype(float)
        lengths = np.fromiter(map(len, arcs), np.intp, len(arcs))
    else:
        xy, lengths = _decoded(arcs)
        xy = xy * transform["scale"] + transform["translate"]
    return _split(xy, np.cumsum(lengths)[:-1])


def dequantize(np_arcs, scale, translate):
    dequantized_arcs = np_arcs.cumsum(axis=1) * scale + translate
    return dequantized_arcs


def get_matches(geoms, tree_idx):
    """
    Function to return the indices of the rtree that intersects with the input geometries

    Parameters
    ----------
    geoms : list
        list of geometries to compare against the STRtree
    tree_idx : STRtree
        a STRtree indexing object

    Returns
    -------
    list
        list of tuples, where the key of each tuple is the linestring index and the
        value of each key is a list of junctions intersecting bounds of linestring.
    """

    # find near linestrings by querying tree and use query items to collect indices.
    matches = []
    for idx_ls, obj in enumerate(geoms):
        intersect_ls = strtree_query_index(tree_idx, obj, geoms)
        if len(intersect_ls):
            matches.extend([[[idx_ls], intersect_ls]])
    return matches


def select_unique(data):
    """
    Function to return unique pairs within a numpy array.
    Example: input as [[1,2], [2,1]] will return as [[1,2]]

    Parameters
    ----------
    data : numpy.array
        2 dimensional array, where each row is a couple

    Returns
    -------
    numpy.array
        2 dimensional array, where each row is unique.
    """

    sorted_data = data[np.lexsort(data.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    return sorted_data[row_mask]


def select_unique_combs(linestrings):
    """
    Given a set of input linestrings will create unique couple combinations.
    Each combination created contains a couple of two linestrings where the envelope
    overlaps each other.
    Linestrings with non-overlapping envelopes are not returned as combination.

    Parameters
    ----------
    linestrings : list of LineString
        list where each item is a shapely LineString

    Returns
    -------
    numpy.array
        2 dimensional array, with on each row the index combination
        of a unique couple LineString with overlapping envelope
    """

    # create spatial index
    tree_idx = STRtree(linestrings)
    # get index of linestrings intersecting each linestring
    idx_match = get_matches(linestrings, tree_idx)

    # make combinations of unique possibilities
    combs = []
    for idx_comb in idx_match:
        combs.extend(list(itertools.product(*idx_comb)))
    combs = np.array(combs)
    combs.sort(axis=1)
    combs = select_unique(combs)

    uniq_line_combs = combs[(np.diff(combs, axis=1) != 0).flatten()]

    return uniq_line_combs, tree_idx


def validate_transform(transform):
    """
    Validate a TopoJSON transform and return it as a new dict with float values. A
    transform that is not valid raises a `ValueError`.

    Parameters
    ----------
    transform : dict
        TopoJSON transform with keys `scale` (`[kx, ky]`, both positive) and
        `translate` (`[x0, y0]`).

    Returns
    -------
    dict
        The transform with float values for `scale` and `translate`
    """

    try:
        kx, ky = (float(v) for v in transform["scale"])
        x0, y0 = (float(v) for v in transform["translate"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "A transform should be a dict of the form "
            "{'scale': [kx, ky], 'translate': [x0, y0]}, "
            f"got: {transform!r}"
        ) from None
    if not all(np.isfinite([kx, ky, x0, y0])) or kx <= 0 or ky <= 0:
        raise ValueError(
            "The values of a transform should be finite and the scale values "
            f"should be positive, got: {transform!r}"
        )
    return {"scale": [kx, ky], "translate": [x0, y0]}


def remove_spikes(line):
    """
    Remove spikes from a quantized line: vertices where the line turns back over the
    same segment. End points of open lines are kept and closed rings stay closed. A
    line that would collapse is returned unchanged.

    Parameters
    ----------
    line : numpy.ndarray
        (n, 2) integer coordinates without consecutive duplicates

    Returns
    -------
    numpy.ndarray
        coordinates without spikes
    """
    closed = len(line) > 3 and (line[0] == line[-1]).all()
    pts = line[:-1] if closed else line
    # each round removes at least one point, so len(pts) rounds always suffice
    for _ in range(len(pts)):
        d1 = pts - np.roll(pts, 1, axis=0)
        d2 = np.roll(d1, -1, axis=0)
        tip = (d1[:, 0] * d2[:, 1] == d1[:, 1] * d2[:, 0]) & ((d1 * d2).sum(1) < 0)
        if not closed:
            tip[[0, -1]] = False
        # remove all tips at once, except a tip directly after another tip
        tip &= ~np.roll(tip, 1)
        if not tip.any():
            break
        pts = pts[~tip]
        # removing a tip can leave two equal consecutive points
        same = (pts == np.roll(pts, 1, axis=0)).all(axis=1)
        if not closed:
            same[0] = False
        pts = pts[~same]
        if len(pts) < (3 if closed else 2):
            return line
    return np.vstack([pts, pts[:1]]) if closed else pts


def _lines_with_spikes(coords, starts, counts):
    """
    Indices of the lines that contain a spike, for all lines at once.

    `coords` holds the vertices of all lines after each other; line `i` starts at
    `starts[i]` and has `counts[i]` vertices.
    """
    d = np.diff(coords, axis=0)
    ends = starts + counts - 1
    # the turn at vertex v is made by segments v-1 and v; for each line, take the
    # turns at its inner vertices and, for a closed ring, the one at its first vertex
    inner = np.ones(len(coords), dtype=bool)
    inner[starts[counts > 0]] = False
    inner[ends[counts > 0]] = False
    v = np.flatnonzero(inner)
    ring = counts > 3
    ring[ring] = (coords[starts[ring]] == coords[ends[ring]]).all(axis=1)
    d1 = np.concatenate([d[v - 1], d[ends[ring] - 1]])
    d2 = np.concatenate([d[v], d[starts[ring]]])
    line = np.concatenate(
        [np.repeat(np.arange(len(counts)), counts)[v], np.flatnonzero(ring)]
    )
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return np.unique(line[(cross == 0) & (np.einsum("ij,ij->i", d1, d2) < 0)])


def quantize(linestrings, bbox, quant_factor=1e5, transform=None):
    """
    Function that applies quantization. Quantization removes information by reducing
    the precision of each coordinate, effectively snapping each point to a regular grid.

    Parameters
    ----------
    linestrings : list of shapely.geometry.LineStrings
        LineStrings that will be quantized
    bbox : tuple
        Bounding box (`x0`, `y0`, `x1`, `y1`) used to derive the grid. Ignored when
        `transform` is given.
    quant_factor : int
        Quantization factor. Normally this varies between 1e4, 1e5, 1e6. Where a
        higher number means a bigger grid where the coordinates can snap to. Ignored
        when `transform` is given.
    transform : dict, optional
        Fixed TopoJSON transform (`{"scale": [kx, ky], "translate": [x0, y0]}`) that
        defines the grid. When given, the grid does not depend on `bbox`.

    Returns
    -------
    list
        quantized linestrings
    dict
        `transform`, scale (`kx`, `ky`) and translation (`x0`, `y0`) values
    """

    if transform is not None:
        transform = validate_transform(transform)
        kx, ky = transform["scale"]
        x0, y0 = transform["translate"]
    else:
        x0, y0, x1, y1 = bbox

        try:
            kx = 1 if (x1 - x0) == 0 else (x1 - x0) / (quant_factor - 1)
            ky = 1 if (y1 - y0) == 0 else (y1 - y0) / (quant_factor - 1)
        except ZeroDivisionError:
            kx, ky = 1, 1
    transform_ = {"scale": [kx, ky], "translate": [x0, y0]}
    n = len(linestrings)
    if n == 0:
        return linestrings, transform_

    # all vertices of all lines at once, with the index of their line
    is_geom = hasattr(linestrings[0], "coords")
    if is_geom:
        xy, idx = shapely.get_coordinates(linestrings, return_index=True)
    else:
        arrays = [np.asarray(ls, dtype=float).reshape(-1, 2) for ls in linestrings]
        xy = np.concatenate(arrays)
        idx = np.repeat(np.arange(n), [len(a) for a in arrays])
    coords = np.round((xy - [x0, y0]) / [kx, ky]).astype(np.int64)

    # remove repeated coordinates, unless a line would become a single point
    repeated = np.zeros(len(coords), dtype=bool)
    repeated[1:] = (idx[1:] == idx[:-1]) & (coords[1:] == coords[:-1]).all(axis=1)
    repeated &= (np.bincount(idx[~repeated], minlength=n) > 1)[idx]
    coords, idx = coords[~repeated], idx[~repeated]
    counts = np.bincount(idx, minlength=n)
    starts = np.cumsum(counts) - counts
    lines = _split(coords, starts[1:])

    # snapping to the grid can create spikes; only a few lines have them
    spiky = _lines_with_spikes(coords, starts, counts)
    for i in spiky:
        lines[i] = remove_spikes(lines[i])
    if len(spiky):
        counts = np.array([len(line) for line in lines])
        coords, idx = np.concatenate(lines), np.repeat(np.arange(n), counts)

    # Geom input → LineStrings (historical API / ops tests). Array input (Extract
    # rings, point coordinates) stays as arrays; Join batch-builds LineStrings.
    out = list(lines)
    if is_geom:
        out = _as_linestring_list(out)
    return out, transform_


def simplify(
    linestrings,
    epsilon,
    algorithm="dp",
    package="simplification",
    input_as="linestring",
    prevent_oversimplify=True,
):
    """
    Function that simplifies linestrings. The goal of line simplification is to reduce
    the number of points by deleting some trivial points, but without destroying the
    essential shape of the lines in the process.

    One can choose between the Douglas-Peucker ["dp"] algorithm (which simplifies
    a line based upon vertical interval) and Visvalingam-Whyatt ["vw"] (which
    progressively removes points with the least-perceptible change).


    Docs
    * https://observablehq.com/@lemonnish/minify-topojson-in-the-browser
    * https://github.com/topojson/topojson-simplify#planarTriangleArea
    * https://www.jasondavies.com/simplify/
    * https://bost.ocks.org/mike/simplify/
    * https://pdfs.semanticscholar.org/9877/cdf50a15367bcb86649b67df8724425c5451.pdf

    Parameters
    ----------
    linestrings : list of shapely.geometry.LineStrings
        LineStrings that will be simplified
    epsilon : int
        Simplification factor. Normally this varies 1.0, 0.1 or 0.001 for "dp" and
        30-100 for "vw".
    algorithm : str, optional
        Choose between `dp` for Douglas-Peucker and `vw` for Visvalingam-Whyatt.
        Defaults to `dp`, as its evaluation maintains to be good (Shi, W. &
        Cheung, C., 2006).
    package : str, optional
        Choose between `simplification` or `shapely`. Both packages contains
        simplification algorithms (`shapely` only `dp`, and `simplification` both `dp`
        and `vw`).
    input_as : str, optional
        Choose between `linestring` or `array`. This function is being called from
        different locations with different input types. Choose `linestring` if the input
        type are shapely.geometry.LineString or `array` if the input is a list of
        coordinate arrays

    Returns
    -------
    list of shapely.geometry.LineStrings or ndarrays, depending on the type of the input
        LineStrings that are simplified
    """
    list_arcs = None

    if package == "shapely":
        if algorithm == "vw":
            msg = (
                "You need to set `simplify_with='simplification'` to use the ",
                "Visvalingam-Whyatt (`vw`) algorithm. This package is optional and ",
                "if not installed, install with `pip install simplification`. ",
                "Continue with Douglas-Peucker (`dp`) algorithm instead.",
            )
            logger.warning("".join(msg))
        keep_valid = prevent_oversimplify
        if input_as == "array":
            # all arcs at once
            lengths = np.fromiter(map(len, linestrings), np.intp, len(linestrings))
            index = np.repeat(np.arange(len(linestrings)), lengths)
            lines = shapely.linestrings(np.concatenate(linestrings), indices=index)
            lines = shapely.simplify(lines, epsilon, preserve_topology=keep_valid)
            xy, index = shapely.get_coordinates(lines, return_index=True)
            count = np.bincount(index, minlength=len(lines))
            list_arcs = [a.tolist() for a in _split(xy, np.cumsum(count)[:-1])]
        elif input_as == "linestring":
            lines = np.asarray(linestrings, dtype=object)
            lines = shapely.simplify(lines, epsilon, preserve_topology=keep_valid)
            linestrings[:] = lines.tolist()
            list_arcs = linestrings
    elif package == "simplification":
        from simplification import cutil

        if algorithm == "vw" and prevent_oversimplify:
            alg = cutil.simplify_coords_vwp
        elif algorithm == "vw" and not prevent_oversimplify:
            alg = cutil.simplify_coords_vw
        elif algorithm == "dp" and prevent_oversimplify:
            msg = (
                "The Douglas-Peucker algorithm from the `simplification` package ",
                "has no options to prevent oversimplification. Use Visvalingam-",
                "Whyatt (`vw`) algorithm when using the simplification package if ",
                "oversimplification should be prevented or use the Douglas-Peucker ",
                "algorithm from `shapely` package to prevent oversimplification. ",
                "Continue without prevention of oversimplification.",
            )
            logger.warning("".join(msg))
            alg = cutil.simplify_coords
        else:
            alg = cutil.simplify_coords
        if input_as == "array":
            list_arcs = []
            for ls in linestrings:
                list_arcs.append(alg(np.asarray(ls, dtype=float), epsilon).tolist())
        elif input_as == "linestring":
            for idx, ls in enumerate(linestrings):
                coords_to_simp = np.array(ls.coords)
                simple_ls = alg(coords_to_simp, epsilon)
                linestrings[idx] = geometry.LineString(simple_ls)
            list_arcs = linestrings
    else:
        raise NameError(
            f"Could not recognize parameter for `simplify_with`. Choose between \
                'shapely' or 'simplification'. '{package}' was given"
        )
    return list_arcs


def dp_weights(xy, starts, ends):
    """
    Weight of each vertex of lines for Douglas-Peucker: simplifying with a tolerance
    `epsilon` keeps exactly the vertices with a weight larger than `epsilon`, as
    `shapely.simplify` with `preserve_topology=False` does.

    Douglas-Peucker is nested: a vertex is kept when its distance to the segment of
    its section is larger than `epsilon` and the vertex that split off its section is
    kept as well. Its weight is therefore the smallest distance along its chain of
    splits. The splits are found level by level, for all sections of all lines at
    once.

    Parameters
    ----------
    xy : numpy.ndarray
        Coordinates of all lines after each other
    starts, ends : numpy.ndarray
        Index in `xy` of the first and the last vertex of each line

    Returns
    -------
    numpy.ndarray
        Weight of each vertex; `inf` for the first and the last vertex of each line
    """
    weight = np.full(len(xy), np.inf)
    s, e = np.asarray(starts), np.asarray(ends)
    cap = np.full(len(s), np.inf)
    for _ in range(len(xy)):
        n = e - s - 1
        s, e, cap, n = s[n > 0], e[n > 0], cap[n > 0], n[n > 0]
        if not len(s):
            break
        k = _ranges(s + 1, n)
        section = np.repeat(np.arange(len(s)), n)
        d = _segment_distance(xy[k], xy[s][section], xy[e][section])
        # the first vertex at the largest distance in each section, as GEOS
        dmax = np.maximum.reduceat(d, np.cumsum(n) - n)
        at = np.flatnonzero(d == dmax[section])
        m = k[at[_changes(section[at])]]
        weight[m] = np.minimum(dmax, cap)
        s, e, cap = np.r_[s, m], np.r_[m, e], np.r_[weight[m], weight[m]]
    return weight


def _segment_distance(p, a, b):
    """Distance of points `p` to the segments `a`-`b`, in the same steps as GEOS
    (`Distance::pointToSegment`), so that equal distances are decided the same way."""
    (px, py), (ax, ay), (bx, by) = p.T, a.T, b.T
    len2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    safe = np.where(len2 > 0, len2, 1)
    r = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / safe
    s = ((ay - py) * (bx - ax) - (ax - px) * (by - ay)) / safe
    to_a = np.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    to_b = np.sqrt((px - bx) * (px - bx) + (py - by) * (py - by))
    inside = np.abs(s) * np.sqrt(len2)
    return np.where((len2 == 0) | (r <= 0), to_a, np.where(r >= 1, to_b, inside))


def simplify_keep(linestrings, keep, algorithm="dp"):
    """
    Simplify lines to a share of their vertices: the share `keep` of the inner
    vertices that the algorithm removes last is kept, and the first and the last
    vertex of each line. The result is that of `simplify` with the tolerance that is
    returned. Vertices with an equal weight are kept or removed together, so the share
    can be a little off.

    Douglas-Peucker uses the weight of each vertex (`dp_weights`). Visvalingam-Whyatt
    searches the tolerance of the package simplification (`_vw_tolerance`).

    Parameters
    ----------
    linestrings : list of numpy.ndarray
        Coordinates of the lines
    keep : float
        Share of the inner vertices to keep, between 0 and 1
    algorithm : str
        `dp` for Douglas-Peucker or `vw` for Visvalingam-Whyatt

    Returns
    -------
    list of list
        Coordinates of the simplified lines
    float
        The tolerance that gives the same result with `simplify`
    """
    count = np.fromiter(map(len, linestrings), np.intp, len(linestrings))
    inner = int(np.maximum(count - 2, 0).sum())
    n = min(round(keep * inner), inner)
    if algorithm == "vw":
        epsilon = _vw_tolerance(linestrings, n)
        simple = simplify(
            linestrings,
            epsilon,
            algorithm="vw",
            package="simplification",
            input_as="array",
            prevent_oversimplify=False,
        )
        return simple, epsilon
    xy = np.concatenate(linestrings)
    ends = np.cumsum(count) - 1
    weight = dp_weights(xy, ends - count + 1, ends)
    finite = weight[np.isfinite(weight)]
    epsilon = -np.inf if n == len(finite) else np.partition(finite, -n - 1)[-n - 1]
    kept = weight > epsilon
    count = np.bincount(
        np.repeat(np.arange(len(count)), count)[kept], minlength=len(count)
    )
    return [a.tolist() for a in _split(xy[kept], np.cumsum(count)[:-1])], epsilon


def _vw_tolerance(linestrings, target):
    """
    The largest tolerance of Visvalingam-Whyatt (package simplification) at which at
    least `target` inner vertices are left, so that a lower one keeps too many.

    VW is nested: from the result of a tolerance, a larger one gives what it gives
    from the input. So each step that keeps enough vertices continues from its
    result, and VW stops at once on a line whose smallest triangle is larger than the
    tolerance, so only the other lines are simplified. The search starts at a
    quantile of the triangle areas and interpolates between its bounds in
    log(tolerance) against log(vertices left), halving every other step.
    """
    from simplification.cutil import simplify_coords_vw_idx

    lines = [np.ascontiguousarray(a, float) for a in linestrings if len(a) > 2]
    left = sum(len(a) - 2 for a in lines)
    if target >= left:
        return -1.0  # below every area: all vertices stay

    def smallest(lines):
        """The smallest triangle area of each line, and all the areas."""
        if not lines:
            return np.zeros(0), np.zeros(0)
        xy = np.concatenate(lines)
        count = np.fromiter(map(len, lines), np.intp, len(lines))
        start = np.cumsum(count) - count
        a, b, c = xy[:-2], xy[1:-1], xy[2:]
        area = (
            np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            )
            / 2
        )
        # triangles across two lines do not count
        area[np.cumsum(count)[:-1] - 2] = area[np.cumsum(count)[:-1] - 1] = np.inf
        return np.minimum.reduceat(area, start), area[np.isfinite(area)]

    least, area = smallest(lines)
    floor = area[area > 0].min(initial=1.0) / 2
    lo, n_lo, hi, n_hi = 0.0, left, np.inf, 0
    for step in range(64):
        if n_lo == target or hi <= lo * (1 + 1e-6):
            break
        if step == 0:
            tolerance = max(np.quantile(area, 1 - target / left), floor)
        elif not np.isfinite(hi):
            tolerance = max(lo, floor) * 4
        elif step % 2 or lo <= 0 or n_hi <= 0:
            tolerance = np.sqrt(max(lo, floor) * hi)
        else:
            f = np.log(n_lo / target) / np.log(n_lo / n_hi)
            tolerance = lo * (hi / lo) ** min(max(f, 0.05), 0.95)
        simple = list(lines)
        for i in np.flatnonzero(least <= tolerance).tolist():
            simple[i] = lines[i][
                np.asarray(simplify_coords_vw_idx(lines[i], tolerance))
            ]
        n = sum(len(a) - 2 for a in simple)
        if n >= target:
            lo, n_lo = tolerance, n
            lines = [a for a in simple if len(a) > 2]
            least = smallest(lines)[0]
        else:
            hi, n_hi = tolerance, n
    return lo


def simplify_coverage(linestrings, polygons, epsilon, exterior_cw=None):
    """
    Simplify the rings of polygons together as a coverage, with GEOS
    (`shapely.coverage_simplify`, Visvalingam-Whyatt): an edge shared by two polygons
    is simplified once, so that they stay matched, and each ring keeps at least three
    points. The polygons should form a valid coverage: no overlaps, and the vertices
    of shared edges equal.

    Parameters
    ----------
    linestrings : list of LineString
        Rings and lines; the rings of the polygons are replaced in place
    polygons : list of list of int
        Index into `linestrings` of the rings of each polygon, exterior first
    epsilon : float
        Tolerance of `shapely.coverage_simplify`
    exterior_cw : bool, optional
        Orientation of the exterior rings in the result; `None` keeps it as is

    Returns
    -------
    list of LineString
        The linestrings, with the rings simplified
    """
    if not polygons:
        return linestrings
    rings = np.concatenate(polygons)
    xy, ring = shapely.get_coordinates(
        [linestrings[i] for i in rings], return_index=True
    )
    polygon = np.repeat(np.arange(len(polygons)), [len(p) for p in polygons])
    shapes = shapely.polygons(shapely.linearrings(xy, indices=ring), indices=polygon)
    shapes = shapely.coverage_simplify(shapes, epsilon)
    if exterior_cw is not None:
        shapes = shapely.orient_polygons(shapes, exterior_cw=exterior_cw)
    xy, ring = shapely.get_coordinates(shapely.get_rings(shapes), return_index=True)
    for i, line in zip(rings.tolist(), shapely.linestrings(xy, indices=ring)):
        linestrings[i] = line
    return linestrings


def _arc_offsets(counts):
    """Start index of each run in a packed buffer whose run lengths are ``counts``."""
    starts = np.empty(len(counts), dtype=np.intp)
    if len(counts):
        starts[0] = 0
        if len(counts) > 1:
            np.cumsum(counts[:-1], out=starts[1:])
    return starts


def _offsets_within(counts):
    """``0 .. n_i-1`` repeated for each run length in ``counts`` (packed)."""
    total = int(np.sum(counts))
    if total == 0:
        return np.empty(0, dtype=np.intp)
    idx = np.arange(total, dtype=np.intp)
    return idx - np.repeat(_arc_offsets(counts), counts)


def segmentize(xy, max_segment_length):
    """
    Densify an open polyline so no edge is longer than ``max_segment_length``.

    Endpoints stay fixed. Used before Chaikin so the corner-cutting radius tracks a
    chosen ground length (Smoothify-style ``segment_length``), without dissolving
    shared borders. Arcs already shorter than the cap are unchanged.

    Parameters
    ----------
    xy : array-like of shape (n, 2)
        Arc coordinates in float space
    max_segment_length : float
        Maximum edge length in the same units as ``xy`` (must be > 0)

    Returns
    -------
    numpy.ndarray
        Densified (m, 2) coordinates; ``xy[0]`` and ``xy[-1]`` unchanged
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (n, 2) coordinates, got shape {pts.shape}")
    if max_segment_length <= 0:
        raise ValueError(f"max_segment_length must be > 0, got: {max_segment_length!r}")
    if len(pts) < 2:
        return pts.copy()

    dxy = np.diff(pts, axis=0)
    dist = np.linalg.norm(dxy, axis=1)
    # Subdivisions per edge (1 = keep the end vertex only).
    n_sub = np.ones(len(dist), dtype=np.intp)
    long = dist > max_segment_length
    if np.any(long):
        n_sub[long] = np.ceil(dist[long] / max_segment_length).astype(np.intp)
    # Output = start vertex + sum(n_sub) end/interior vertices.
    out_len = 1 + int(n_sub.sum())
    out = np.empty((out_len, 2), dtype=np.float64)
    out[0] = pts[0]
    cursor = 1
    for i, n in enumerate(n_sub.tolist()):
        p0 = pts[i]
        p1 = pts[i + 1]
        if n == 1:
            out[cursor] = p1
            cursor += 1
            continue
        # Embed interiors then the endpoint.
        for k in range(1, n):
            out[cursor] = p0 + (p1 - p0) * (k / n)
            cursor += 1
        out[cursor] = p1
        cursor += 1
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def segmentize_many(arcs, max_segment_length):
    """Densify many float arcs; same contract as :func:`segmentize` per arc."""
    if max_segment_length <= 0:
        raise ValueError(f"max_segment_length must be > 0, got: {max_segment_length!r}")
    return [segmentize(arc, max_segment_length) for arc in arcs]


def chaikin(xy, iterations=2):
    """
    Chaikin corner-cutting on an open polyline with endpoints fixed (Chaikin 1974).

    Each iteration replaces every segment with its 1/4 and 3/4 points and keeps the
    first and last coordinates exactly, converging toward a quadratic B-spline. Arcs
    with fewer than three vertices are returned unchanged (a two-point shared edge
    stays a straight junction-to-junction link).

    Parameters
    ----------
    xy : array-like of shape (n, 2)
        Arc coordinates in float space
    iterations : int
        Number of corner-cutting passes (growth ~2x per pass)

    Returns
    -------
    numpy.ndarray
        Smoothed (m, 2) coordinates; ``xy[0]`` and ``xy[-1]`` unchanged
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (n, 2) coordinates, got shape {pts.shape}")
    if len(pts) < 3 or iterations <= 0:
        return pts.copy()
    rust = _rust_core()
    if rust is not None:
        try:
            return np.asarray(rust.chaikin(pts, int(iterations)), dtype=np.float64)
        except Exception:
            logger.debug("rust chaikin failed; using NumPy path", exc_info=True)
    packed = chaikin_many([pts], iterations=iterations)
    return packed[0]


def _chaikin_iteration_packed(xy, lengths):
    """One Chaikin pass on a packed ``(V, 2)`` buffer with per-arc ``lengths``."""
    starts = _arc_offsets(lengths)
    n_out = lengths * 2
    out_starts = _arc_offsets(n_out)
    out = np.empty((int(n_out.sum()), 2), dtype=np.float64)
    # Pin endpoints to the iteration inputs (then to the originals across passes).
    out[out_starts] = xy[starts]
    out[out_starts + n_out - 1] = xy[starts + lengths - 1]

    seg_counts = lengths - 1
    total_seg = int(seg_counts.sum())
    if total_seg == 0:
        return out, n_out

    within = _offsets_within(seg_counts)
    left = np.repeat(starts, seg_counts) + within
    p0 = xy[left]
    p1 = xy[left + 1]
    q_idx = np.repeat(out_starts, seg_counts) + 1 + 2 * within
    out[q_idx] = 0.75 * p0 + 0.25 * p1
    out[q_idx + 1] = 0.25 * p0 + 0.75 * p1
    return out, n_out


def chaikin_many(arcs, iterations=2):
    """
    Chaikin corner-cutting on many open polylines via one packed float buffer.

    Short arcs (fewer than three vertices) are returned unchanged. Endpoints of
    every arc stay fixed. Prefer this over a Python loop of :func:`chaikin` when
    smoothing wide meshes (millions of vertices).
    """
    if iterations <= 0:
        return [np.asarray(a, dtype=np.float64).copy() for a in arcs]

    arrays = []
    for arc in arcs:
        pts = np.asarray(arc, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"expected (n, 2) coordinates, got shape {pts.shape}")
        arrays.append(pts)

    out: list[np.ndarray | None] = [None] * len(arrays)
    long_idx: list[int] = []
    long_arcs: list[np.ndarray] = []
    for i, pts in enumerate(arrays):
        if len(pts) < 3:
            out[i] = pts.copy()
        else:
            long_idx.append(i)
            long_arcs.append(pts)

    if not long_arcs:
        return [a if a is not None else arrays[i].copy() for i, a in enumerate(out)]

    lengths = np.fromiter((len(a) for a in long_arcs), np.intp, len(long_arcs))
    xy = np.concatenate(long_arcs)
    # Snapshot original endpoints so every iteration pins to the input ends.
    starts0 = _arc_offsets(lengths)
    ends0 = xy[starts0 + lengths - 1].copy()
    starts_xy0 = xy[starts0].copy()

    for _ in range(iterations):
        xy, lengths = _chaikin_iteration_packed(xy, lengths)
        # Re-pin to the pre-smooth endpoints (numerical drift guard).
        out_starts = _arc_offsets(lengths)
        xy[out_starts] = starts_xy0
        xy[out_starts + lengths - 1] = ends0

    starts = _arc_offsets(lengths)
    for i, src in enumerate(long_idx):
        s = int(starts[i])
        out[src] = xy[s : s + int(lengths[i])].copy()
    return [a if a is not None else arrays[i].copy() for i, a in enumerate(out)]


def _lerp_knot(t_a, t_b, pa, pb, t):
    """Linear blend on a Catmull-Rom knot interval; stable when knots coincide."""
    den = t_b - t_a
    if abs(den) < 1e-15:
        return np.asarray(pa, dtype=np.float64)
    u = (t - t_a) / den
    return (1.0 - u) * pa + u * pb


def _catmull_rom_point(p0, p1, p2, p3, t0, t1, t2, t3, t):
    """One centripetal Catmull-Rom sample (Barry-Goldman)."""
    a1 = _lerp_knot(t0, t1, p0, p1, t)
    a2 = _lerp_knot(t1, t2, p1, p2, t)
    a3 = _lerp_knot(t2, t3, p2, p3, t)
    b1 = _lerp_knot(t0, t2, a1, a2, t)
    b2 = _lerp_knot(t1, t3, a2, a3, t)
    return _lerp_knot(t1, t2, b1, b2, t)


def catmull_rom(xy, samples=6, alpha=0.5):
    """
    Centripetal Catmull-Rom resampling of an open polyline with endpoints fixed.

    The curve passes through every control point. Tangents at the ends use a
    duplicated endpoint so smoothing never borrows coordinates from a neighbouring
    arc (shared borders must stay coincident). ``samples`` is the number of
    interpolated points inserted on each segment between consecutive controls.

    Parameters
    ----------
    xy : array-like of shape (n, 2)
        Arc coordinates in float space
    samples : int
        Interpolated points per segment (not counting the segment endpoints)
    alpha : float
        Knot parameterization; ``0.5`` is centripetal (avoids loops)

    Returns
    -------
    numpy.ndarray
        Smoothed (m, 2) coordinates; ``xy[0]`` and ``xy[-1]`` unchanged
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (n, 2) coordinates, got shape {pts.shape}")
    n = len(pts)
    if n < 3 or samples <= 0:
        return pts.copy()

    def knot(ti, pi, pj):
        d = float(np.linalg.norm(pj - pi))
        # Duplicated endpoints (open-arc tangents) make d==0; keep knots strictly
        # increasing so Barry-Goldman denominators stay defined.
        return ti + (d**alpha if d > 0 else 1e-8)

    out = [pts[0]]
    for i in range(n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        p0 = pts[i - 1] if i > 0 else p1
        p3 = pts[i + 2] if i + 2 < n else p2
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        if t2 - t1 < 1e-15:
            out.append(p2)
            continue
        # Uniform parameter steps between the knots of p1 and p2 (exclude p1, include
        # interiors, then append p2 so every control vertex is kept).
        for k in range(1, samples + 1):
            t = t1 + (t2 - t1) * (k / (samples + 1))
            out.append(_catmull_rom_point(p0, p1, p2, p3, t0, t1, t2, t3, t))
        out.append(p2)
    result = np.asarray(out, dtype=np.float64)
    result[0] = pts[0]
    result[-1] = pts[-1]
    return result


def smooth_arcs(arcs, method="chaikin", iterations=2, samples=6, segment_length=None):
    """
    Smooth each decoded float arc in place-style, returning new coordinate lists.

    Parameters
    ----------
    arcs : list of array-like
        Decoded arc coordinates (one (n, 2) array per arc)
    method : {"chaikin", "catmull_rom"}
        Smoothing algorithm
    iterations : int
        Chaikin passes (ignored for Catmull-Rom)
    samples : int
        Catmull-Rom points inserted per segment (ignored for Chaikin)
    segment_length : float, optional
        If set, densify each arc so no edge exceeds this length before smoothing
        (Smoothify-style controlled Chaikin radius). Endpoints stay fixed.

    Returns
    -------
    list of list
        Smoothed arcs as nested Python lists (TopoJSON float arcs)
    """
    if method not in ("chaikin", "catmull_rom"):
        raise ValueError(
            f"unknown smooth method {method!r}; choose 'chaikin' or 'catmull_rom'"
        )
    coords = [np.asarray(arc, dtype=np.float64) for arc in arcs]
    if segment_length is not None:
        coords = segmentize_many(coords, segment_length)
    if method == "chaikin":
        return [a.tolist() for a in chaikin_many(coords, iterations=iterations)]
    return [catmull_rom(xy, samples=samples).tolist() for xy in coords]


def restore_collapsed_rings(arcs, original, rings):
    """
    Put back vertices of the original arcs in rings that simplification reduced to
    fewer than three distinct points, so that each ring stays at least a triangle. Each
    time the original vertex farthest from the ring is put back in its arc; as arcs are
    shared, the rings next to it stay matched.

    Parameters
    ----------
    arcs : list of list
        Coordinates of the simplified arcs
    original : list of numpy.ndarray
        Coordinates of the arcs before simplification
    rings : list of list of int
        Arc references (~index for an arc used backward) of each ring

    Returns
    -------
    list of list
        The arcs, with vertices put back where needed
    """
    points = np.fromiter((len(a) - 1 for a in arcs), np.intp, len(arcs))
    for ring in rings:
        ids = [r if r >= 0 else ~r for r in ring]
        for _ in range(3 - points[ids].sum()):
            line = shapely.linestrings(np.concatenate([arcs[i] for i in ids]))
            xy = [original[i] for i in ids]
            far = shapely.distance(shapely.points(np.concatenate(xy)), line)
            if far.max() == 0:
                break
            arc = np.repeat(ids, [len(c) for c in xy])[far.argmax()]
            pos = np.concatenate([np.arange(len(c)) for c in xy])[far.argmax()]
            vertices = original[arc]
            keep = (vertices[:, None] == np.asarray(arcs[arc])[None]).all(-1).any(1)
            keep[pos] = True
            arcs[arc] = vertices[keep].tolist()
            points[arc] += 1
    return arcs


def winding_order(geom, order="CW_CCW"):
    """
    Function that force a certain winding order on the resulting output geometries. One
    can choose between `CCW_CW` and `CW_CCW`.

    `CW_CCW` implies clockwise for exterior polygons and counterclockwise for interior
    polygons (aka the geographical right-hand-rule where the right hand is in the area
    of interest as you walk the line).

    `CCW_CW` implies counterclockwise for exterior polygons and clockwise for interior
    polygons (aka the mathematical right-hand-rule where the right hand curls around
    the polygon's exterior with your thumb pointing "up" (toward space), signing a
    positive area for the polygon in the signed area sense).

    TopoJSON, and so this package, defaults to `CW_CCW`, but depending on the
    application you might decide differently.

    * https://bl.ocks.org/mbostock/a7bdfeb041e850799a8d3dce4d8c50c8

    Only applies to Polygons and MultiPolygons.

    Parameters
    ----------
    geom : geometry or shapely.geometry.GeometryCollection
        Geometry objects where the winding order will be forced upon.
    order : str, optional
        Choose `CW_CCW` for clockwise for exterior- and counterclockwise for
        interior polygons or `CCW_CW` for counterclockwise for exterior- and clockwise
        for interior polygons, by default `CW_CCW`.

    Returns
    -------
    geometry or shapely.geometry.GeometryCollection
        Geometry objects where the chosen winding order is forced upon.
    """

    # CW_CWW will orient the outer polygon clockwise and the inner polygon counter-
    # clockwise to conform TopoJSON standard

    if order not in ("CW_CCW", "CCW_CW"):
        raise NameError(f"parameter {order} was not recognized")
    return shapely.orient_polygons(geom, exterior_cw=order == "CW_CCW")


def round_coordinates(linestrings, rounding_precision):
    """
    Round all coordinates to a specified precision, e.g. `rounding_precision=3` will round
    to 3 decimals on the resulting output geometries (after the topology is computed).

    Parameters
    ----------
    linestrings : list of shapely.geometry.LineStrings
        LineStrings of which the coordinates will be rounded
    rounding_precision : int
        Precision value. Up till how many decimals the coordinates should be rounded.

    Returns
    -------
    list of shapely.geometry.LineStrings
        LineStrings of which the coordinates are rounded
    """
    for idx, geom in enumerate(linestrings):
        linestrings[idx] = wkt.loads(
            wkt.dumps(geom, rounding_precision=rounding_precision)
        )
    return linestrings


def prettify(topojson_object):
    """
    prettify TopoJSON Format output for readability.

    Parameters
    ----------
    topojson_object : topojson.Topojson
        object to be pretty printed

    Returns
    -------
    topojson.Topojson
        pretty printed JSON variant of the topology object
    """
    return pprint.pprint(topojson_object)


def properties_level(topojson_object, position="nested"):
    """
    Define where the attributes of the geometry object should be placed. Choose between
    `nested` or `foreign`. Default is `nested` where the attribute information is placed
    within the "properties" dictionary, part of the geometry.
    `foreign`, tries to place the attributes on the same level as the geometry.

    Parameters
    ----------
    topojson_object : topojson.Topojson
        [description]
    position : str, optional
        [description], by default "nested"
    """

    import warnings

    warnings.warn(("\nNot yet implemented."), DeprecationWarning, stacklevel=2)


def delta_encoding(linestrings):
    """
    Delta-encode linestrings: the first coordinate of each linestring is absolute,
    every next coordinate is relative to the previous one. All linestrings are
    encoded at once.

    Parameters
    ----------
    linestrings : list of shapely.geometry.LineStrings, arrays or lists
        Linestrings with integer coordinates

    Returns
    -------
    list of lists
        Delta-encoded linestrings
    """
    if not len(linestrings):
        return linestrings
    if hasattr(linestrings[0], "coords"):
        xy, idx = shapely.get_coordinates(linestrings, return_index=True)
        lengths = np.bincount(idx, minlength=len(linestrings))
    else:
        xy = np.concatenate(linestrings)
        lengths = np.fromiter(map(len, linestrings), dtype=np.intp)
    xy = xy.astype(np.int64)
    starts = np.cumsum(lengths) - lengths
    delta = np.diff(xy, axis=0, prepend=xy[:1])
    delta[starts] = xy[starts]
    delta = delta.tolist()
    return [delta[s : s + n] for s, n in zip(starts.tolist(), lengths.tolist())]


def delta_decoding(arcs):
    """
    Decode delta-encoded arcs to absolute coordinates. All arcs are decoded at once.

    Parameters
    ----------
    arcs : list of lists
        Delta-encoded arcs

    Returns
    -------
    list of numpy.ndarray
        (n, 2) integer coordinates of each arc
    """
    if not len(arcs):
        return []
    xy, lengths = _decoded(arcs)
    return _split(xy, np.cumsum(lengths)[:-1])


def _split(array, indices):
    """As `np.split(array, indices)`, but quicker for many small parts."""
    bounds = [0, *np.asarray(indices).tolist(), len(array)]
    return [array[a:b] for a, b in itertools.pairwise(bounds)]


def arc_areas(arcs):
    """
    Twice the signed area that each arc adds to a ring (shoelace formula), exact on
    the integer grid. The area of a ring is the sum over its arcs, with the sign
    flipped for an arc used backward.

    Parameters
    ----------
    arcs : list of numpy.ndarray
        Integer coordinates of the arcs

    Returns
    -------
    numpy.ndarray
        Twice the signed area of each arc, as integers
    """
    if not len(arcs):
        return np.zeros(0, np.int64)
    lengths = np.fromiter(map(len, arcs), np.intp, len(arcs))
    xy = np.concatenate(arcs).astype(np.int64)
    cross = np.r_[0, np.cumsum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])]
    ends = np.cumsum(lengths) - 1
    return cross[ends] - cross[ends - lengths + 1]


def _decoded(arcs):
    """Absolute coordinates of delta-encoded arcs, all arcs in one array, and the
    number of coordinates of each arc."""
    lengths = np.fromiter(map(len, arcs), np.intp, len(arcs))
    xy = np.concatenate(arcs).astype(np.int64).cumsum(axis=0)
    starts = np.cumsum(lengths) - lengths
    # restart the cumulative sum at the first coordinate of each arc
    offset = np.zeros((len(arcs), 2), dtype=np.int64)
    offset[1:] = xy[starts[1:] - 1]
    return xy - np.repeat(offset, lengths, axis=0), lengths


def cart(arr):
    """
    Function that returns all combinations as a 2D array
    [3, 152,  62, 52] is returned as [[152,  62], [152,  52], [152,   3]]
    """
    arr = -np.sort(-arr)
    arr = np.array(np.meshgrid(arr[0], arr[1:])).T.reshape(-1, 2)
    return arr


def hash_paths(paths):
    """
    Hash of each path that is the same for duplicate paths: equal coordinates in any
    direction and, for a closed path, from any start. The x and y values of a path are
    hashed as multisets (the closing point of a closed path left out), together with
    their number and whether the path is closed.

    Parameters
    ----------
    paths : list of numpy.ndarray
        Coordinates of each path

    Returns
    -------
    numpy.ndarray
        int64 hash of each path
    """
    if len(paths) == 0:
        return np.empty(0, np.int64)
    count = np.fromiter(map(len, paths), np.intp, len(paths))
    xy = np.concatenate(paths).astype(float)
    last = np.cumsum(count) - 1
    closed = (count > 1) & (xy[last - count + 1] == xy[last]).all(axis=1)
    keep = np.ones(len(xy), bool)
    keep[last[closed]] = False
    count -= closed
    start = np.cumsum(count) - count
    x, y = (_mix(np.ascontiguousarray(col).view(np.uint64)) for col in xy[keep].T)
    x, y = np.add.reduceat(x, start), np.add.reduceat(y, start)
    return _hash(x.view(np.int64), y.view(np.int64), count, closed)


def find_duplicates(segments_list, type="array"):
    """
    Function for solely detecting and recording duplicate LineStrings. The function
    converts and sorts the coordinates of each linestring and gets the hash. Using the
    hashes it can quickly detect duplicates and return the indices.

    Parameters
    ----------
    segments_list : list of paths
        list of valid paths
    type : str
        set if paths is `array` or `linestring`

    """

    if type != "array":
        segments_list = [
            np.array(list(linestring.coords)) for linestring in segments_list
        ]
    hash_segments = hash_paths(segments_list)

    # get split locations of dups
    idx_sort = np.argsort(hash_segments)
    sorted_hashes = hash_segments[idx_sort]
    _, idx_start, count = np.unique(
        sorted_hashes, return_counts=True, return_index=True
    )
    if count.max() > 1:
        # split on indices that occurs > 1
        idx_dups = np.split(idx_sort, idx_start[1:])
        list_dups = []
        for dup in idx_dups:
            if dup.size > 2:
                list_dups.append(cart(dup))
            elif dup.size > 1:
                list_dups.append(dup)
        idx_dups = np.vstack(list_dups)

        # apply sorting on duplicate-pairs
        # pylint: disable=invalid-unary-operand-type
        idx_dups = -np.sort(-idx_dups, axis=1)
        idx_dups = idx_dups[np.argsort(idx_dups[:, 0])]
        return idx_dups
    else:
        return []


def map_values(arr, search_vals, replace_vals):
    """
    This function replace values element-wise in a numpy array.
    Its quick and avoids a np.where-loop (which is slow).
    The result is a new array, not inplace.

    Parameters
    ----------
    arr : np.array
        input array
    search_vals : list or 1D np.array
        array with 'bad' values
    replace_vals : list or 1D np.array
        array with 'good' values

    Returns
    -------
    np.array
        new array with replaced values
    """
    N = max(arr.max(), max(search_vals)) + 1
    maparr = np.empty(N, dtype=np.intp)

    maparr[arr] = arr
    maparr[search_vals] = replace_vals

    arr_upd = maparr[arr]
    return arr_upd


def remove_collinear_points(line: np.ndarray) -> np.ndarray:
    # If only 2 points, no use checking
    if len(line) <= 2:
        return line
    # Prepare "points"
    p1_x = line[:-2, 0]
    p1_y = line[:-2, 1]
    p2_x = line[1:-1, 0]
    p2_y = line[1:-1, 1]
    p3_x = line[2:, 0]
    p3_y = line[2:, 1]

    # Calculate
    collinear_mask = (p2_x - p1_x) * (p3_y - p1_y) == (p3_x - p1_x) * (p2_y - p1_y)
    collinear_mask = np.concatenate([[False], collinear_mask, [False]])
    return line[~np.array(collinear_mask)]
