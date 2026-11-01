//! Quantized cut on packed integer coordinates (no Shapely / STRtree).
//! Mirrors `topojson.ops.cut_lines_on_grid` after any non-vertex inserts.

use crate::hash::hash2;
use std::collections::HashSet;

/// Cut quantized lines at junction vertices.
///
/// Returns `(parts_xy flat, part_offsets (n_parts+1), n_parts_per_line)`.
pub fn cut_lines_on_grid(
    xy_in: &[[i64; 2]],
    li: &[i32],
    jxy: &[[i64; 2]],
    is_ring: &[bool],
) -> (Vec<[i64; 2]>, Vec<usize>, Vec<i32>) {
    let n_lines = is_ring.len();
    if xy_in.is_empty() || n_lines == 0 {
        return (Vec::new(), vec![0], vec![0i32; n_lines]);
    }

    let xs: Vec<i64> = xy_in.iter().map(|p| p[0]).collect();
    let ys: Vec<i64> = xy_in.iter().map(|p| p[1]).collect();
    let jxs: Vec<i64> = jxy.iter().map(|p| p[0]).collect();
    let jys: Vec<i64> = jxy.iter().map(|p| p[1]).collect();
    let j_set: HashSet<i64> = hash2(&jxs, &jys).into_iter().collect();
    let mut split: Vec<bool> = hash2(&xs, &ys)
        .into_iter()
        .map(|h| j_set.contains(&h))
        .collect();

    let mut count = vec![0usize; n_lines];
    for &line in li {
        count[line as usize] += 1;
    }
    let mut pos = vec![0usize; xy_in.len()];
    {
        let mut seen = vec![0usize; n_lines];
        for (i, &line) in li.iter().enumerate() {
            let l = line as usize;
            pos[i] = seen[l];
            seen[l] += 1;
        }
    }

    let mut first_j = vec![0usize; n_lines];
    for i in (0..xy_in.len()).rev() {
        if split[i] {
            first_j[li[i] as usize] = pos[i];
        }
    }

    let mut order: Vec<usize> = (0..xy_in.len()).collect();
    for i in 0..xy_in.len() {
        let l = li[i] as usize;
        let shift = if is_ring[l] { first_j[l] } else { 0 };
        if shift == 0 {
            continue;
        }
        let n = count[l].saturating_sub(1).max(1);
        let p = pos[i];
        let rotated = if p == count[l] - 1 {
            shift
        } else {
            (p + shift) % n
        };
        order[i] = (i as isize + rotated as isize - p as isize) as usize;
    }

    let xy: Vec<[i64; 2]> = order.iter().map(|&i| xy_in[i]).collect();
    split = order.iter().map(|&i| split[i]).collect();
    let n_per_vertex: Vec<usize> = li
        .iter()
        .map(|&line| count[line as usize].saturating_sub(1))
        .collect();

    let mut take: Vec<usize> = Vec::with_capacity(xy.len() * 2);
    for i in 0..xy.len() {
        take.push(i);
        let p = pos[i];
        let n = n_per_vertex[i];
        if split[i] && p > 0 && p < n {
            take.push(i);
        }
    }

    let mut start = vec![false; take.len()];
    if !take.is_empty() {
        start[0] = true;
        for t in 1..take.len() {
            let gap = take[t] != take[t - 1] + 1;
            let new_line = li[take[t]] != li[take[t - 1]];
            start[t] = gap || new_line;
        }
    }

    let taken_xy: Vec<[i64; 2]> = take.iter().map(|&i| xy[i]).collect();
    let taken_li: Vec<i32> = take.iter().map(|&i| li[i]).collect();

    let mut keep = vec![true; taken_xy.len()];
    if taken_xy.len() >= 3 {
        for t in 1..taken_xy.len() - 1 {
            if start[t] || start[t + 1] {
                continue;
            }
            let a = taken_xy[t - 1];
            let b = taken_xy[t];
            let c = taken_xy[t + 1];
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if cross == 0 {
                keep[t] = false;
            }
        }
    }

    // Mirror: parts = split(xy[keep], flatnonzero(start[keep])[1:])
    // n_parts = bincount(li[take][start])
    let mut kept_xy = Vec::new();
    let mut kept_start_idx = Vec::new(); // indices into kept_xy that are starts
    let mut n_parts = vec![0i32; n_lines];
    for t in 0..taken_xy.len() {
        if !keep[t] {
            continue;
        }
        if start[t] {
            kept_start_idx.push(kept_xy.len());
            n_parts[taken_li[t] as usize] += 1;
        }
        kept_xy.push(taken_xy[t]);
    }

    let mut part_offsets = vec![0usize];
    // split at kept_start_idx[1:]
    for &idx in kept_start_idx.iter().skip(1) {
        part_offsets.push(idx);
    }
    part_offsets.push(kept_xy.len());

    (kept_xy, part_offsets, n_parts)
}
