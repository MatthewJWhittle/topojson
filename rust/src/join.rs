//! Quantized `shared_path_ends_on_grid` on packed int64 coordinates.
//! Ports `topojson.ops` grid join helpers. Pairwise GEOS fallback pairs are
//! returned to Python (`shared_path_ends`).

use crate::hash::{changes, hash2, hash3, hash4, hash_cols, mix, ranges, repeated};
use std::collections::BTreeSet;

#[inline]
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn grid_segments(xy: &[[i64; 2]], li: &[i32]) -> (Vec<usize>, Vec<i64>, Vec<i64>) {
    let xs: Vec<i64> = xy.iter().map(|p| p[0]).collect();
    let ys: Vec<i64> = xy.iter().map(|p| p[1]).collect();
    let vertex = hash2(&xs, &ys);
    let mut s = Vec::new();
    for i in 0..xy.len().saturating_sub(1) {
        if li[i + 1] == li[i] {
            s.push(i);
        }
    }
    let mut a = Vec::with_capacity(s.len());
    let mut b = Vec::with_capacity(s.len());
    for &i in &s {
        a.push(vertex[i]);
        b.push(vertex[i + 1]);
    }
    let mut h = Vec::with_capacity(s.len());
    for i in 0..s.len() {
        // _hash(a ^ b, a + b)
        let mut hv = 0x9E3779B97F4A7C15u64;
        hv = mix(hv.wrapping_add((a[i] ^ b[i]) as u64));
        hv = mix(hv.wrapping_add(a[i].wrapping_add(b[i]) as u64));
        h.push(hv as i64);
    }
    (s, h, vertex)
}

fn t_vertices(
    xy: &[[i64; 2]],
    li: &[i32],
    s: &[usize],
    h: &[i64],
) -> (Vec<usize>, Vec<[i64; 2]>) {
    if s.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_seg = s.len();
    let mut dx = vec![0i64; n_seg];
    let mut dy = vec![0i64; n_seg];
    let mut ax = vec![0i64; n_seg];
    let mut ay = vec![0i64; n_seg];
    let mut bx = vec![0i64; n_seg];
    let mut by = vec![0i64; n_seg];
    for i in 0..n_seg {
        let p0 = xy[s[i]];
        let p1 = xy[s[i] + 1];
        ax[i] = p0[0];
        ay[i] = p0[1];
        bx[i] = p1[0];
        by[i] = p1[1];
        let mut d0 = p1[0] - p0[0];
        let mut d1 = p1[1] - p0[1];
        let g = gcd_i64(d0, d1).max(1);
        d0 /= g;
        d1 /= g;
        if d0 < 0 || (d0 == 0 && d1 < 0) {
            d0 = -d0;
            d1 = -d1;
        }
        dx[i] = d0;
        dy[i] = d1;
    }
    let mut c = vec![0i64; n_seg];
    let mut lo = vec![0i64; n_seg];
    let mut hi = vec![0i64; n_seg];
    for i in 0..n_seg {
        c[i] = dy[i] * ax[i] - dx[i] * ay[i];
        let pa = dx[i] * ax[i] + dy[i] * ay[i];
        let pb = dx[i] * bx[i] + dy[i] * by[i];
        lo[i] = pa.min(pb);
        hi[i] = pa.max(pb);
    }
    let carrier = hash3(&dx, &dy, &c);

    // stretches: changes of (li[s], carrier)
    let sl: Vec<i64> = s.iter().map(|&i| li[i] as i64).collect();
    let new_stretch = changes(&[&sl, &carrier]);
    let mut r = Vec::new();
    for i in 0..n_seg {
        if new_stretch[i] {
            r.push(i);
        }
    }
    let mut key = Vec::with_capacity(r.len());
    let mut start = Vec::with_capacity(r.len());
    let mut end = Vec::with_capacity(r.len());
    // reduceat lo/hi and xor-reduceat h over stretch groups
    let mut stretch_xor = Vec::with_capacity(r.len());
    for (gi, &ri) in r.iter().enumerate() {
        key.push(carrier[ri]);
        let end_i = if gi + 1 < r.len() {
            r[gi + 1]
        } else {
            n_seg
        };
        let mut lo_m = lo[ri];
        let mut hi_m = hi[ri];
        let mut hx = 0i64;
        for j in ri..end_i {
            lo_m = lo_m.min(lo[j]);
            hi_m = hi_m.max(hi[j]);
            hx ^= h[j];
        }
        start.push(lo_m);
        end.push(hi_m);
        stretch_xor.push(hx);
    }

    let rep = repeated(&key);
    let mut u: Vec<usize> = rep
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    // unique by stretch hash
    let stretch_h = hash4(
        &u.iter().map(|&i| key[i]).collect::<Vec<_>>(),
        &u.iter().map(|&i| start[i]).collect::<Vec<_>>(),
        &u.iter().map(|&i| end[i]).collect::<Vec<_>>(),
        &u.iter().map(|&i| stretch_xor[i]).collect::<Vec<_>>(),
    );
    let mut seen = std::collections::HashMap::new();
    let mut u_unique = Vec::new();
    for (idx, &uh) in stretch_h.iter().enumerate() {
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(uh) {
            e.insert(());
            u_unique.push(u[idx]);
        }
    }
    u = u_unique;
    u.sort_by(|&i, &j| key[i].cmp(&key[j]).then(start[i].cmp(&start[j])));

    let mut overlap_keys = Vec::new();
    for w in u.windows(2) {
        if key[w[0]] == key[w[1]] && start[w[1]] < end[w[0]] {
            overlap_keys.push(key[w[1]]);
        }
    }
    let overlap_set: std::collections::HashSet<i64> = overlap_keys.into_iter().collect();
    let k: Vec<usize> = (0..n_seg)
        .filter(|&i| overlap_set.contains(&carrier[i]))
        .collect();
    if k.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // rank segment ends along carrier
    let m = k.len();
    let mut d0 = Vec::with_capacity(2 * m);
    let mut d1 = Vec::with_capacity(2 * m);
    let mut cc = Vec::with_capacity(2 * m);
    let mut rank_val = Vec::with_capacity(2 * m);
    for &i in &k {
        d0.push(dx[i]);
        d1.push(dy[i]);
        cc.push(c[i]);
        rank_val.push(lo[i]);
    }
    for &i in &k {
        d0.push(dx[i]);
        d1.push(dy[i]);
        cc.push(c[i]);
        rank_val.push(hi[i]);
    }
    let mut o: Vec<usize> = (0..2 * m).collect();
    o.sort_by(|&i, &j| {
        d0[i]
            .cmp(&d0[j])
            .then(d1[i].cmp(&d1[j]))
            .then(cc[i].cmp(&cc[j]))
            .then(rank_val[i].cmp(&rank_val[j]))
    });
    let cols_o: Vec<Vec<i64>> = {
        let mut out = vec![vec![0i64; 2 * m]; 4];
        for (pos, &oi) in o.iter().enumerate() {
            out[0][pos] = d0[oi];
            out[1][pos] = d1[oi];
            out[2][pos] = cc[oi];
            out[3][pos] = rank_val[oi];
        }
        // changes needs original column order at o indices — use sorted order cols
        out
    };
    // Actually Python: o = lexsort(cols[::-1]); new = changes(*(col[o] for col in cols))
    let mut new = vec![false; 2 * m];
    new[0] = true;
    for i in 1..2 * m {
        let a = o[i];
        let b = o[i - 1];
        new[i] = d0[a] != d0[b] || d1[a] != d1[b] || cc[a] != cc[b] || rank_val[a] != rank_val[b];
    }
    let mut rank = vec![0isize; 2 * m];
    let mut cum = 0isize;
    for (pos, &oi) in o.iter().enumerate() {
        if new[pos] {
            // first of group — cum already counts previous groups
        }
        if pos == 0 || new[pos] {
            if pos > 0 {
                cum += 1;
            }
        }
        rank[oi] = cum;
    }
    // Fix cumsum of new - 1
    let mut cum2 = 0isize;
    let mut rank2 = vec![0isize; 2 * m];
    for (pos, &oi) in o.iter().enumerate() {
        if new[pos] {
            if pos > 0 {
                cum2 += 1;
            }
            // keep
        } else if pos == 0 {
            cum2 = 0;
        }
        rank2[oi] = cum2;
    }
    // recount properly
    {
        let mut csum = -1isize;
        for (pos, &oi) in o.iter().enumerate() {
            if new[pos] {
                csum += 1;
            }
            rank[oi] = csum;
        }
    }

    let mut count = vec![0usize; m];
    for i in 0..m {
        let c = (rank[m + i] - rank[i] - 1).max(0) as usize;
        count[i] = c;
    }
    let mut seg = Vec::new();
    let mut starts_for_ranges = Vec::new();
    for i in 0..m {
        starts_for_ranges.push((rank[i] + 1) as usize);
        for _ in 0..count[i] {
            seg.push(k[i]);
        }
    }
    // t = cols[3][o][new][_ranges(rank[:m]+1, count)]
    let mut uniq_rank_vals = Vec::new();
    for (pos, &oi) in o.iter().enumerate() {
        if new[pos] {
            uniq_rank_vals.push(rank_val[oi]);
        }
    }
    let range_idx = ranges(&starts_for_ranges, &count);
    let mut t_vals = Vec::with_capacity(range_idx.len());
    for &ri in &range_idx {
        t_vals.push(uniq_rank_vals[ri]);
    }

    let mut at = Vec::with_capacity(seg.len());
    let mut pts = Vec::with_capacity(seg.len());
    for i in 0..seg.len() {
        let si = seg[i];
        let ds0 = dx[si];
        let ds1 = dy[si];
        let origin = [ax[si], ay[si]];
        let denom = ds0 * ds0 + ds1 * ds1;
        let step = if denom == 0 {
            0
        } else {
            (t_vals[i] - (ds0 * origin[0] + ds1 * origin[1])) / denom
        };
        at.push(s[si]);
        pts.push([origin[0] + ds0 * step, origin[1] + ds1 * step]);
    }
    let _ = (cols_o, rank2, hash_cols);
    (at, pts)
}

fn pairs_lines(
    line: &[i32],
    group: &[i32],
    candidate: &[bool],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    // first index of each group
    let mut first = Vec::new();
    let mut prev = -1i32;
    for (i, &g) in group.iter().enumerate() {
        if i == 0 || g != prev {
            first.push(i);
            prev = g;
        }
    }
    let mut size = Vec::with_capacity(first.len());
    for i in 0..first.len() {
        let end = if i + 1 < first.len() {
            first[i + 1]
        } else {
            group.len()
        };
        size.push(end - first[i]);
    }
    let mut rows = Vec::new();
    for (i, &ln) in line.iter().enumerate() {
        if candidate[ln as usize] {
            rows.push(i);
        }
    }
    let mut i_idx = Vec::new();
    let mut j_idx = Vec::new();
    for &row in &rows {
        let g = group[row] as usize;
        let count = size[g] - 1;
        let base = first[g];
        let mut skipped = false;
        for k in 0..size[g] {
            let j = base + k;
            if j == row {
                skipped = true;
                continue;
            }
            if i_idx.len() - (rows.iter().position(|&r| r == row).unwrap() * 0) >= 0 {
                // always push
            }
            let _ = (count, skipped);
            i_idx.push(row);
            j_idx.push(j);
        }
    }
    // unique pair counts
    let n = candidate.len() as i64;
    let mut map = std::collections::HashMap::<i64, i32>::new();
    for (&i, &j) in i_idx.iter().zip(j_idx.iter()) {
        let key = line[i] as i64 * n + line[j] as i64;
        *map.entry(key).or_insert(0) += 1;
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut times = Vec::new();
    for (key, t) in map {
        a.push((key / n) as i32);
        b.push((key % n) as i32);
        times.push(t);
    }
    (a, b, times)
}

fn junction_vertices(
    xy: &[[i64; 2]],
    li: &[i32],
    s: &[usize],
    h: &[i64],
    vertex: &[i64],
    closed: &[bool],
) -> (Vec<[i64; 2]>, Vec<(i32, i32)>) {
    let n = closed.len();
    if n == 0 || s.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let key: Vec<i64> = (0..n).map(|i| {
        let mut hv = 0x9E3779B97F4A7C15u64;
        hv = mix(hv.wrapping_add(i as u64));
        hv as i64
    }).collect();

    let mut count = vec![0usize; n];
    for &l in li {
        count[l as usize] += 1;
    }
    let mut last = vec![0usize; n];
    {
        let mut acc = 0usize;
        for i in 0..n {
            acc += count[i];
            last[i] = acc - 1;
        }
    }
    let ring: Vec<usize> = (0..n).filter(|&i| closed[i] && count[i] > 1).collect();

    let sl: Vec<i32> = s.iter().map(|&i| li[i]).collect();
    let rep = repeated(h);
    let mut o: Vec<usize> = rep
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();
    o.sort_by(|&i, &j| h[i].cmp(&h[j]).then(i.cmp(&j))); // stable by index

    // group by h
    let mut group = vec![0i32; o.len()];
    if !o.is_empty() {
        let mut g = 0i32;
        group[0] = 0;
        for i in 1..o.len() {
            if h[o[i]] != h[o[i - 1]] {
                g += 1;
            }
            group[i] = g;
        }
    }
    let mut gcount = vec![0i32; group.iter().copied().max().unwrap_or(0) as usize + 1];
    for &g in &group {
        gcount[g as usize] += 1;
    }
    let mut shared = vec![false; s.len()];
    for (idx, &oi) in o.iter().enumerate() {
        shared[oi] = gcount[group[idx] as usize] > 1;
    }
    o = o.into_iter().filter(|&oi| shared[oi]).collect();
    // re-filter changes(h[o], sl[o])
    {
        o.sort_by(|&i, &j| h[i].cmp(&h[j]).then(sl[i].cmp(&sl[j])).then(i.cmp(&j)));
        let mut filtered = Vec::new();
        for (i, &oi) in o.iter().enumerate() {
            if i == 0
                || h[oi] != h[o[i - 1]]
                || sl[oi] != sl[o[i - 1]]
            {
                filtered.push(oi);
            }
        }
        o = filtered;
    }

    // equal lines count once
    let mut only = vec![false; n];
    {
        let mut binc = vec![0usize; n];
        for &oi in &o {
            binc[sl[oi] as usize] += 1;
        }
        // only[line] true if line has shared segs; then false if has unshared
        for i in 0..n {
            only[i] = binc[i] > 0;
        }
        for i in 0..s.len() {
            if !shared[i] {
                only[sl[i] as usize] = false;
            }
        }
    }
    let mut m: Vec<usize> = o
        .iter()
        .copied()
        .filter(|&oi| only[sl[oi] as usize])
        .collect();
    m.sort_by_key(|&oi| sl[oi]);
    let mut lines_hash = key.clone();
    {
        let mut i = 0;
        while i < m.len() {
            let line = sl[m[i]] as usize;
            let mut j = i;
            let mut xor = 0i64;
            while j < m.len() && sl[m[j]] as usize == line {
                xor ^= h[m[j]];
                j += 1;
            }
            lines_hash[line] = xor;
            i = j;
        }
    }
    let mut keep = vec![false; n];
    {
        let mut seen = std::collections::HashMap::new();
        for i in 0..n {
            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(lines_hash[i]) {
                e.insert(());
                keep[i] = true;
            }
        }
    }
    o.retain(|&oi| keep[sl[oi] as usize]);

    // odd lines → pairwise
    let mut visit = vec![true; xy.len()];
    for &r in &ring {
        visit[last[r]] = false;
    }
    let mut v: Vec<usize> = (0..xy.len()).filter(|&i| visit[i]).collect();
    let mut at: Vec<i64> = v.iter().map(|&i| vertex[i] ^ key[li[i] as usize]).collect();
    let rep_at = repeated(&at);
    let mut v2 = Vec::new();
    let mut at2 = Vec::new();
    for i in 0..v.len() {
        if rep_at[i] {
            v2.push(v[i]);
            at2.push(at[i]);
        }
    }
    v = v2;
    at = at2;
    // unique counts
    let mut odd = vec![false; n];
    {
        let mut counts = std::collections::HashMap::<i64, usize>::new();
        for &a in &at {
            *counts.entry(a).or_insert(0) += 1;
        }
        for (i, &a) in at.iter().enumerate() {
            if counts[&a] > 1 {
                odd[li[v[i]] as usize] = true;
            }
        }
    }

    // pairs for candidates
    let mut group2 = vec![0i32; o.len()];
    if !o.is_empty() {
        o.sort_by(|&i, &j| h[i].cmp(&h[j]).then(i.cmp(&j)));
        let mut g = 0i32;
        for i in 0..o.len() {
            if i > 0 && h[o[i]] != h[o[i - 1]] {
                g += 1;
            }
            group2[i] = g;
        }
    }
    let sl_o: Vec<i32> = o.iter().map(|&i| sl[i]).collect();
    let mut cand = vec![false; n];
    for i in 0..n {
        cand[i] = keep[i] && (odd[i] || (only[i] && closed[i]));
    }
    let (mut a, mut b, times) = pairs_lines(&sl_o, &group2, &cand);
    // remap a,b through o? Python: _pairs(sl[o], group, ...) returns line indices directly
    let mut binc_sl_o = vec![0usize; n];
    for &oi in &o {
        binc_sl_o[sl[oi] as usize] += 1;
    }
    let mut loop_mask = vec![false; a.len()];
    for i in 0..a.len() {
        let ai = a[i] as usize;
        loop_mask[i] = only[ai] && closed[ai] && (times[i] as usize == binc_sl_o[ai]);
        if loop_mask[i] {
            odd[ai] = true;
        }
    }
    for i in 0..n {
        odd[i] &= keep[i];
    }
    let mut pairs = BTreeSet::new();
    for i in 0..a.len() {
        if odd[a[i] as usize] {
            let x = a[i].min(b[i]);
            let y = a[i].max(b[i]);
            pairs.insert((x, y));
        }
    }
    // filter a,b to odd[a] — pairs already built

    // other lines on each segment
    let regular: Vec<bool> = sl.iter().map(|&l| keep[l as usize] && !odd[l as usize]).collect();
    o.retain(|&oi| regular[oi]);
    let mut group3 = vec![0i32; o.len()];
    if !o.is_empty() {
        o.sort_by(|&i, &j| h[i].cmp(&h[j]).then(i.cmp(&j)));
        let mut g = 0i32;
        for i in 0..o.len() {
            if i > 0 && h[o[i]] != h[o[i - 1]] {
                g += 1;
            }
            group3[i] = g;
        }
    }
    let mut other = vec![0i64; s.len()];
    if !o.is_empty() {
        // xor of key[sl] per group, then ^ key[sl[o]]
        let mut first = Vec::new();
        let mut gxor = Vec::new();
        let mut gi = 0usize;
        while gi < o.len() {
            first.push(gi);
            let g = group3[gi];
            let mut xor = 0i64;
            let mut gj = gi;
            while gj < o.len() && group3[gj] == g {
                xor ^= key[sl[o[gj]] as usize];
                gj += 1;
            }
            gxor.push(xor);
            gi = gj;
        }
        for (idx, &oi) in o.iter().enumerate() {
            let g = group3[idx] as usize;
            other[oi] = gxor[g] ^ key[sl[oi] as usize];
        }
    }
    let mut hin = vec![0i64; xy.len()];
    let mut hout = vec![0i64; xy.len()];
    for i in 0..s.len() {
        if regular[i] {
            hout[s[i]] = other[i];
            hin[s[i] + 1] = other[i];
        }
    }
    for &r in &ring {
        let begin = last[r] + 1 - count[r];
        hin[begin] = hin[last[r]];
        hout[last[r]] = hout[begin];
    }
    let mut points = Vec::new();
    for i in 0..xy.len() {
        if hin[i] != hout[i] {
            points.push(xy[i]);
        }
    }
    let pair_list: Vec<(i32, i32)> = pairs.into_iter().collect();
    let _ = (a, b);
    (points, pair_list)
}

/// Insert T-vertices into packed coordinates (sorted by insertion index).
fn insert_points(
    xy: &[[i64; 2]],
    li: &[i32],
    at: &[usize],
    pts: &[[i64; 2]],
) -> (Vec<[i64; 2]>, Vec<i32>) {
    if at.is_empty() {
        return (xy.to_vec(), li.to_vec());
    }
    // sort by |pts-xy[at]| then at (Python lexsort)
    let mut order: Vec<usize> = (0..at.len()).collect();
    order.sort_by(|&i, &j| {
        let di = (pts[i][0] - xy[at[i]][0]).abs() + (pts[i][1] - xy[at[i]][1]).abs();
        let dj = (pts[j][0] - xy[at[j]][0]).abs() + (pts[j][1] - xy[at[j]][1]).abs();
        di.cmp(&dj).then(at[i].cmp(&at[j]))
    });
    // insert from the back so indices stay valid
    let mut pairs: Vec<(usize, [i64; 2], i32)> = order
        .iter()
        .map(|&i| (at[i] + 1, pts[i], li[at[i]]))
        .collect();
    pairs.sort_by_key(|(idx, _, _)| *idx);
    let mut new_xy = Vec::with_capacity(xy.len() + pairs.len());
    let mut new_li = Vec::with_capacity(li.len() + pairs.len());
    let mut pi = 0;
    for i in 0..xy.len() {
        while pi < pairs.len() && pairs[pi].0 == i {
            new_xy.push(pairs[pi].1);
            new_li.push(pairs[pi].2);
            pi += 1;
        }
        new_xy.push(xy[i]);
        new_li.push(li[i]);
    }
    while pi < pairs.len() {
        new_xy.push(pairs[pi].1);
        new_li.push(pairs[pi].2);
        pi += 1;
    }
    (new_xy, new_li)
}

/// Returns (junction points as (J,2) i64, fallback pairs as (P,2) i32).
pub fn shared_path_ends_on_grid(
    xy: &[[i64; 2]],
    li: &[i32],
    closed: &[bool],
) -> (Vec<[i64; 2]>, Vec<[i32; 2]>) {
    if xy.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let (s, h, vertex) = grid_segments(xy, li);
    let (at, pts) = t_vertices(xy, li, &s, &h);
    let (xy, li) = insert_points(xy, li, &at, &pts);
    let (s, h, vertex) = grid_segments(&xy, &li);
    let (points, pairs) = junction_vertices(&xy, &li, &s, &h, &vertex, closed);
    let pair_arr = pairs.into_iter().map(|(a, b)| [a, b]).collect();
    (points, pair_arr)
}
