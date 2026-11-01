//! Chaikin corner-cutting matching `topojson.ops.chaikin` / `chaikin_many`.
//! Each iteration grows an arc of length `n` to `2n`, then re-pins endpoints.

/// One Chaikin pass: length `n` → `2n`, endpoints copied from the input.
fn chaikin_once(xy: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = xy.len();
    if n < 3 {
        return xy.to_vec();
    }
    let mut out = vec![[0.0, 0.0]; 2 * n];
    out[0] = xy[0];
    out[2 * n - 1] = xy[n - 1];
    for i in 0..n - 1 {
        let a = xy[i];
        let b = xy[i + 1];
        let q = [0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]];
        let r = [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]];
        out[1 + 2 * i] = q;
        out[1 + 2 * i + 1] = r;
    }
    out
}

/// Chaikin with `iterations` passes; endpoints pinned to the original ends.
pub fn chaikin(xy: &[[f64; 2]], iterations: usize) -> Vec<[f64; 2]> {
    if xy.len() < 3 || iterations == 0 {
        return xy.to_vec();
    }
    let ends = (xy[0], xy[xy.len() - 1]);
    let mut cur = xy.to_vec();
    for _ in 0..iterations {
        cur = chaikin_once(&cur);
        let last = cur.len() - 1;
        cur[0] = ends.0;
        cur[last] = ends.1;
    }
    cur
}

/// Packed multi-arc Chaikin. `offsets` length = n_arcs + 1.
pub fn chaikin_many(
    xy: &[[f64; 2]],
    offsets: &[usize],
    iterations: usize,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut out = Vec::new();
    let mut new_off = vec![0usize];
    for i in 0..offsets.len().saturating_sub(1) {
        let smoothed = chaikin(&xy[offsets[i]..offsets[i + 1]], iterations);
        out.extend_from_slice(&smoothed);
        new_off.push(out.len());
    }
    (out, new_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_to_2n_per_iteration() {
        let xy = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
        assert_eq!(chaikin(&xy, 1).len(), 6);
        assert_eq!(chaikin(&xy, 2).len(), 12);
        assert_eq!(chaikin(&xy, 1)[0], [0.0, 0.0]);
        assert_eq!(chaikin(&xy, 1)[5], [2.0, 0.0]);
    }
}
