//! 64-bit row hashes matching `topojson.ops._hash` / `_mix` (splitmix64).

#[inline]
pub fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline]
pub fn hash_cols(cols: &[&[i64]]) -> Vec<i64> {
    let n = cols[0].len();
    let mut h = vec![0x9E3779B97F4A7C15u64; n];
    for col in cols {
        debug_assert_eq!(col.len(), n);
        for i in 0..n {
            h[i] = mix(h[i].wrapping_add(col[i] as u64));
        }
    }
    h.into_iter().map(|v| v as i64).collect()
}

#[inline]
pub fn hash2(a: &[i64], b: &[i64]) -> Vec<i64> {
    hash_cols(&[a, b])
}

#[inline]
pub fn hash3(a: &[i64], b: &[i64], c: &[i64]) -> Vec<i64> {
    hash_cols(&[a, b, c])
}

#[inline]
pub fn hash4(a: &[i64], b: &[i64], c: &[i64], d: &[i64]) -> Vec<i64> {
    hash_cols(&[a, b, c, d])
}

/// True where a hash may occur more than once (bucket counts, no sort).
pub fn repeated(h: &[i64]) -> Vec<bool> {
    let n = h.len();
    let size = (8 * n).next_power_of_two().max(16);
    let mask = (size - 1) as i64;
    let mut counts = vec![0u32; size];
    for &v in h {
        counts[(v & mask) as usize] += 1;
    }
    h.iter()
        .map(|&v| counts[(v & mask) as usize] > 1)
        .collect()
}

/// True at the first row and where any column differs from the row before.
pub fn changes(cols: &[&[i64]]) -> Vec<bool> {
    let n = cols[0].len();
    let mut out = vec![false; n];
    if n == 0 {
        return out;
    }
    out[0] = true;
    for i in 1..n {
        let mut diff = false;
        for col in cols {
            if col[i] != col[i - 1] {
                diff = true;
                break;
            }
        }
        out[i] = diff;
    }
    out
}

/// Concatenated ranges `[starts[i], starts[i] + counts[i])`.
pub fn ranges(starts: &[usize], counts: &[usize]) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(total);
    let mut cum = 0usize;
    for i in 0..starts.len() {
        let c = counts[i];
        let offset = starts[i].wrapping_sub(cum).wrapping_add(c);
        // matches: offset = starts - cumsum(counts) + counts, then + arange(counts.sum())
        // Per-group: for j in 0..c: out.push(starts[i] - (cum) + c + (cum + j) - c) wait
        // Python: offset = np.repeat(starts - np.cumsum(counts) + counts, counts)
        //         return offset + np.arange(counts.sum())
        let base = starts[i] as isize - (cum as isize + c as isize) + c as isize;
        for j in 0..c {
            out.push((base + (cum + j) as isize) as usize);
        }
        cum += c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_matches_python_shape() {
        let starts = vec![1usize, 5, 10];
        let counts = vec![2usize, 0, 3];
        // cumsum counts = [2,2,5]; starts - cumsum + counts = [1, 3, 8]
        // repeat -> [1,1, 8,8,8]; + arange(5) -> [1,2, 10,11,12]
        assert_eq!(ranges(&starts, &counts), vec![1, 2, 10, 11, 12]);
    }
}
