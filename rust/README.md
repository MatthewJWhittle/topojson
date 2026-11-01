# topojson `_core` (Rust / PyO3)

Optional coordinate kernels for the Python package:

- Quantized `cut_lines_on_grid` on packed `int64` buffers
- Chaikin corner-cutting (FFI / packaging spike; oracle vs NumPy)
- Quantized `shared_path_ends_on_grid` (opt-in via `TOPOJSON_RUST_JOIN=1`; NumPy remains default)

**License:** BSD-3-Clause. Copyright (c) 2026 Matthew J Whittle.  
**Not linked:** GEOS. **Not vendored:** CartAGen, Smoothify.

## Build (editable checkout)

From the repo root (so `topojson._core` lands next to the Python package):

```bash
pip install maturin
# Build the cdylib and copy into the source tree for editable installs:
cargo build --release --manifest-path rust/Cargo.toml
cp rust/target/release/lib_core.so topojson/_core.cpython-$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')-x86_64-linux-gnu.so
# Or use maturin against rust/ with a venv; ensure the .so is importable as topojson._core.
```

Verify:

```bash
python -c "from topojson import ops; print(ops.rust_core_available())"
```

Pure Python/NumPy continues to work when the extension is absent.
