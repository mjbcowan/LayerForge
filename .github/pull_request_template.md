## What this changes

<!-- One or two sentences. If it fixes a reported error, paste the key line of
     the traceback so the regression test can be traced back to it. -->

## Testing checklist

See [CONTRIBUTING.md](../CONTRIBUTING.md) for why each of these exists.

- [ ] `pytest` passes locally (on macOS: **without** `QT_QPA_PLATFORM=offscreen`)
- [ ] New/changed reader behaviour is asserted on **pixel values**, not just
      array shapes or dimension sizes
- [ ] Anything touching the reader, manifest, or layer metadata has a test that
      goes through real plugin dispatch — `viewer.open(..., plugin="LayerForge")`
      in `tests/test_plugin_integration.py`, not just a direct function call
- [ ] New/changed SpatialData elements are asserted valid (`get_model` /
      `Model.validate`) and survive a zarr write/read round-trip
      (`assert_spatial_data_objects_are_identical`)
- [ ] napari viewers come from the `make_napari_viewer` fixture, not a
      hand-rolled `napari.Viewer(show=False)`
- [ ] Pure functions with a stateable invariant have a property-based test
- [ ] Coverage did not fall; if it rose durably, `fail_under` in
      `pyproject.toml` was raised in this PR

## Bug fixes only

- [ ] There is a test that **fails without the fix** — verified by stashing the
      change and watching it fail, not assumed
- [ ] That test reproduces the user-facing path, not a simplified stand-in

## Notes for the reviewer

<!-- Anything unverifiable locally, deliberately deferred, or worth a second
     opinion. Say so explicitly rather than leaving it to be discovered. -->
