"""
Tests for the LayerForge napari plugin widgets (_widgets.py) — headless,
using napari.Viewer(show=False) like the rest of the suite.
"""
import numpy as np
import napari
import pytest
import spatialdata
from spatialdata.models import Image2DModel

from LayerForge._version import __version__
from LayerForge._widgets import AnnotationWidget, _sdata_from_viewer


@pytest.fixture
def demo_sdata():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 255, size=(1, 32, 32), dtype=np.uint8)
    image = Image2DModel.parse(data, dims=["c", "y", "x"])
    return spatialdata.SpatialData(images={"image": image})


@pytest.fixture
def headless_viewer():
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_sdata_from_viewer_returns_none_when_no_layers(headless_viewer):
    sdata, image_key = _sdata_from_viewer(headless_viewer)
    assert sdata is None
    assert image_key is None


def test_sdata_from_viewer_recovers_metadata(headless_viewer, demo_sdata):
    headless_viewer.add_image(
        np.zeros((32, 32)), name="ch-0", metadata={"sdata": demo_sdata, "image_key": "image"}
    )
    sdata, image_key = _sdata_from_viewer(headless_viewer)
    assert sdata is demo_sdata
    assert image_key == "image"


def test_annotation_widget_is_instantiated_by_napari_dock_widget_injection(headless_viewer):
    """
    Regression test: napari's dock-widget injection only recognises a
    `viewer: napari.viewer.Viewer` annotation on a widget *class*'s
    `__init__` (matched as an exact string/type, see
    napari._qt.qt_main_window._instantiate_dock_widget). A quoted forward
    reference under `from __future__ import annotations` silently becomes
    the literal string '"napari.viewer.Viewer"' instead, which breaks this
    without raising an import-time error — so this must be exercised via
    napari's own widget-injection entry point, not by constructing
    AnnotationWidget directly.
    """
    _dock_widget, widget = headless_viewer.window.add_plugin_dock_widget(
        "LayerForge", "LayerForge annotation panel"
    )
    assert isinstance(widget, AnnotationWidget)
    assert widget._about.value == f"LayerForge v{__version__}"


def test_annotation_widget_default_class_row(headless_viewer):
    widget = AnnotationWidget(headless_viewer)
    assert widget._class_labels() == {1: "class-1"}


def test_annotation_widget_add_and_remove_class_row(headless_viewer):
    widget = AnnotationWidget(headless_viewer)
    widget._add_class_row()
    assert widget._class_labels() == {1: "class-1", 2: "class-2"}

    widget._remove_class_row()
    assert widget._class_labels() == {1: "class-1"}

    # never remove the last remaining row
    widget._remove_class_row()
    assert len(widget._class_labels()) == 1


def test_annotation_widget_launch_without_image_shows_error(headless_viewer, monkeypatch):
    widget = AnnotationWidget(headless_viewer)

    errors = []
    monkeypatch.setattr(
        "napari.utils.notifications.show_error", lambda msg: errors.append(msg)
    )

    widget._on_launch()

    assert len(errors) == 1
    assert "open an image first" in errors[0]


def test_annotation_widget_launch_wires_up_run_annotation_session(
    headless_viewer, demo_sdata, monkeypatch
):
    headless_viewer.add_image(
        np.zeros((32, 32)), name="ch-0", metadata={"sdata": demo_sdata, "image_key": "image"}
    )
    widget = AnnotationWidget(headless_viewer)

    calls = {}

    def fake_run(sdata, **kwargs):
        calls["sdata"] = sdata
        calls["kwargs"] = kwargs

    monkeypatch.setattr("LayerForge._widgets.run_annotation_session", fake_run)

    widget._on_launch()

    assert calls["sdata"] is demo_sdata
    assert calls["kwargs"]["image_key"] == "image"
    assert calls["kwargs"]["mode"] == "labels"
    assert calls["kwargs"]["class_labels"] == {1: "class-1"}
