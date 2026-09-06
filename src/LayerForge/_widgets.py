"""
magicgui-based widgets for the LayerForge napari plugin.

Built entirely from ``magicgui.widgets`` (Container / SpinBox / LineEdit /
ComboBox / FileEdit / PushButton / Label) rather than a hand-rolled
QMainWindow/QWizard, and reports errors via napari's built-in
``napari.utils.notifications.show_error`` instead of a custom QMessageBox.

Public API
----------
AnnotationWidget      – dock widget: define classes, pick a mode, launch a session.
make_annotation_widget – npe2 widget-contribution factory (injects the current Viewer).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox, Container, FileEdit, Label, LineEdit, PushButton, SpinBox

from LayerForge._version import __version__
from LayerForge.annotate_sdata import run_annotation_session

if TYPE_CHECKING:
    import napari


def _sdata_from_viewer(viewer: "napari.viewer.Viewer"):
    """
    Recover the ``(sdata, image_key)`` pair attached to a viewer's layers.

    LayerForge's npe2 reader (see ``_reader.py`` /
    ``sdata_utils.sdata_image_to_layer_data``) stashes the originating
    SpatialData object on every layer's ``metadata["sdata"]`` — the same
    convention napari-spatialdata uses — so this widget never needs its own
    "open file" dialog to find the data being annotated.
    """
    for layer in reversed(list(viewer.layers)):
        sdata = layer.metadata.get("sdata")
        if sdata is not None:
            return sdata, layer.metadata.get("image_key")
    return None, None


class _ClassRow(Container):
    """One editable row of the class-definition table: an integer class id + name."""

    def __init__(self, class_id: int = 1, name: "str | None" = None) -> None:
        # NB: the child widgets are named `class_id`/`class_name`, not `id`/
        # `name` — `Widget.name` is a reserved property on magicgui widgets,
        # so `self.name = ...` would silently overwrite the container's own
        # name instead of storing our LineEdit.
        self.class_id = SpinBox(value=class_id, min=1, max=99, label="id")
        self.class_name = LineEdit(value=name or f"class-{class_id}", label="name")
        super().__init__(widgets=[self.class_id, self.class_name], layout="horizontal", labels=True)


class AnnotationWidget(Container):
    """
    LayerForge dock widget.

    Lets colleagues define semantic classes, choose a mode
    (``labels``/``shapes``) and a mask output path, then launches
    :func:`LayerForge.run_annotation_session` against whichever image was
    opened via napari's native ``File > Open`` / drag-and-drop.
    """

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        self._viewer = viewer

        self._about = Label(value=f"LayerForge v{__version__}")
        self._mode = ComboBox(label="Mode", choices=["labels", "shapes"], value="labels")

        self._class_rows = Container(widgets=[_ClassRow(1)], label="Classes")
        self._add_row_btn = PushButton(text="+ class")
        self._remove_row_btn = PushButton(text="- class")
        row_buttons = Container(
            widgets=[self._add_row_btn, self._remove_row_btn], layout="horizontal", labels=False
        )

        self._output_path = FileEdit(
            label="Mask output (.tif/.npy)",
            mode="w",
            value=str(Path.home() / "layerforge_mask.tif"),
        )
        self._launch_btn = PushButton(text="Launch annotation session")

        super().__init__(
            widgets=[
                self._about,
                self._mode,
                self._class_rows,
                row_buttons,
                self._output_path,
                self._launch_btn,
            ]
        )

        self._add_row_btn.clicked.connect(self._add_class_row)
        self._remove_row_btn.clicked.connect(self._remove_class_row)
        self._launch_btn.clicked.connect(self._on_launch)

    def _add_class_row(self) -> None:
        existing_ids = [row.class_id.value for row in self._class_rows]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        self._class_rows.append(_ClassRow(next_id))

    def _remove_class_row(self) -> None:
        if len(self._class_rows) > 1:
            self._class_rows.pop(-1)

    def _class_labels(self) -> dict[int, str]:
        return {row.class_id.value: row.class_name.value for row in self._class_rows}

    def _on_launch(self) -> None:
        from napari.utils.notifications import show_error

        sdata, image_key = _sdata_from_viewer(self._viewer)
        if sdata is None:
            show_error(
                "LayerForge: open an image first (File > Open, or drag-and-drop) "
                "before launching an annotation session."
            )
            return

        output_path = str(self._output_path.value) if self._output_path.value else None
        if output_path and Path(output_path).suffix.lower() not in (".tif", ".tiff", ".npy"):
            show_error("LayerForge: mask output path must end in .tif, .tiff, or .npy")
            return

        try:
            run_annotation_session(
                sdata,
                image_key=image_key or "image",
                mode=self._mode.value,
                class_labels=self._class_labels(),
                output_path=output_path,
            )
        except Exception as exc:  # surfaced to the user, never raised into napari's event loop
            show_error(f"LayerForge: annotation session failed — {exc}")


def make_annotation_widget(viewer: "napari.viewer.Viewer") -> AnnotationWidget:
    """npe2 widget-contribution factory — napari injects the current Viewer."""
    return AnnotationWidget(viewer)
