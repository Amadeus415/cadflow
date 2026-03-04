"""CAD generation: convert a validated CADModel into CadQuery geometry and export."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cadquery as cq

from .schemas import (
    CADModel,
    ChamferOp,
    CircleSketch,
    CutCylinderOp,
    CutExtrudeOp,
    ExtrudeOp,
    FilletOp,
    PolygonSketch,
    RectangleSketch,
    RevolveOp,
    SketchPrimitive,
)


def _make_sketch(wp: cq.Workplane, sketch: SketchPrimitive) -> cq.Workplane:
    """Add a sketch primitive to a CadQuery workplane."""
    if isinstance(sketch, RectangleSketch):
        return wp.center(sketch.center.x, sketch.center.y).rect(
            sketch.width, sketch.height
        )
    elif isinstance(sketch, CircleSketch):
        return wp.center(sketch.center.x, sketch.center.y).circle(sketch.radius)
    elif isinstance(sketch, PolygonSketch):
        pts = [(p.x, p.y) for p in sketch.points]
        return wp.polyline(pts).close()
    else:
        raise ValueError(f"Unknown sketch type: {type(sketch)}")


def _apply_extrude(result: cq.Workplane, op: ExtrudeOp) -> cq.Workplane:
    wp = result.workplane()
    if op.center:
        wp = wp.center(op.center[0], op.center[1])
    wp = _make_sketch(wp, op.sketch)
    return wp.extrude(op.depth)


def _apply_revolve(result: cq.Workplane, op: RevolveOp) -> cq.Workplane:
    wp = result.workplane()
    wp = _make_sketch(wp, op.sketch)
    axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}.get(
        op.axis.upper(), (0, 0, 1)
    )
    return wp.revolve(op.angle, axisStart=(0, 0, 0), axisEnd=axis_vec)


def _apply_fillet(result: cq.Workplane, op: FilletOp) -> cq.Workplane:
    # Try with requested selector and radius, then halved radius, then skip.
    for radius in (op.radius, op.radius / 2):
        try:
            return result.edges(op.edge_selector).fillet(radius)
        except Exception:
            pass
    warnings.warn(
        f"Fillet (r={op.radius}, sel={op.edge_selector!r}) could not be applied; skipping.",
        stacklevel=2,
    )
    return result


def _apply_chamfer(result: cq.Workplane, op: ChamferOp) -> cq.Workplane:
    # Try with requested selector and length, then halved length, then skip.
    for length in (op.length, op.length / 2):
        try:
            return result.edges(op.edge_selector).chamfer(length)
        except Exception:
            pass
    warnings.warn(
        f"Chamfer (l={op.length}, sel={op.edge_selector!r}) could not be applied; skipping.",
        stacklevel=2,
    )
    return result


def _apply_cut_extrude(result: cq.Workplane, op: CutExtrudeOp) -> cq.Workplane:
    wp = result.workplane()
    if op.center:
        wp = wp.center(op.center[0], op.center[1])
    wp = _make_sketch(wp, op.sketch)
    return wp.cutBlind(-op.depth)


def _apply_cut_cylinder(result: cq.Workplane, op: CutCylinderOp) -> cq.Workplane:
    center = op.center or [0, 0, 0]
    hole = (
        cq.Workplane("XY")
        .transformed(offset=(center[0], center[1], center[2]))
        .circle(op.radius)
        .extrude(op.depth)
    )
    return result.cut(hole)


_APPLICATORS = {
    "extrude": _apply_extrude,
    "revolve": _apply_revolve,
    "fillet": _apply_fillet,
    "chamfer": _apply_chamfer,
    "cut_extrude": _apply_cut_extrude,
    "cut_cylinder": _apply_cut_cylinder,
}


# Operations that are cosmetic and can be safely skipped on failure.
_SKIPPABLE_OPS = {"fillet", "chamfer"}


def build_geometry(model: CADModel) -> cq.Workplane:
    """Build CadQuery geometry from a CADModel."""
    result = cq.Workplane("XY")
    for i, op in enumerate(model.operations):
        applicator = _APPLICATORS.get(op.op)
        if applicator is None:
            raise ValueError(f"Operation #{i} has unknown type: {op.op}")
        try:
            result = applicator(result, op)
        except Exception as exc:
            if op.op in _SKIPPABLE_OPS:
                warnings.warn(
                    f"Operation #{i} ({op.op}) failed and was skipped: {exc}",
                    stacklevel=2,
                )
            else:
                raise
    return result


def export_step(result: cq.Workplane, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(result, str(output_path), exportType="STEP")
    return output_path


def export_stl(result: cq.Workplane, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(result, str(output_path), exportType="STL")
    return output_path


def export_all(
    result: cq.Workplane,
    output_dir: Path,
    name: str,
    formats: Optional[list[str]] = None,
) -> dict[str, Path]:
    """Export geometry in multiple formats. Returns {format: path} dict."""
    formats = formats or ["step", "stl"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for fmt in formats:
        fl = fmt.lower()
        if fl == "step":
            exported["step"] = export_step(result, output_dir / f"{name}.step")
        elif fl == "stl":
            exported["stl"] = export_stl(result, output_dir / f"{name}.stl")
        else:
            raise ValueError(f"Unsupported export format: {fmt!r}")
    return exported
