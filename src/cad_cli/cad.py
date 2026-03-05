"""CAD generation: convert a validated CADModel into CadQuery geometry and export."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import cadquery as cq

from .errors import ExportFailedError, SchemaInvalidError, TopologyInvalidError
from .schemas import (
    CADModel,
    CADOperation,
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

EPSILON = 1e-6
_ADD_OPS = {"extrude", "revolve"}
_CUT_OPS = {"cut_extrude", "cut_cylinder"}
_COSMETIC_OPS = {"fillet", "chamfer"}


def _new_workplane(plane: str, origin: tuple[float, float, float]) -> cq.Workplane:
    """Create an explicitly placed world workplane."""
    return cq.Workplane(plane).transformed(offset=origin)


def _make_sketch(wp: cq.Workplane, sketch: SketchPrimitive) -> cq.Workplane:
    """Add a sketch primitive to a CadQuery workplane."""
    if isinstance(sketch, RectangleSketch):
        return wp.center(sketch.center.x, sketch.center.y).rect(
            sketch.width, sketch.height
        )
    if isinstance(sketch, CircleSketch):
        return wp.center(sketch.center.x, sketch.center.y).circle(sketch.radius)
    if isinstance(sketch, PolygonSketch):
        pts = [(p.x, p.y) for p in sketch.points]
        return wp.polyline(pts).close()
    raise ValueError(f"Unknown sketch type: {type(sketch)}")


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(area) * 0.5


def _validate_sketch(sketch: SketchPrimitive, *, op_index: int) -> None:
    if isinstance(sketch, RectangleSketch):
        if sketch.width <= EPSILON or sketch.height <= EPSILON:
            raise SchemaInvalidError(
                f"Operation #{op_index} rectangle dimensions must be > 0."
            )
        return

    if isinstance(sketch, CircleSketch):
        if sketch.radius <= EPSILON:
            raise SchemaInvalidError(f"Operation #{op_index} circle radius must be > 0.")
        return

    if isinstance(sketch, PolygonSketch):
        points = [(p.x, p.y) for p in sketch.points]
        if _polygon_area(points) <= EPSILON:
            raise SchemaInvalidError(
                f"Operation #{op_index} polygon area must be > 0."
            )
        return

    raise SchemaInvalidError(f"Operation #{op_index} has unsupported sketch type.")


def _validate_operation(op: CADOperation, *, op_index: int) -> None:
    if op.op in {"extrude", "cut_extrude", "revolve"}:
        _validate_sketch(op.sketch, op_index=op_index)

    if isinstance(op, ExtrudeOp) and op.depth <= EPSILON:
        raise SchemaInvalidError(f"Operation #{op_index} extrude depth must be > 0.")
    if isinstance(op, CutExtrudeOp) and op.depth <= EPSILON:
        raise SchemaInvalidError(
            f"Operation #{op_index} cut_extrude depth must be > 0."
        )
    if isinstance(op, RevolveOp) and abs(op.angle) <= EPSILON:
        raise SchemaInvalidError(f"Operation #{op_index} revolve angle must be non-zero.")
    if isinstance(op, CutCylinderOp):
        if op.radius <= EPSILON:
            raise SchemaInvalidError(
                f"Operation #{op_index} cut_cylinder radius must be > 0."
            )
        if op.depth <= EPSILON:
            raise SchemaInvalidError(
                f"Operation #{op_index} cut_cylinder depth must be > 0."
            )
    if isinstance(op, FilletOp) and op.radius <= EPSILON:
        raise SchemaInvalidError(f"Operation #{op_index} fillet radius must be > 0.")
    if isinstance(op, ChamferOp) and op.length <= EPSILON:
        raise SchemaInvalidError(f"Operation #{op_index} chamfer length must be > 0.")


def normalize_model(model: CADModel) -> CADModel:
    """Validate sequencing and dimensional sanity for a CAD model."""
    if model.unit != "mm":
        raise SchemaInvalidError(
            f"Unsupported unit {model.unit!r}. Only 'mm' is currently supported."
        )

    seen_base = False
    for i, op in enumerate(model.operations):
        _validate_operation(op, op_index=i)

        if op.op in _ADD_OPS:
            seen_base = True
            continue
        if op.op in _CUT_OPS and not seen_base:
            raise SchemaInvalidError(
                f"Operation #{i} ({op.op}) appears before any additive base geometry."
            )
        if op.op in _COSMETIC_OPS and not seen_base:
            raise SchemaInvalidError(
                f"Operation #{i} ({op.op}) appears before base geometry."
            )

    return model


def _solid_from_extrude(op: ExtrudeOp) -> cq.Shape:
    wp = _new_workplane(op.plane, op.origin)
    return _make_sketch(wp, op.sketch).extrude(op.depth).val()


def _solid_from_revolve(op: RevolveOp) -> cq.Shape:
    wp = _make_sketch(cq.Workplane("XY"), op.sketch)
    axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[op.axis]
    return wp.revolve(op.angle, axisStart=(0, 0, 0), axisEnd=axis_vec).val()


def _solid_from_cut_extrude(op: CutExtrudeOp) -> cq.Shape:
    wp = _new_workplane(op.plane, op.origin)
    return _make_sketch(wp, op.sketch).extrude(op.depth).val()


def _solid_from_cut_cylinder(op: CutCylinderOp) -> cq.Shape:
    axis_plane = {"X": "YZ", "Y": "XZ", "Z": "XY"}[op.axis]
    wp = _new_workplane(axis_plane, op.origin)
    return wp.circle(op.radius).extrude(op.depth).val()


def _apply_fillet(shape: cq.Shape, op: FilletOp) -> cq.Shape:
    wp = cq.Workplane("XY").newObject([shape])
    for radius in (op.radius, op.radius / 2):
        try:
            return wp.edges(op.edge_selector).fillet(radius).val()
        except Exception:
            pass
    warnings.warn(
        f"Fillet (r={op.radius}, sel={op.edge_selector!r}) could not be applied; skipping.",
        stacklevel=2,
    )
    return shape


def _apply_chamfer(shape: cq.Shape, op: ChamferOp) -> cq.Shape:
    wp = cq.Workplane("XY").newObject([shape])
    for length in (op.length, op.length / 2):
        try:
            return wp.edges(op.edge_selector).chamfer(length).val()
        except Exception:
            pass
    warnings.warn(
        f"Chamfer (l={op.length}, sel={op.edge_selector!r}) could not be applied; skipping.",
        stacklevel=2,
    )
    return shape


def _solid_metrics(shape: cq.Shape) -> dict[str, object]:
    solids = list(shape.Solids())
    bb = shape.BoundingBox()
    total_volume = sum(abs(s.Volume()) for s in solids)
    return {
        "solid_count": len(solids),
        "valid": bool(shape.isValid()),
        "volume_mm3": round(total_volume, 6),
        "bbox": {
            "min": [round(bb.xmin, 6), round(bb.ymin, 6), round(bb.zmin, 6)],
            "max": [round(bb.xmax, 6), round(bb.ymax, 6), round(bb.zmax, 6)],
            "extents": [round(bb.xlen, 6), round(bb.ylen, 6), round(bb.zlen, 6)],
        },
    }


def _ensure_single_valid_solid(shape: cq.Shape, *, op_index: int, op_name: str) -> cq.Shape:
    solids = list(shape.Solids())
    if len(solids) != 1:
        raise TopologyInvalidError(
            f"Operation #{op_index} ({op_name}) produced {len(solids)} solids; expected 1."
        )
    solid = solids[0]
    if not solid.isValid():
        raise TopologyInvalidError(
            f"Operation #{op_index} ({op_name}) produced invalid topology."
        )
    if abs(solid.Volume()) <= EPSILON:
        raise TopologyInvalidError(
            f"Operation #{op_index} ({op_name}) produced near-zero volume geometry."
        )
    return solid


def _write_debug_checkpoint(
    debug_dir: Path,
    *,
    op_index: int,
    op_name: str,
    shape: cq.Shape,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    wp = cq.Workplane("XY").newObject([shape])
    step_path = debug_dir / f"op_{op_index:02d}_{op_name}.step"
    cq.exporters.export(wp, str(step_path), exportType="STEP")


def build_geometry(model: CADModel, debug_dir: Optional[Path] = None) -> cq.Workplane:
    """Build CadQuery geometry from a CADModel with strict topology invariants."""
    model = normalize_model(model)
    current_shape: Optional[cq.Shape] = None
    debug_rows: list[dict[str, object]] = []

    for i, op in enumerate(model.operations):
        previous_shape = current_shape
        if op.op == "extrude":
            tool = _solid_from_extrude(op)
            current_shape = tool if current_shape is None else current_shape.fuse(tool)
        elif op.op == "revolve":
            tool = _solid_from_revolve(op)
            current_shape = tool if current_shape is None else current_shape.fuse(tool)
        elif op.op == "cut_extrude":
            if current_shape is None:
                raise SchemaInvalidError(
                    f"Operation #{i} ({op.op}) requires existing base geometry."
                )
            tool = _solid_from_cut_extrude(op)
            current_shape = current_shape.cut(tool)
        elif op.op == "cut_cylinder":
            if current_shape is None:
                raise SchemaInvalidError(
                    f"Operation #{i} ({op.op}) requires existing base geometry."
                )
            tool = _solid_from_cut_cylinder(op)
            current_shape = current_shape.cut(tool)
        elif op.op == "fillet":
            if current_shape is None:
                raise SchemaInvalidError(
                    f"Operation #{i} ({op.op}) requires existing base geometry."
                )
            current_shape = _apply_fillet(current_shape, op)
        elif op.op == "chamfer":
            if current_shape is None:
                raise SchemaInvalidError(
                    f"Operation #{i} ({op.op}) requires existing base geometry."
                )
            current_shape = _apply_chamfer(current_shape, op)
        else:
            raise SchemaInvalidError(f"Operation #{i} has unknown type: {op.op}")

        if current_shape is None:
            raise TopologyInvalidError(
                f"Operation #{i} ({op.op}) did not produce geometry."
            )
        try:
            current_shape = _ensure_single_valid_solid(
                current_shape, op_index=i, op_name=op.op
            )
        except TopologyInvalidError:
            if op.op in _COSMETIC_OPS and previous_shape is not None:
                warnings.warn(
                    f"Operation #{i} ({op.op}) produced invalid topology and was skipped.",
                    stacklevel=2,
                )
                current_shape = previous_shape
            else:
                raise

        metrics = _solid_metrics(current_shape)
        debug_rows.append(
            {
                "op_index": i,
                "op": op.op,
                **metrics,
            }
        )
        if debug_dir is not None:
            _write_debug_checkpoint(
                debug_dir=debug_dir,
                op_index=i,
                op_name=op.op,
                shape=current_shape,
            )

    if current_shape is None:
        raise TopologyInvalidError("No geometry was produced from model operations.")

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "op_metrics.json").write_text(json.dumps(debug_rows, indent=2))

    return cq.Workplane("XY").newObject([current_shape])


def export_step(result: cq.Workplane, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cq.exporters.export(result, str(output_path), exportType="STEP")
    except Exception as exc:  # pragma: no cover - CadQuery backend exception types vary.
        raise ExportFailedError(f"STEP export failed for {output_path}: {exc}") from exc
    return output_path


def export_stl(result: cq.Workplane, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cq.exporters.export(result, str(output_path), exportType="STL")
    except Exception as exc:  # pragma: no cover - CadQuery backend exception types vary.
        raise ExportFailedError(f"STL export failed for {output_path}: {exc}") from exc
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
            raise SchemaInvalidError(f"Unsupported export format: {fmt!r}")
    return exported
