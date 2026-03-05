"""Pydantic schemas for structured CAD operation descriptions."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag, model_validator


# ---------------------------------------------------------------------------
# Sketch primitives
# ---------------------------------------------------------------------------

class Point2D(BaseModel):
    x: float
    y: float


Point3D = tuple[float, float, float]
PlaneName = Literal["XY", "YZ", "XZ"]
AxisName = Literal["X", "Y", "Z"]


def _coerce_center_to_origin(data: Any) -> Any:
    """Backward-compatibility adapter for historical `center` field."""
    if not isinstance(data, dict):
        return data
    if "origin" not in data and "center" in data:
        center = data["center"]
        if isinstance(center, (list, tuple)) and len(center) == 3:
            data = {**data, "origin": center}
    return data


class RectangleSketch(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    width: float = Field(..., gt=0, description="Width in mm")
    height: float = Field(..., gt=0, description="Height in mm")
    center: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))


class CircleSketch(BaseModel):
    type: Literal["circle"] = "circle"
    radius: float = Field(..., gt=0, description="Radius in mm")
    center: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))


class PolygonSketch(BaseModel):
    type: Literal["polygon"] = "polygon"
    points: list[Point2D] = Field(..., min_length=3)


SketchPrimitive = Annotated[
    Union[
        Annotated[RectangleSketch, Tag("rectangle")],
        Annotated[CircleSketch, Tag("circle")],
        Annotated[PolygonSketch, Tag("polygon")],
    ],
    Discriminator("type"),
]


# ---------------------------------------------------------------------------
# CAD operations
# ---------------------------------------------------------------------------

class ExtrudeOp(BaseModel):
    op: Literal["extrude"] = "extrude"
    sketch: SketchPrimitive
    depth: float = Field(..., gt=0, description="Extrusion depth in mm")
    plane: PlaneName = Field("XY", description="Sketch plane in world coordinates")
    origin: Point3D = Field((0.0, 0.0, 0.0), description="[x, y, z] origin in mm")

    @model_validator(mode="before")
    @classmethod
    def _compat_center(cls, data: Any) -> Any:
        return _coerce_center_to_origin(data)


class RevolveOp(BaseModel):
    op: Literal["revolve"] = "revolve"
    sketch: SketchPrimitive
    angle: float = Field(360.0, description="Rotation angle in degrees")
    axis: AxisName = Field("Z", description="Axis of revolution")


class FilletOp(BaseModel):
    op: Literal["fillet"] = "fillet"
    radius: float = Field(..., description="Fillet radius in mm")
    edge_selector: str = Field("|Z", description="CadQuery edge selector string")


class ChamferOp(BaseModel):
    op: Literal["chamfer"] = "chamfer"
    length: float = Field(..., description="Chamfer length in mm")
    edge_selector: str = Field("|Z", description="CadQuery edge selector string")


class CutExtrudeOp(BaseModel):
    op: Literal["cut_extrude"] = "cut_extrude"
    sketch: SketchPrimitive
    depth: float = Field(..., gt=0, description="Cut depth in mm")
    plane: PlaneName = Field("XY", description="Sketch plane in world coordinates")
    origin: Point3D = Field((0.0, 0.0, 0.0), description="[x, y, z] origin in mm")

    @model_validator(mode="before")
    @classmethod
    def _compat_center(cls, data: Any) -> Any:
        return _coerce_center_to_origin(data)


class CutCylinderOp(BaseModel):
    op: Literal["cut_cylinder"] = "cut_cylinder"
    radius: float = Field(..., gt=0, description="Hole radius in mm")
    depth: float = Field(..., gt=0, description="Hole depth in mm")
    axis: AxisName = Field("Z", description="Hole axis direction")
    origin: Point3D = Field((0.0, 0.0, 0.0), description="[x, y, z] hole origin in mm")

    @model_validator(mode="before")
    @classmethod
    def _compat_center(cls, data: Any) -> Any:
        return _coerce_center_to_origin(data)


CADOperation = Annotated[
    Union[
        Annotated[ExtrudeOp, Tag("extrude")],
        Annotated[RevolveOp, Tag("revolve")],
        Annotated[FilletOp, Tag("fillet")],
        Annotated[ChamferOp, Tag("chamfer")],
        Annotated[CutExtrudeOp, Tag("cut_extrude")],
        Annotated[CutCylinderOp, Tag("cut_cylinder")],
    ],
    Discriminator("op"),
]


# ---------------------------------------------------------------------------
# Top-level model description
# ---------------------------------------------------------------------------

class CADModel(BaseModel):
    """Complete description of a 3D object as a sequence of CAD operations."""

    name: str = Field(..., description="Short descriptive name for the object")
    description: str = Field("", description="Human-readable description")
    unit: Literal["mm"] = Field("mm", description="Unit of measurement")
    operations: list[CADOperation] = Field(
        ...,
        min_length=1,
        description="Ordered list of CAD operations to build the object",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "simple_bracket",
                "description": "An L-shaped mounting bracket with two holes",
                "unit": "mm",
                "operations": [
                    {
                        "op": "extrude",
                        "plane": "XY",
                        "origin": [0, 0, 0],
                        "sketch": {"type": "rectangle", "width": 40, "height": 60},
                        "depth": 5,
                    },
                    {
                        "op": "cut_cylinder",
                        "radius": 3,
                        "depth": 5,
                        "axis": "Z",
                        "origin": [10, 15, 0],
                    },
                ],
            }
        }
