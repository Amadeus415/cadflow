"""Pydantic schemas for structured CAD operation descriptions."""

from __future__ import annotations

from typing import Literal, Optional, Union, Annotated

from pydantic import BaseModel, Discriminator, Field, Tag


# ---------------------------------------------------------------------------
# Sketch primitives
# ---------------------------------------------------------------------------

class Point2D(BaseModel):
    x: float
    y: float


class RectangleSketch(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    width: float = Field(..., description="Width in mm")
    height: float = Field(..., description="Height in mm")
    center: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))


class CircleSketch(BaseModel):
    type: Literal["circle"] = "circle"
    radius: float = Field(..., description="Radius in mm")
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
    depth: float = Field(..., description="Extrusion depth in mm")
    center: Optional[list[float]] = Field(None, description="[x, y, z] placement offset")


class RevolveOp(BaseModel):
    op: Literal["revolve"] = "revolve"
    sketch: SketchPrimitive
    angle: float = Field(360.0, description="Rotation angle in degrees")
    axis: str = Field("Z", description="Axis of revolution: X, Y, or Z")


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
    depth: float = Field(..., description="Cut depth in mm")
    center: Optional[list[float]] = Field(None, description="[x, y, z] placement offset")


class CutCylinderOp(BaseModel):
    op: Literal["cut_cylinder"] = "cut_cylinder"
    radius: float = Field(..., description="Hole radius in mm")
    depth: float = Field(..., description="Hole depth in mm")
    center: Optional[list[float]] = Field(None, description="[x, y, z] position of hole center")


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
    unit: str = Field("mm", description="Unit of measurement")
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
                        "sketch": {"type": "rectangle", "width": 40, "height": 60},
                        "depth": 5,
                    },
                    {
                        "op": "cut_cylinder",
                        "radius": 3,
                        "depth": 5,
                        "center": [10, 15, 5],
                    },
                ],
            }
        }
