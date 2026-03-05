from __future__ import annotations

import unittest

from cad_cli.cad import build_geometry
from cad_cli.errors import SchemaInvalidError, TopologyInvalidError
from cad_cli.schemas import CADModel


class CADBuilderTests(unittest.TestCase):
    def test_extrude_honors_origin_z(self) -> None:
        model = CADModel.model_validate(
            {
                "name": "z_offset_block",
                "unit": "mm",
                "operations": [
                    {
                        "op": "extrude",
                        "plane": "XY",
                        "origin": [0, 0, 5],
                        "depth": 10,
                        "sketch": {
                            "type": "rectangle",
                            "width": 20,
                            "height": 20,
                        },
                    }
                ],
            }
        )
        shape = build_geometry(model).val()
        bb = shape.BoundingBox()
        self.assertAlmostEqual(bb.zmin, 5.0, places=5)
        self.assertAlmostEqual(bb.zmax, 15.0, places=5)

    def test_cut_cylinder_axis_x(self) -> None:
        model = CADModel.model_validate(
            {
                "name": "x_axis_hole",
                "unit": "mm",
                "operations": [
                    {
                        "op": "extrude",
                        "plane": "XY",
                        "origin": [0, 0, 0],
                        "depth": 20,
                        "sketch": {
                            "type": "rectangle",
                            "width": 20,
                            "height": 20,
                            "center": {"x": 10, "y": 10},
                        },
                    },
                    {
                        "op": "cut_cylinder",
                        "axis": "X",
                        "origin": [0, 10, 10],
                        "radius": 2,
                        "depth": 20,
                    },
                ],
            }
        )
        shape = build_geometry(model).val()
        bb = shape.BoundingBox()
        self.assertAlmostEqual(bb.xlen, 20.0, places=5)
        self.assertAlmostEqual(bb.ylen, 20.0, places=5)
        self.assertAlmostEqual(bb.zlen, 20.0, places=5)
        self.assertTrue(shape.isValid())

    def test_center_field_is_backward_compatible(self) -> None:
        model = CADModel.model_validate(
            {
                "name": "compat_center",
                "unit": "mm",
                "operations": [
                    {
                        "op": "extrude",
                        "center": [0, 0, 7],
                        "depth": 3,
                        "sketch": {
                            "type": "rectangle",
                            "width": 10,
                            "height": 10,
                        },
                    }
                ],
            }
        )
        shape = build_geometry(model).val()
        bb = shape.BoundingBox()
        self.assertAlmostEqual(bb.zmin, 7.0, places=5)
        self.assertAlmostEqual(bb.zmax, 10.0, places=5)

    def test_disjoint_solids_are_rejected(self) -> None:
        model = CADModel.model_validate(
            {
                "name": "disjoint",
                "unit": "mm",
                "operations": [
                    {
                        "op": "extrude",
                        "origin": [0, 0, 0],
                        "depth": 5,
                        "sketch": {
                            "type": "rectangle",
                            "width": 10,
                            "height": 10,
                        },
                    },
                    {
                        "op": "extrude",
                        "origin": [30, 0, 0],
                        "depth": 5,
                        "sketch": {
                            "type": "rectangle",
                            "width": 10,
                            "height": 10,
                        },
                    },
                ],
            }
        )
        with self.assertRaises(TopologyInvalidError):
            build_geometry(model)

    def test_cut_before_base_is_rejected(self) -> None:
        model = CADModel.model_validate(
            {
                "name": "bad_order",
                "unit": "mm",
                "operations": [
                    {
                        "op": "cut_cylinder",
                        "radius": 2,
                        "depth": 5,
                    }
                ],
            }
        )
        with self.assertRaises(SchemaInvalidError):
            build_geometry(model)


if __name__ == "__main__":
    unittest.main()
