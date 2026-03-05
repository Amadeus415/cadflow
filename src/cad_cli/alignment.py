"""Image vs CAD alignment scoring utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .errors import AlignmentLowError

_VIEWS = ("pos_z", "neg_z", "pos_x", "neg_x", "pos_y", "neg_y")


def _normalize_mask(mask: np.ndarray, *, size: int = 256, margin: int = 12) -> np.ndarray:
    """Crop to foreground and fit into a fixed-size canvas."""
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    h, w = crop.shape

    fit = max(1, size - 2 * margin)
    scale = fit / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = (
        Image.fromarray((crop * 255).astype(np.uint8))
        .resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    )
    canvas = np.zeros((size, size), dtype=bool)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(resized) > 0
    return canvas


def _extract_image_mask(image_path: Path, *, size: int = 256) -> np.ndarray:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)

    bg = np.percentile(gray, 95)
    mask = (gray < (bg - 18.0)) | (chroma > 20.0)
    fg_ratio = float(mask.mean())

    if fg_ratio < 0.005:
        fallback = gray < np.percentile(gray, 70)
        if fallback.mean() > 0.005:
            mask = fallback
    if mask.mean() > 0.9:
        fallback = gray < np.percentile(gray, 60)
        if fallback.mean() < 0.9:
            mask = fallback

    return _normalize_mask(mask, size=size)


def _project_vertices(vertices: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    if view == "pos_z":
        return x, y
    if view == "neg_z":
        return -x, y
    if view == "pos_x":
        return y, z
    if view == "neg_x":
        return -y, z
    if view == "pos_y":
        return x, z
    if view == "neg_y":
        return -x, z
    raise ValueError(f"Unknown view: {view}")


def _render_mesh_mask(mesh: trimesh.Trimesh, *, view: str, size: int = 256) -> np.ndarray:
    u, v = _project_vertices(mesh.vertices, view)
    points = np.column_stack([u, v])
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if points.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((size, size), dtype=bool)

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    fit = size - 24
    scale = fit / float(max(span[0], span[1]))
    pix = (points - min_xy) * scale
    width = (max_xy[0] - min_xy[0]) * scale
    height = (max_xy[1] - min_xy[1]) * scale
    pix[:, 0] += (size - width) / 2
    pix[:, 1] += (size - height) / 2
    pix[:, 1] = size - 1 - pix[:, 1]

    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for tri in faces:
        tri_pts = [
            (float(pix[tri[0], 0]), float(pix[tri[0], 1])),
            (float(pix[tri[1], 0]), float(pix[tri[1], 1])),
            (float(pix[tri[2], 0]), float(pix[tri[2], 1])),
        ]
        draw.polygon(tri_pts, fill=255)
    return _normalize_mask(np.asarray(image) > 0, size=size)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        s = mask.shape[0]
        return np.array([s / 2.0, s / 2.0], dtype=np.float64)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _load_mesh_from_geometry(geometry: cq.Workplane) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cq.exporters.export(geometry, str(tmp_path), exportType="STL")
        mesh = trimesh.load_mesh(tmp_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise AlignmentLowError("Could not load CAD geometry into a trimesh mesh.")
        if len(mesh.faces) == 0:
            raise AlignmentLowError("Generated mesh is empty; cannot score alignment.")
        return mesh
    finally:
        tmp_path.unlink(missing_ok=True)


def score_alignment(
    *,
    image_path: Path,
    geometry: cq.Workplane,
    output_path: Path | None = None,
    min_iou: float = 0.70,
) -> dict[str, object]:
    """Compute silhouette alignment score between an input image and CAD geometry."""
    image_mask = _extract_image_mask(image_path, size=256)
    mesh = _load_mesh_from_geometry(geometry)

    best: dict[str, object] | None = None
    candidates: list[dict[str, object]] = []

    image_centroid = _centroid(image_mask)
    for view in _VIEWS:
        base_mask = _render_mesh_mask(mesh, view=view, size=256)
        for rot in range(4):
            cad_mask = np.rot90(base_mask, k=rot)
            iou = _iou(image_mask, cad_mask)
            centroid_error = float(np.linalg.norm(image_centroid - _centroid(cad_mask)) / 256.0)
            score = iou - 0.15 * centroid_error
            candidate = {
                "view": view,
                "rotation_quarter_turns": rot,
                "iou": round(iou, 6),
                "centroid_error_norm": round(centroid_error, 6),
                "score": round(score, 6),
            }
            candidates.append(candidate)
            if best is None or score > float(best["score"]):
                best = candidate

    if best is None:
        raise AlignmentLowError("Alignment scoring could not generate any candidates.")

    report = {
        "threshold_iou": min_iou,
        "passes_threshold": float(best["iou"]) >= min_iou,
        "best": best,
        "image_foreground_ratio": round(float(image_mask.mean()), 6),
        "top_candidates": sorted(candidates, key=lambda c: c["score"], reverse=True)[:8],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    return report
