#!/usr/bin/env python3
"""
Build and draw full-map lane curves from tile control points.

The script reads control-point templates from the map, expands them into
world-space curves, and writes an SVG with control polygons and connected
component annotations.

Example:
  python draw_control_point_curves.py --map loop
  python draw_control_point_curves.py
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from ruamel.yaml import YAML

import lane_utils as lu
from map_interpreter_patch import PatchedMapInterpreter


Point = Tuple[float, float, float]


@dataclass(frozen=True)
class Tile:
    key: str
    map_name: str
    i: int
    j: int
    tile_type: str
    pose_x: float
    pose_y: float
    yaw: float


@dataclass(frozen=True)
class Curve:
    curve_id: int
    tile_key: str
    lane_idx: int
    cps: Tuple[Point, Point, Point, Point]

    @property
    def start(self) -> Point:
        return self.cps[0]

    @property
    def end(self) -> Point:
        return self.cps[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map",
        default="all",
        type=str,
        help="map folder under maps/, or 'all' to draw every map",
    )
    parser.add_argument(
        "--maps-dir",
        default="maps",
        type=str,
        help="root directory that contains all map folders",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="output SVG path for a single map run",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="output directory for generated SVGs (default: maps/control_point_curves)",
    )
    parser.add_argument(
        "--samples",
        default=40,
        type=int,
        help="sample count per Bezier segment for drawing",
    )
    return parser.parse_args()


def _discover_map_dirs(maps_root: Path) -> List[Path]:
    if not maps_root.exists():
        raise FileNotFoundError(f"Maps root not found: {maps_root}")
    map_dirs: List[Path] = []
    for child in sorted(maps_root.iterdir()):
        if child.is_dir() and (child / "main.yaml").exists():
            map_dirs.append(child)
    if not map_dirs:
        raise ValueError(f"No maps found under: {maps_root}")
    return map_dirs


def _selected_map_dirs(maps_root: Path, map_arg: str) -> List[Path]:
    if map_arg.strip().lower() in {"all", "*"}:
        return _discover_map_dirs(maps_root)

    map_dir = maps_root / map_arg
    if not map_dir.exists():
        raise FileNotFoundError(f"Map directory not found: {map_dir}")
    if not (map_dir / "main.yaml").exists():
        raise FileNotFoundError(f"Map directory does not contain main.yaml: {map_dir}")
    return [map_dir]


def _load_yaml(path: Path) -> dict:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root is not a dict: {path}")
    return data


def _parse_tile_key(key: str) -> Tuple[str, int, int]:
    m = re.fullmatch(r"([^/]+)/tile_(\d+)_(\d+)", key)
    if not m:
        raise ValueError(f"Invalid tile key: {key}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rot_y(p: Point, angle: float) -> Point:
    # Rotate a point around the y axis using row-vector matrix multiplication.
    x, y, z = p
    c = math.cos(angle)
    s = math.sin(angle)
    return (x * c - z * s, y, x * s + z * c)


def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _sdk_to_lane_point(p_sdk: Point) -> Point:
    if lu.LANE_UP_AXIS == "z":
        sign = float(lu.LANE_Z_SIGN) if float(lu.LANE_Z_SIGN) != 0.0 else 1.0
        return (float(p_sdk[0]), float(p_sdk[2]), float(p_sdk[1] * sign))
    return (float(p_sdk[0]), float(p_sdk[1]), float(p_sdk[2]))


def _forward_patch_point_lane(tile: Tile, tile_size: float, local_unit: Point) -> Point:
    # Transform a local control point into world coordinates in lane space.
    local_metric = (
        float(local_unit[0]) * float(tile_size),
        float(local_unit[1]) * float(tile_size),
        float(local_unit[2]) * float(tile_size),
    )
    rel_sdk = _rot_y(local_metric, tile.yaw)
    offset_sdk = ((tile.pose_x + 0.5) * tile_size, 0.0, (tile.pose_y + 0.5) * tile_size)
    world_sdk = _add(rel_sdk, offset_sdk)
    return _sdk_to_lane_point(world_sdk)


def _straight_center_line_world(tile: Tile, tile_size: float) -> Tuple[Point, Point]:
    p0_lane = _forward_patch_point_lane(tile, tile_size, (0.0, 0.0, -0.5))
    p1_lane = _forward_patch_point_lane(tile, tile_size, (0.0, 0.0, 0.5))
    return p0_lane, p1_lane


def _center_line_control_points_world(tile: Tile, tile_size: float) -> Optional[Tuple[Point, Point, Point, Point]]:
    templates: Dict[str, List[Point]] = {
        "straight_lane_0": [
            (-0.20, 0.0, 0.50),
            (-0.20, 0.0, 0.25),
            (-0.20, 0.0, -0.25),
            (-0.20, 0.0, -0.50),
        ],
        "straight_lane_1": [
            (0.20, 0.0, -0.50),
            (0.20, 0.0, -0.25),
            (0.20, 0.0, 0.25),
            (0.20, 0.0, 0.50),
        ],
        "curve_lane_0": [
            (-0.50, 0.0, -0.20),
            (0.00, 0.0, -0.20),
            (0.20, 0.0, 0.00),
            (0.20, 0.0, 0.50),
        ],
        "curve_lane_1": [
            (-0.20, 0.0, 0.50),
            (-0.20, 0.0, 0.30),
            (-0.30, 0.0, 0.20),
            (-0.50, 0.0, 0.20),
        ],
    }

    if tile.tile_type == "straight":
        lane0 = templates["straight_lane_0"]
        lane1 = list(reversed(templates["straight_lane_1"]))
    elif tile.tile_type == "curve":
        lane0 = templates["curve_lane_0"]
        lane1 = list(reversed(templates["curve_lane_1"]))
    else:
        return None

    center_local: List[Point] = []
    for a, b in zip(lane0, lane1):
        center_local.append(
            (
                0.5 * (float(a[0]) + float(b[0])),
                0.5 * (float(a[1]) + float(b[1])),
                0.5 * (float(a[2]) + float(b[2])),
            )
        )

    world = tuple(_forward_patch_point_lane(tile, tile_size, p) for p in center_local)
    return world  # type: ignore[return-value]


def _bezier_point(cps: Tuple[Point, Point, Point, Point], t: float) -> Point:
    p0, p1, p2, p3 = cps
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    return (
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
        b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2],
    )


def _sample_bezier(cps: Tuple[Point, Point, Point, Point], n: int) -> List[Point]:
    if n < 2:
        n = 2
    return [_bezier_point(cps, i / float(n - 1)) for i in range(n)]


def _load_map_tiles(map_dir: Path) -> Tuple[Dict[str, Tile], float, Dict[str, Dict]]:
    tiles_yaml = _load_yaml(map_dir / "tiles.yaml").get("tiles", {})
    frames_yaml = _load_yaml(map_dir / "frames.yaml").get("frames", {})
    tile_maps_yaml = _load_yaml(map_dir / "tile_maps.yaml").get("tile_maps", {})

    if not isinstance(tiles_yaml, dict) or not isinstance(frames_yaml, dict):
        raise ValueError("Invalid tiles.yaml / frames.yaml structure")

    if not tile_maps_yaml:
        raise ValueError("tile_maps.yaml is empty")
    first_map = next(iter(tile_maps_yaml.values()))
    if not isinstance(first_map, dict):
        raise ValueError("Invalid tile_maps.yaml structure")
    tile_size_dict = first_map.get("tile_size")
    if not isinstance(tile_size_dict, dict):
        raise ValueError("tile_maps.yaml missing tile_size")
    tile_size = float(tile_size_dict.get("x", 0.0))
    if tile_size <= 0.0:
        raise ValueError(f"Invalid tile size: {tile_size}")

    out: Dict[str, Tile] = {}
    for key, tile_desc in tiles_yaml.items():
        if not isinstance(key, str) or not isinstance(tile_desc, dict):
            continue
        if key not in frames_yaml:
            continue
        frame = frames_yaml[key]
        if not isinstance(frame, dict):
            continue
        pose = frame.get("pose")
        if not isinstance(pose, dict):
            continue

        map_name, i, j = _parse_tile_key(key)
        tile_type = str(tile_desc.get("type", "")).strip().lower()
        out[key] = Tile(
            key=key,
            map_name=map_name,
            i=i,
            j=j,
            tile_type=tile_type,
            pose_x=float(pose.get("x", i)),
            pose_y=float(pose.get("y", j)),
            yaw=float(pose.get("yaw", 0.0)),
        )
    map_payload = {
        "frames": {"data": frames_yaml},
        "tiles": {"data": tiles_yaml},
        "tile_info": {"data": tile_maps_yaml},
    }
    return out, tile_size, map_payload


def build_curves(tiles: Dict[str, Tile], map_payload: Dict[str, Dict]) -> List[Curve]:
    map_int = PatchedMapInterpreter(map=map_payload)
    curves: List[Curve] = []
    idx = 0
    for key in sorted(tiles.keys()):
        tile = tiles[key]
        map_tile = map_int.get_tile(tile.i, tile.j)
        if map_tile is None or not map_tile.get("drivable", False):
            continue

        map_curves = map_tile.get("curves")
        if map_curves is None:
            continue
        lane_curves = lu._apply_curve_offset(None, map_curves) or []
        for lane_idx, cps in enumerate(lane_curves):
            cps_arr = np.asarray(cps, dtype=np.float32)
            if cps_arr.shape[0] < 4:
                continue
            world_cps = [
                (float(p[0]), float(p[1]), float(p[2]))
                for p in cps_arr[:4]
            ]
            curves.append(
                Curve(
                    curve_id=idx,
                    tile_key=tile.key,
                    lane_idx=lane_idx,
                    cps=(world_cps[0], world_cps[1], world_cps[2], world_cps[3]),
                )
            )
            idx += 1
    return curves


def _curve_neighbors(curves: List[Curve], eps: float = 1e-6) -> Dict[int, Set[int]]:
    # Build an undirected graph by matching curve endpoints.
    neighbors: Dict[int, Set[int]] = {c.curve_id: set() for c in curves}
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            ci = curves[i]
            cj = curves[j]
            matched = (
                _dist(ci.start, cj.start) <= eps
                or _dist(ci.start, cj.end) <= eps
                or _dist(ci.end, cj.start) <= eps
                or _dist(ci.end, cj.end) <= eps
            )
            if matched:
                neighbors[ci.curve_id].add(cj.curve_id)
                neighbors[cj.curve_id].add(ci.curve_id)
    return neighbors


def _ground_coords(p: Point) -> Tuple[float, float]:
    if lu.LANE_UP_AXIS == "z":
        return float(p[0]), float(p[1])
    return float(p[0]), float(p[2])


def _connected_components(neighbors: Dict[int, Set[int]]) -> List[List[int]]:
    comps: List[List[int]] = []
    seen: Set[int] = set()
    for node in sorted(neighbors.keys()):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comp.sort()
        comps.append(comp)
    return comps


def _canvas_transform(
    gx: float,
    gy: float,
    min_gx: float,
    max_gx: float,
    min_gy: float,
    max_gy: float,
    width: int,
    height: int,
    margin: int,
) -> Tuple[float, float]:
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    span_x = max(max_gx - min_gx, 1e-9)
    span_y = max(max_gy - min_gy, 1e-9)
    s = min(usable_w / span_x, usable_h / span_y)
    x0 = margin + (gx - min_gx) * s
    y0 = height - margin - (gy - min_gy) * s
    return x0, y0


def _svg_polyline(points_2d: Iterable[Tuple[float, float]]) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points_2d)
    return f'points="{pts}"'


def draw_svg(
    map_name: str,
    tiles: Dict[str, Tile],
    curves: List[Curve],
    components: List[List[int]],
    tile_size: float,
    out_path: Path,
    samples: int,
) -> None:
    if not curves:
        raise ValueError("No drivable curves found")

    width, height = 1100, 900
    margin = 70
    bg = "#ffffff"
    tile_fill = "#f8f8f8"
    tile_stroke = "#d0d0d0"
    text = "#222222"
    ctrl_line = "#9ca3af"
    ctrl_pt = "#111827"
    start_pt = "#16a34a"
    end_pt = "#dc2626"
    center_yellow = "#facc15"
    palette = [
        "#2563eb",
        "#e11d48",
        "#0891b2",
        "#a16207",
        "#7c3aed",
        "#15803d",
        "#ea580c",
    ]

    all_ground = [_ground_coords(p) for c in curves for p in c.cps]
    min_gx = min(min(g[0] for g in all_ground), min(t.pose_x * tile_size for t in tiles.values()))
    max_gx = max(max(g[0] for g in all_ground), max((t.pose_x + 1.0) * tile_size for t in tiles.values()))
    min_gy = min(min(g[1] for g in all_ground), min(t.pose_y * tile_size for t in tiles.values()))
    max_gy = max(max(g[1] for g in all_ground), max((t.pose_y + 1.0) * tile_size for t in tiles.values()))

    comp_color: Dict[int, str] = {}
    for idx, comp in enumerate(components):
        col = palette[idx % len(palette)]
        for cid in comp:
            comp_color[cid] = col

    curve_by_id = {c.curve_id: c for c in curves}
    unique_curve_colors = sorted(set(comp_color.values()))

    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}" />')
    out.append("<defs>")
    for idx, color in enumerate(unique_curve_colors):
        out.append(
            f'<marker id="curve-arrow-{idx}" markerWidth="10" markerHeight="10" refX="8" refY="5" '
            'orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}" />'
            "</marker>"
        )
    out.append("</defs>")
    color_marker = {color: f"curve-arrow-{idx}" for idx, color in enumerate(unique_curve_colors)}

    # Draw the tile grid first.
    for tile in sorted(tiles.values(), key=lambda t: (t.i, t.j)):
        cx, cy = _canvas_transform(
            (tile.pose_x + 0.5) * tile_size,
            (tile.pose_y + 0.5) * tile_size,
            min_gx,
            max_gx,
            min_gy,
            max_gy,
            width,
            height,
            margin,
        )
        p1 = _canvas_transform(
            tile.pose_x * tile_size,
            tile.pose_y * tile_size,
            min_gx,
            max_gx,
            min_gy,
            max_gy,
            width,
            height,
            margin,
        )
        p2 = _canvas_transform(
            (tile.pose_x + 1.0) * tile_size,
            (tile.pose_y + 1.0) * tile_size,
            min_gx,
            max_gx,
            min_gy,
            max_gy,
            width,
            height,
            margin,
        )
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        out.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{tile_fill}" stroke="{tile_stroke}" stroke-width="1.5" />'
        )
        out.append(
            f'<text x="{(cx - w * 0.45):.2f}" y="{(cy - h * 0.38):.2f}" fill="{text}" '
            f'font-size="13" font-family="monospace">{tile.tile_type} ({tile.i},{tile.j})</text>'
        )

    # Draw center reference lines for straight and curve tiles.
    for tile in sorted(tiles.values(), key=lambda t: (t.i, t.j)):
        center_cps = _center_line_control_points_world(tile, tile_size)
        if center_cps is None:
            continue
        center_pts = _sample_bezier(center_cps, samples)
        center_xy = [
            _canvas_transform(
                _ground_coords(p)[0],
                _ground_coords(p)[1],
                min_gx,
                max_gx,
                min_gy,
                max_gy,
                width,
                height,
                margin,
            )
            for p in center_pts
        ]
        out.append(
            f'<polyline {_svg_polyline(center_xy)} fill="none" stroke="{center_yellow}" stroke-width="5" '
            f'stroke-linecap="round" stroke-linejoin="round" />'
        )

    # Draw control polygons and control points.
    for curve in curves:
        cp_xy: List[Tuple[float, float]] = []
        for p in curve.cps:
            gx, gy = _ground_coords(p)
            x, y = _canvas_transform(gx, gy, min_gx, max_gx, min_gy, max_gy, width, height, margin)
            cp_xy.append((x, y))

        out.append(
            f'<polyline {_svg_polyline(cp_xy)} fill="none" stroke="{ctrl_line}" '
            f'stroke-width="1.5" stroke-dasharray="5 4" />'
        )
        for k, (x, y) in enumerate(cp_xy):
            fill = start_pt if k == 0 else (end_pt if k == 3 else ctrl_pt)
            radius = 4.0 if k in (0, 3) else 3.0
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" />')

    # Draw curves using one color per connected component.
    for comp in components:
        for cid in comp:
            curve = curve_by_id[cid]
            pts = _sample_bezier(curve.cps, samples)
            pts_xy = [
                _canvas_transform(
                    _ground_coords(p)[0],
                    _ground_coords(p)[1],
                    min_gx,
                    max_gx,
                    min_gy,
                    max_gy,
                    width,
                    height,
                    margin,
                )
                for p in pts
            ]
            color = comp_color[cid]
            out.append(
                f'<polyline {_svg_polyline(pts_xy)} fill="none" stroke="{color}" stroke-width="4" '
                f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{color_marker[color]})" />'
            )

    # Draw the title and legend text.
    title = f"Bezier Lane Curves From Tile Control Points [{map_name}]"
    out.append(
        f'<text x="{margin}" y="36" fill="{text}" font-size="22" font-family="monospace">{title}</text>'
    )
    out.append(
        f'<text x="{margin}" y="{height - 44}" fill="{text}" font-size="15" font-family="monospace">'
        "Gray dashed: control polygon; Green/Red: p0/p3 control points; Colored arrows: lane direction p0 -> p3"
        "</text>"
    )

    # Show connected-component counts.
    comp_sizes = ", ".join(str(len(c)) for c in components)
    out.append(
        f'<text x="{margin}" y="{height - 20}" fill="{text}" font-size="14" font-family="monospace">'
        f"Connected components (curve count): [{comp_sizes}]"
        "</text>"
    )

    out.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")


def _render_map(map_dir: Path, out_path: Path, samples: int) -> Tuple[str, float, int, List[List[int]], Path]:
    tiles, tile_size, map_payload = _load_map_tiles(map_dir)
    curves = build_curves(tiles, map_payload)
    neighbors = _curve_neighbors(curves)
    components = _connected_components(neighbors)
    draw_svg(map_dir.name, tiles, curves, components, tile_size, out_path, samples=samples)
    return map_dir.name, tile_size, len(curves), components, out_path


def main() -> None:
    args = parse_args()
    maps_root = Path(args.maps_dir)
    map_dirs = _selected_map_dirs(maps_root, args.map)

    if args.out and len(map_dirs) != 1:
        raise ValueError("--out can only be used when rendering exactly one map.")

    default_out_dir = maps_root / "control_point_curves"
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    for map_dir in map_dirs:
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = out_dir / f"{map_dir.name}_control_point_curves.svg"

        map_name, tile_size, curve_count, components, written_path = _render_map(
            map_dir,
            out_path,
            samples=args.samples,
        )

        print(f"map={map_name}")
        print(f"tile_size={tile_size:.6f}")
        print(f"curves={curve_count}")
        print(f"connected_components={len(components)}")
        for i, comp in enumerate(components):
            print(f"component_{i}: curve_ids={comp}")
        print(f"svg={written_path}")


if __name__ == "__main__":
    main()
