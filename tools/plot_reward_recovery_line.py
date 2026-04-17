#!/usr/bin/env python3
"""
Visualize the reward-optimal recovery line for off-center starts on a map.

The right panel shows the theoretical reward landscape in lane coordinates
for the current reward settings. The left panel projects the corresponding
best-reward recovery lines back onto a specific map route.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import lane_utils as lu
from core.map_interpreter_patch import PatchedMapInterpreter
from wrappers.reward_wrappers import LaneFollowingRewardWrapper


DEFAULT_OUT_ROOT = PROJECT_ROOT / "tools" / "reward_recovery_lines"


Point = tuple[float, float, float]


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
    cps: tuple[Point, Point, Point, Point]

    @property
    def start(self) -> Point:
        return self.cps[0]

    @property
    def end(self) -> Point:
        return self.cps[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward-optimal recovery lines for off-center starts.")
    parser.add_argument("--map", type=str, default="_huge_C_floor", help="map folder under --maps-dir")
    parser.add_argument("--maps-dir", type=str, default=str(PROJECT_ROOT / "maps"))
    parser.add_argument("--seed-tile", type=str, default="0,1", help="seed tile i,j; auto fallback if invalid")
    parser.add_argument("--seed-lane-idx", type=int, default=1, help="preferred lane index for route selection")
    parser.add_argument("--samples-per-curve", type=int, default=80, help="sampling density along route curves")
    parser.add_argument(
        "--offsets",
        type=str,
        default="-0.04,-0.02,0.02,0.04",
        help="comma-separated initial lateral offsets in meters",
    )
    parser.add_argument("--integration-step", type=float, default=0.01, help="integration step in meters")
    parser.add_argument("--out", type=str, default=None, help="output PNG path")
    return parser.parse_args()


def _reward_defaults() -> dict[str, float]:
    params = inspect.signature(LaneFollowingRewardWrapper.__init__).parameters
    keys = (
        "max_lp_dist",
        "max_dev_from_target_angle_deg_narrow",
        "max_dev_from_target_angle_deg_wide",
        "target_angle_deg_at_edge",
        "orientation_scale",
        "velocity_reward_scale",
    )
    return {key: float(params[key].default) for key in keys}


def _leaky_cosine(x: np.ndarray | float) -> np.ndarray | float:
    slope = 0.05
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.where(
        np.abs(x_arr) < math.pi,
        np.cos(x_arr),
        -1.0 - slope * (np.abs(x_arr) - math.pi),
    )
    if np.isscalar(x):
        return float(out)
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root is not a dict: {path}")
    return data


def _parse_tile_key(key: str) -> tuple[str, int, int]:
    prefix, suffix = key.split("/tile_", 1)
    i_str, j_str = suffix.split("_", 1)
    return prefix, int(i_str), int(j_str)


def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _ground_coords(point: Point) -> tuple[float, float]:
    if lu.LANE_UP_AXIS == "z":
        return float(point[0]), float(point[1])
    return float(point[0]), float(point[2])


def _sample_bezier(cps: tuple[Point, Point, Point, Point], n: int) -> list[Point]:
    if n < 2:
        n = 2
    return [tuple(float(v) for v in lu._bezier_point(np.asarray(cps, dtype=np.float32), i / float(n - 1))) for i in range(n)]


def _load_map_tiles(map_dir: Path) -> tuple[dict[str, Tile], float, dict[str, dict]]:
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

    out: dict[str, Tile] = {}
    for key, tile_desc in tiles_yaml.items():
        if not isinstance(key, str) or not isinstance(tile_desc, dict):
            continue
        frame = frames_yaml.get(key)
        if not isinstance(frame, dict):
            continue
        pose = frame.get("pose")
        if not isinstance(pose, dict):
            continue
        map_name, i, j = _parse_tile_key(key)
        out[key] = Tile(
            key=key,
            map_name=map_name,
            i=i,
            j=j,
            tile_type=str(tile_desc.get("type", "")).strip().lower(),
            pose_x=float(pose.get("x", i)),
            pose_y=float(pose.get("y", j)),
            yaw=float(pose.get("yaw", 0.0)),
        )

    payload = {
        "frames": {"data": frames_yaml},
        "tiles": {"data": tiles_yaml},
        "tile_info": {"data": tile_maps_yaml},
    }
    return out, tile_size, payload


def build_curves(tiles: dict[str, Tile], map_payload: dict[str, dict]) -> list[Curve]:
    map_int = PatchedMapInterpreter(map=map_payload)
    curves: list[Curve] = []
    curve_id = 0
    for key in sorted(tiles.keys()):
        tile = tiles[key]
        map_tile = map_int.get_tile(tile.i, tile.j)
        if map_tile is None or not map_tile.get("drivable", False):
            continue
        map_curves = map_tile.get("curves")
        lane_curves = lu._apply_curve_offset(None, map_curves) or []
        for lane_idx, cps in enumerate(lane_curves):
            cps_arr = np.asarray(cps, dtype=np.float32)
            if cps_arr.shape[0] < 4:
                continue
            world_cps = [tuple(float(v) for v in p) for p in cps_arr[:4]]
            curves.append(
                Curve(
                    curve_id=curve_id,
                    tile_key=tile.key,
                    lane_idx=lane_idx,
                    cps=(world_cps[0], world_cps[1], world_cps[2], world_cps[3]),
                )
            )
            curve_id += 1
    return curves


def _curve_neighbors(curves: list[Curve], eps: float = 1e-6) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = {curve.curve_id: set() for curve in curves}
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


def _curve_component_map(curves: list[Curve], components: list[list[int]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for comp_idx, comp in enumerate(components):
        for curve_id in comp:
            out[curve_id] = comp_idx
    return out


def _find_seed_curve(curves: list[Curve], tile_i: int, tile_j: int, lane_idx: int) -> Curve | None:
    suffix = f"tile_{tile_i}_{tile_j}"
    for curve in curves:
        if curve.tile_key.endswith(suffix) and curve.lane_idx == lane_idx:
            return curve
    return None


def _curve_sort_key(curve: Curve) -> tuple[int, int, int, int]:
    _, tile_i, tile_j = _parse_tile_key(curve.tile_key)
    return (tile_j, tile_i, curve.lane_idx, curve.curve_id)


def _auto_select_component(curves: list[Curve], components: list[list[int]], preferred_lane_idx: int) -> tuple[int, int, str]:
    comp_by_curve = _curve_component_map(curves, components)
    candidates = sorted(
        curves,
        key=lambda curve: (
            0 if curve.lane_idx == preferred_lane_idx else 1,
            -len(components[comp_by_curve[curve.curve_id]]),
            *_curve_sort_key(curve),
        ),
    )
    if not candidates:
        raise RuntimeError("No curves available for automatic component selection")
    seed_curve = candidates[0]
    _, tile_i, tile_j = _parse_tile_key(seed_curve.tile_key)
    return (
        comp_by_curve[seed_curve.curve_id],
        seed_curve.curve_id,
        f"auto(tile=({tile_i},{tile_j}), lane_idx={seed_curve.lane_idx})",
    )


def _select_component(curves: list[Curve], components: list[list[int]], tile_i: int, tile_j: int, lane_idx: int) -> tuple[int, int, str]:
    comp_by_curve = _curve_component_map(curves, components)
    seed_curve = _find_seed_curve(curves, tile_i, tile_j, lane_idx)
    if seed_curve is not None:
        return comp_by_curve[seed_curve.curve_id], seed_curve.curve_id, f"manual(tile=({tile_i},{tile_j}), lane_idx={lane_idx})"
    return _auto_select_component(curves, components, preferred_lane_idx=lane_idx)


def _order_component_curves(curves: list[Curve], component_curve_ids: list[int], seed_curve_id: int, eps: float = 1e-6) -> list[Curve]:
    curve_by_id = {curve.curve_id: curve for curve in curves}
    remaining = set(component_curve_ids)
    ordered: list[Curve] = []
    current_id = seed_curve_id
    while current_id in remaining:
        ordered.append(curve_by_id[current_id])
        remaining.remove(current_id)
        current = curve_by_id[current_id]
        next_id = None
        for candidate_id in list(remaining):
            candidate = curve_by_id[candidate_id]
            if _dist(current.end, candidate.start) <= eps:
                next_id = candidate_id
                break
        if next_id is None:
            break
        current_id = next_id
    if remaining:
        raise RuntimeError(f"Could not order all curves in component, leftover={sorted(remaining)}")
    return ordered


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
) -> tuple[float, float]:
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    span_x = max(max_gx - min_gx, 1e-9)
    span_y = max(max_gy - min_gy, 1e-9)
    scale = min(usable_w / span_x, usable_h / span_y)
    x0 = margin + (gx - min_gx) * scale
    y0 = height - margin - (gy - min_gy) * scale
    return x0, y0


def _draw_background(ax, tiles: dict[str, Tile], tile_size: float):
    min_gx = min(tile.pose_x * tile_size for tile in tiles.values())
    max_gx = max((tile.pose_x + 1.0) * tile_size for tile in tiles.values())
    min_gy = min(tile.pose_y * tile_size for tile in tiles.values())
    max_gy = max((tile.pose_y + 1.0) * tile_size for tile in tiles.values())
    width, height, margin = 1100, 900, 70

    for tile in sorted(tiles.values(), key=lambda item: (item.i, item.j)):
        p1 = _canvas_transform(tile.pose_x * tile_size, tile.pose_y * tile_size, min_gx, max_gx, min_gy, max_gy, width, height, margin)
        p2 = _canvas_transform((tile.pose_x + 1.0) * tile_size, (tile.pose_y + 1.0) * tile_size, min_gx, max_gx, min_gy, max_gy, width, height, margin)
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="#f8f8f8",
                edgecolor="#d0d0d0",
                linewidth=1.4,
                zorder=0,
            )
        )
        cx, cy = _canvas_transform((tile.pose_x + 0.5) * tile_size, (tile.pose_y + 0.5) * tile_size, min_gx, max_gx, min_gy, max_gy, width, height, margin)
        ax.text(cx, cy, f"{tile.tile_type} ({tile.i},{tile.j})", ha="center", va="center", fontsize=7, color="#555555", zorder=1)

    return min_gx, max_gx, min_gy, max_gy, width, height, margin


def _target_angle_deg(lp_dist: np.ndarray | float, cfg: dict[str, float]) -> np.ndarray | float:
    clipped = np.clip(np.asarray(lp_dist, dtype=np.float64) / cfg["max_lp_dist"], -1.0, 1.0)
    out = -clipped * cfg["target_angle_deg_at_edge"]
    if np.isscalar(lp_dist):
        return float(out)
    return out


def _final_reward(lp_dist: np.ndarray, lp_angle_deg: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    target_angle_deg = _target_angle_deg(lp_dist, cfg)
    narrow = 0.5 + 0.5 * _leaky_cosine(
        math.pi * (target_angle_deg - lp_angle_deg) / cfg["max_dev_from_target_angle_deg_narrow"]
    )
    wide = 0.5 + 0.5 * _leaky_cosine(
        math.pi * (target_angle_deg - lp_angle_deg) / cfg["max_dev_from_target_angle_deg_wide"]
    )
    orientation_reward = cfg["orientation_scale"] * (narrow + wide)
    return orientation_reward + cfg["velocity_reward_scale"]


def _connected_components(neighbors: dict[int, set[int]]) -> list[list[int]]:
    components: list[list[int]] = []
    seen: set[int] = set()
    for node in sorted(neighbors):
        if node in seen:
            continue
        stack = [node]
        comp: list[int] = []
        seen.add(node)
        while stack:
            current = stack.pop()
            comp.append(current)
            for nxt in neighbors[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comp.sort()
        components.append(comp)
    return components


def _route_samples(ordered_curves, samples_per_curve: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy_points: list[tuple[float, float]] = []
    for idx, curve in enumerate(ordered_curves):
        pts = _sample_bezier(curve.cps, max(2, int(samples_per_curve)))
        if idx > 0:
            pts = pts[1:]
        xy_points.extend(_ground_coords(p) for p in pts)

    xy = np.asarray(xy_points, dtype=np.float64)
    deltas = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = np.diff(s) > 1e-9
    xy = xy[keep]
    s = s[keep]

    tx = np.gradient(xy[:, 0], s, edge_order=1)
    ty = np.gradient(xy[:, 1], s, edge_order=1)
    tangent = np.column_stack((tx, ty))
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-9] = 1.0
    tangent = tangent / tangent_norm

    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return s, xy, tangent, normal


def _interp_vec(s: np.ndarray, values: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = np.interp(q, s, values[:, 0])
    y = np.interp(q, s, values[:, 1])
    return np.column_stack((x, y))


def _simulate_recovery_line(
    s_route: np.ndarray,
    center_xy: np.ndarray,
    normal_xy: np.ndarray,
    initial_offset: float,
    cfg: dict[str, float],
    integration_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_length = float(s_route[-1])
    s_values = [0.0]
    d_values = [float(initial_offset)]
    a_values = [float(_target_angle_deg(initial_offset, cfg))]
    rewards = [float(_final_reward(np.array([initial_offset]), np.array([a_values[-1]]), cfg)[0])]

    while s_values[-1] < total_length:
        d_curr = d_values[-1]
        angle_deg = float(_target_angle_deg(d_curr, cfg))
        angle_rad = math.radians(angle_deg)
        ds_center = integration_step * max(math.cos(angle_rad), 1e-4)
        s_next = min(total_length, s_values[-1] + ds_center)
        d_next = d_curr + integration_step * math.sin(angle_rad)

        s_values.append(s_next)
        d_values.append(d_next)
        a_values.append(float(_target_angle_deg(d_next, cfg)))
        rewards.append(float(_final_reward(np.array([d_next]), np.array([a_values[-1]]), cfg)[0]))

        if s_next >= total_length:
            break

    s_arr = np.asarray(s_values, dtype=np.float64)
    d_arr = np.asarray(d_values, dtype=np.float64)
    center_interp = _interp_vec(s_route, center_xy, s_arr)
    normal_interp = _interp_vec(s_route, normal_xy, s_arr)
    xy = center_interp + d_arr[:, None] * normal_interp
    return xy, d_arr, np.asarray(a_values, dtype=np.float64), np.asarray(rewards, dtype=np.float64)


def _plot_map_panel(ax, tiles, tile_size, route_xy, recovery_lines, map_name: str):
    min_gx, max_gx, min_gy, max_gy, width, height, margin = _draw_background(ax, tiles, tile_size)

    def to_canvas(points_xy: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _canvas_transform(float(x), float(y), min_gx, max_gx, min_gy, max_gy, width, height, margin)
                for x, y in points_xy
            ],
            dtype=np.float64,
        )

    route_canvas = to_canvas(route_xy)
    ax.plot(route_canvas[:, 0], route_canvas[:, 1], color="#facc15", linewidth=3.5, label="lane center", zorder=3)

    colors = ["#2563eb", "#0891b2", "#dc2626", "#ea580c", "#7c3aed", "#15803d"]
    for idx, (offset, xy, _, _, _) in enumerate(recovery_lines):
        canvas_xy = to_canvas(xy)
        label = f"start offset {offset * 100:.0f} cm"
        ax.plot(canvas_xy[:, 0], canvas_xy[:, 1], color=colors[idx % len(colors)], linewidth=2.4, label=label, zorder=4)
        ax.scatter(canvas_xy[0, 0], canvas_xy[0, 1], color=colors[idx % len(colors)], s=36, zorder=5)

    ax.set_title(f"Reward-Optimal Recovery Lines on {map_name}")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", fontsize=8)


def _plot_reward_panel(ax, cfg: dict[str, float], offsets: list[float]):
    dist_grid = np.linspace(-cfg["max_lp_dist"], cfg["max_lp_dist"], 241, dtype=np.float64)
    angle_grid = np.linspace(-45.0, 45.0, 281, dtype=np.float64)
    dist_mesh, angle_mesh = np.meshgrid(dist_grid, angle_grid)
    reward_mesh = _final_reward(dist_mesh, angle_mesh, cfg)

    image = ax.imshow(
        reward_mesh,
        origin="lower",
        aspect="auto",
        extent=[dist_grid[0] * 100.0, dist_grid[-1] * 100.0, angle_grid[0], angle_grid[-1]],
        cmap="magma",
    )

    ridge_angle = _target_angle_deg(dist_grid, cfg)
    ax.plot(dist_grid * 100.0, ridge_angle, color="white", linewidth=2.4, label="best-reward ridge")
    for offset in offsets:
        ax.scatter(offset * 100.0, float(_target_angle_deg(offset, cfg)), color="cyan", s=34, edgecolors="black", linewidths=0.5)

    ax.set_title("Reward Landscape in Lane Coordinates")
    ax.set_xlabel("Lateral offset from lane center (cm)")
    ax.set_ylabel("Heading error in lane frame (deg)")
    ax.legend(loc="upper right", fontsize=8)
    return image


def main() -> None:
    args = parse_args()
    cfg = _reward_defaults()

    maps_dir = Path(args.maps_dir).expanduser().resolve()
    map_dir = maps_dir / args.map
    if not map_dir.exists():
        raise FileNotFoundError(f"Map directory not found: {map_dir}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else (DEFAULT_OUT_ROOT / f"{args.map}_best_reward_recovery.png").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_i, seed_j = [int(part.strip()) for part in args.seed_tile.split(",")]
    initial_offsets = [float(v.strip()) for v in args.offsets.split(",") if v.strip()]

    tiles, tile_size, payload = _load_map_tiles(map_dir)
    curves = build_curves(tiles, payload)
    components = _connected_components(_curve_neighbors(curves))
    component_idx, seed_curve_id, seed_desc = _select_component(curves, components, seed_i, seed_j, args.seed_lane_idx)
    ordered_curves = _order_component_curves(curves, components[component_idx], seed_curve_id)

    s_route, route_xy, _, normal_xy = _route_samples(ordered_curves, args.samples_per_curve)
    recovery_lines = []
    for offset in initial_offsets:
        xy, d_values, a_values, rewards = _simulate_recovery_line(
            s_route,
            route_xy,
            normal_xy,
            offset,
            cfg,
            max(1e-3, float(args.integration_step)),
        )
        recovery_lines.append((offset, xy, d_values, a_values, rewards))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    _plot_map_panel(axes[0], tiles, tile_size, route_xy, recovery_lines, args.map)
    image = _plot_reward_panel(axes[1], cfg, initial_offsets)
    fig.colorbar(image, ax=axes[1], label="theoretical final_reward")
    fig.suptitle(
        f"Best-Reward Recovery Visualization [{args.map}]\n"
        f"seed={seed_desc}, narrow={cfg['max_dev_from_target_angle_deg_narrow']:.0f}, "
        f"wide={cfg['max_dev_from_target_angle_deg_wide']:.0f}, "
        f"target_angle_at_edge={cfg['target_angle_deg_at_edge']:.0f}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"map={args.map}")
    print(f"seed_selection={seed_desc}")
    print(f"component_idx={component_idx}")
    print(f"route_points={len(route_xy)}")
    print(f"offsets={initial_offsets}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
