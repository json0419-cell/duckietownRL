import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lane_utils as lu
from draw_control_point_curves import (
    _canvas_transform,
    _center_line_control_points_world,
    _curve_neighbors,
    _ground_coords,
    _load_map_tiles,
    _parse_tile_key,
    _sample_bezier,
    build_curves,
    _connected_components,
)


DEFAULT_OUT_ROOT = PROJECT_ROOT / "test" / "optimal_reward_routes"


def parse_args():
    parser = argparse.ArgumentParser(description="Draw a representative highest-reward route on a map.")
    parser.add_argument("--map", default="all", help="Map folder under maps/, or 'all' for every map.")
    parser.add_argument("--maps-dir", default=str(PROJECT_ROOT / "maps"))
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    parser.add_argument("--samples", type=int, default=60, help="Samples per Bezier curve.")
    parser.add_argument(
        "--seed-tile",
        default="0,1",
        help="Seed tile i,j used to choose the ego-lane connected component. Default 0,1 for loop.",
    )
    parser.add_argument(
        "--seed-lane-idx",
        type=int,
        default=1,
        help="Lane index inside the seed tile used to pick the route component. Default 1 for loop ego-lane.",
    )
    parser.add_argument("--annotate-every", type=int, default=25, help="Annotate every Nth sampled point.")
    return parser.parse_args()


def _bezier_tangent(cps, t: float):
    cps = np.asarray(cps, dtype=np.float32)
    p = 3 * ((1 - t) ** 2) * (cps[1, :] - cps[0, :])
    p += 6 * (1 - t) * t * (cps[2, :] - cps[1, :])
    p += 3 * (t**2) * (cps[3, :] - cps[2, :])
    norm = float(np.linalg.norm(p))
    if norm > 1e-6:
        p = p / norm
    return p


def _yaw_from_tangent(tangent):
    tangent = np.asarray(tangent, dtype=np.float32)
    if lu.LANE_UP_AXIS == "z":
        return float(math.atan2(float(tangent[1]), float(tangent[0])))
    return float(math.atan2(-float(tangent[2]), float(tangent[0])))


def _leaky_cosine(x: float) -> float:
    slope = 0.05
    if abs(x) < math.pi:
        return float(math.cos(x))
    return float(-1.0 - slope * (abs(x) - math.pi))


def _orientation_reward(lp_dist: float, lp_angle_deg: float) -> tuple[float, float]:
    max_lp_dist = 0.05
    max_dev_narrow = 10.0
    max_dev_wide = 50.0
    target_angle_at_edge = 45.0
    clipped_dist = float(np.clip(float(lp_dist) / max_lp_dist, -1.0, 1.0))
    target_angle_deg = -clipped_dist * target_angle_at_edge
    narrow = 0.5 + 0.5 * _leaky_cosine(math.pi * (target_angle_deg - float(lp_angle_deg)) / max_dev_narrow)
    wide = 0.5 + 0.5 * _leaky_cosine(math.pi * (target_angle_deg - float(lp_angle_deg)) / max_dev_wide)
    return float(0.5 * (narrow + wide)), float(target_angle_deg)


def _curve_component_map(curves, components):
    comp_by_curve = {}
    for comp_idx, comp in enumerate(components):
        for cid in comp:
            comp_by_curve[cid] = comp_idx
    return comp_by_curve


def _find_seed_curve(curves, tile_i: int, tile_j: int, lane_idx: int):
    suffix = f"tile_{tile_i}_{tile_j}"
    for curve in curves:
        if curve.tile_key.endswith(suffix) and curve.lane_idx == lane_idx:
            return curve
    return None


def _curve_sort_key(curve):
    _, i, j = _parse_tile_key(curve.tile_key)
    return (j, i, curve.lane_idx, curve.curve_id)


def _auto_select_component(curves, components, preferred_lane_idx: int) -> tuple[int, int, str]:
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


def _select_component(curves, components, tile_i: int, tile_j: int, lane_idx: int) -> tuple[int, int, str]:
    comp_by_curve = _curve_component_map(curves, components)
    seed_curve = _find_seed_curve(curves, tile_i, tile_j, lane_idx)
    if seed_curve is not None:
        return comp_by_curve[seed_curve.curve_id], seed_curve.curve_id, f"manual(tile=({tile_i},{tile_j}), lane_idx={lane_idx})"
    return _auto_select_component(curves, components, preferred_lane_idx=lane_idx)


def _order_component_curves(curves, component_curve_ids, seed_curve_id: int, eps: float = 1e-6):
    curve_by_id = {curve.curve_id: curve for curve in curves}
    remaining = set(component_curve_ids)
    ordered = []
    current_id = seed_curve_id
    while current_id in remaining:
        ordered.append(curve_by_id[current_id])
        remaining.remove(current_id)
        current = curve_by_id[current_id]
        next_id = None
        for candidate_id in list(remaining):
            candidate = curve_by_id[candidate_id]
            if np.linalg.norm(np.asarray(current.end) - np.asarray(candidate.start)) <= eps:
                next_id = candidate_id
                break
        if next_id is None:
            break
        current_id = next_id
    if remaining:
        raise RuntimeError(f"Could not order all curves in component, leftover={sorted(remaining)}")
    return ordered


def _sample_route(curves, samples_per_curve: int):
    rows = []
    for curve_idx, curve in enumerate(curves):
        ts = np.linspace(0.0, 1.0, samples_per_curve, dtype=np.float32)
        if curve_idx > 0:
            ts = ts[1:]
        for t in ts:
            p = lu._bezier_point(np.asarray(curve.cps, dtype=np.float32), float(t))
            tangent = _bezier_tangent(np.asarray(curve.cps, dtype=np.float32), float(t))
            yaw = _yaw_from_tangent(tangent)
            orientation_reward, target_angle_deg = _orientation_reward(0.0, 0.0)
            velocity_reward = 0.25
            collision_reward = 0.0
            final_reward = orientation_reward + velocity_reward + collision_reward
            rows.append(
                {
                    "curve_id": curve.curve_id,
                    "tile_key": curve.tile_key,
                    "lane_idx": curve.lane_idx,
                    "t": float(t),
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "yaw": float(yaw),
                    "dist": 0.0,
                    "dot": 1.0,
                    "angle_deg": 0.0,
                    "target_angle_deg": float(target_angle_deg),
                    "orientation_reward": float(orientation_reward),
                    "velocity_reward": float(velocity_reward),
                    "collision_reward": float(collision_reward),
                    "final_reward": float(final_reward),
                }
            )
    return rows


def _write_csv(rows, out_path: Path):
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _draw_background(ax, tiles, tile_size: float):
    min_gx = min(t.pose_x * tile_size for t in tiles.values())
    max_gx = max((t.pose_x + 1.0) * tile_size for t in tiles.values())
    min_gy = min(t.pose_y * tile_size for t in tiles.values())
    max_gy = max((t.pose_y + 1.0) * tile_size for t in tiles.values())
    width, height, margin = 1100, 900, 70

    for tile in sorted(tiles.values(), key=lambda t: (t.i, t.j)):
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
        ax.text(
            cx,
            cy,
            f"{tile.tile_type} ({tile.i},{tile.j})",
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
            zorder=1,
        )

    for tile in sorted(tiles.values(), key=lambda t: (t.i, t.j)):
        center_cps = _center_line_control_points_world(tile, tile_size)
        if center_cps is None:
            continue
        center_pts = _sample_bezier(center_cps, 40)
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
        ax.plot(
            [p[0] for p in center_xy],
            [p[1] for p in center_xy],
            color="#facc15",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )

    return min_gx, max_gx, min_gy, max_gy, width, height, margin


def _plot_route(rows, ordered_curves, curves, tiles, tile_size: float, out_path: Path, map_name: str, annotate_every: int):
    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)
    min_gx, max_gx, min_gy, max_gy, width, height, margin = _draw_background(ax, tiles, tile_size)

    def to_canvas(x, y):
        return _canvas_transform(x, y, min_gx, max_gx, min_gy, max_gy, width, height, margin)

    ordered_ids = {curve.curve_id for curve in ordered_curves}
    for curve in curves:
        pts = _sample_bezier(curve.cps, 50)
        pts_xy = [to_canvas(_ground_coords(p)[0], _ground_coords(p)[1]) for p in pts]
        color = "#c5c5c5"
        linewidth = 1.8
        zorder = 2
        if curve.curve_id in ordered_ids:
            color = "#ef4444"
            linewidth = 3.0
            zorder = 3
        ax.plot(
            [p[0] for p in pts_xy],
            [p[1] for p in pts_xy],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=zorder,
        )

    pts_xy = [to_canvas(row["x"], row["y"]) for row in rows]
    rewards = np.array([row["final_reward"] for row in rows], dtype=np.float32)
    scatter = ax.scatter(
        [p[0] for p in pts_xy],
        [p[1] for p in pts_xy],
        c=rewards,
        cmap="plasma",
        s=42,
        zorder=4,
    )
    ax.scatter(pts_xy[0][0], pts_xy[0][1], color="tab:green", s=90, zorder=5, label="route start")
    ax.scatter(pts_xy[-1][0], pts_xy[-1][1], color="tab:red", s=90, zorder=5, label="route end")

    offsets = [(8, 8), (8, -8), (-8, 8), (-8, -8)]
    for idx, (row, pt) in enumerate(zip(rows, pts_xy)):
        if annotate_every > 1 and idx % annotate_every != 0:
            continue
        dx, dy = offsets[idx % len(offsets)]
        label = (
            f"{idx:03d}\n"
            f"r={row['final_reward']:.2f}\n"
            f"yaw={row['yaw']:.2f}"
        )
        ax.annotate(
            label,
            xy=pt,
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.5},
            zorder=6,
        )

    ax.set_title(
        f"Representative Highest-Reward Route [{map_name}]\n"
        f"Canonical ego-lane centerline route, final_reward={rows[0]['final_reward']:.2f} per sample"
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="final_reward")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _iter_map_names(maps_dir: Path):
    return sorted(p.name for p in maps_dir.iterdir() if p.is_dir() and (p / "main.yaml").exists())


def _run_for_map(map_name: str, args, maps_dir: Path):
    map_dir = maps_dir / map_name
    if not map_dir.exists():
        raise FileNotFoundError(f"Map directory not found: {map_dir}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve() / map_name
        if args.out_dir is not None
        else DEFAULT_OUT_ROOT / map_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_i, seed_j = [int(part.strip()) for part in args.seed_tile.split(",")]
    tiles, tile_size, payload = _load_map_tiles(map_dir)
    curves = build_curves(tiles, payload)
    components = _connected_components(_curve_neighbors(curves))
    component_idx, seed_curve_id, seed_desc = _select_component(curves, components, seed_i, seed_j, args.seed_lane_idx)
    ordered_curves = _order_component_curves(curves, components[component_idx], seed_curve_id)
    rows = _sample_route(ordered_curves, samples_per_curve=max(2, int(args.samples)))

    fig_path = out_dir / f"{map_name}_optimal_reward_route.png"
    csv_path = out_dir / f"{map_name}_optimal_reward_route.csv"
    _plot_route(
        rows,
        ordered_curves,
        curves,
        tiles,
        tile_size,
        fig_path,
        map_name=map_name,
        annotate_every=max(1, int(args.annotate_every)),
    )
    _write_csv(rows, csv_path)

    print(f"map={map_name}")
    print(f"seed_selection={seed_desc}")
    print(f"requested_seed_tile=({seed_i},{seed_j}) seed_lane_idx={args.seed_lane_idx}")
    print(f"component_idx={component_idx}")
    print(f"curves_in_route={len(ordered_curves)}")
    print(f"samples={len(rows)}")
    print(f"final_reward_per_sample={rows[0]['final_reward']:.3f}")
    print(f"figure={fig_path}")
    print(f"csv={csv_path}")


def main():
    args = parse_args()
    maps_dir = Path(args.maps_dir).expanduser().resolve()
    map_names = _iter_map_names(maps_dir) if args.map == "all" else [args.map]
    for map_name in map_names:
        _run_for_map(map_name, args, maps_dir)


if __name__ == "__main__":
    main()
