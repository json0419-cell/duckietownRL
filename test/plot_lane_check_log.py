import argparse
import ast
import csv
import math
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT_ROOT = PROJECT_ROOT / "test" / "lane_check_plots"
DEFAULT_TILE_SIZE = 0.585

from draw_control_point_curves import (  # noqa: E402
    _canvas_transform,
    _center_line_control_points_world,
    _ground_coords,
    _load_map_tiles,
    _sample_bezier,
)


HEADER_RE = re.compile(
    r"^\[(?P<step>\d+)\]\s+pose=\((?P<x>[-+0-9.eE]+),(?P<y>[-+0-9.eE]+),(?P<z>[-+0-9.eE]+)\)\s+"
    r"yaw=\s*(?P<yaw>[-+0-9.eE]+)\s+tile=\((?P<tile_i>-?\d+),\s*(?P<tile_j>-?\d+)\)\s+kind=(?P<kind>\S+)"
)
ACTION_RE = re.compile(
    r"^\s*action_reward:\s+heading=(?P<heading>\[[^\]]*\])\s+wheels=(?P<wheels>\[[^\]]*\])\s+"
    r"base_reward=\s*(?P<base_reward>[-+0-9.eE]+)\s+orientation=\s*(?P<orientation>[-+0-9.eE]+)\s+"
    r"velocity=\s*(?P<velocity>[-+0-9.eE]+)\s+"
    r"final_reward=\s*(?P<final_reward>[-+0-9.eE]+)\s+total=\s*(?P<total>[-+0-9.eE]+)"
)
LANE_RE = re.compile(
    r"^\s*lane_terms:\s+dist=\s*(?P<dist>[-+0-9.eE]+)\s+dot=\s*(?P<dot>[-+0-9.eE]+)\s+"
    r"angle_deg=\s*(?P<angle_deg>[-+0-9.eE]+)\s+target_angle_deg=\s*(?P<target_angle_deg>[-+0-9.eE]+)\s+"
    r"speed=\s*(?P<speed>[-+0-9.eE]+)"
)
STATUS_RE = re.compile(
    r"^\s*status:\s+terminated=(?P<terminated>True|False)\s+truncated=(?P<truncated>True|False)\s+"
    r"terminated_from_env=(?P<terminated_from_env>True|False)\s+pose_valid=(?P<pose_valid>True|False)\s+"
    r"invalid_points=(?P<invalid_points>.*?)\s+first_invalid=(?P<first_invalid>.*?)\s+reasons=(?P<reasons>.*)$"
)


def _parse_bool(value: str) -> bool:
    return value == "True"


def _parse_float_list(text: str):
    text = text.strip()
    if not text or text == "[]":
        return []
    values = ast.literal_eval(text)
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(v) for v in values]


def _parse_literal(text: str):
    text = text.strip()
    if text == "":
        return None
    if text == "None":
        return None
    return ast.literal_eval(text)


def parse_log(log_path: Path):
    rows = []
    current = None
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            match = HEADER_RE.match(line)
            if match:
                if current is not None:
                    rows.append(current)
                current = {
                    "step": int(match.group("step")),
                    "x": float(match.group("x")),
                    "y": float(match.group("y")),
                    "z": float(match.group("z")),
                    "yaw": float(match.group("yaw")),
                    "tile_i": int(match.group("tile_i")),
                    "tile_j": int(match.group("tile_j")),
                    "kind": match.group("kind"),
                }
                continue

            if current is None:
                continue

            match = ACTION_RE.match(line)
            if match:
                current.update(
                    {
                        "heading": _parse_float_list(match.group("heading")),
                        "wheels": _parse_float_list(match.group("wheels")),
                        "base_reward": float(match.group("base_reward")),
                        "orientation_reward": float(match.group("orientation")),
                        "velocity_reward": float(match.group("velocity")),
                        "final_reward": float(match.group("final_reward")),
                        "total_reward": float(match.group("total")),
                    }
                )
                continue

            match = LANE_RE.match(line)
            if match:
                current.update(
                    {
                        "dist": float(match.group("dist")),
                        "dot": float(match.group("dot")),
                        "angle_deg": float(match.group("angle_deg")),
                        "target_angle_deg": float(match.group("target_angle_deg")),
                        "speed": float(match.group("speed")),
                    }
                )
                continue

            match = STATUS_RE.match(line)
            if match:
                current.update(
                    {
                        "terminated": _parse_bool(match.group("terminated")),
                        "truncated": _parse_bool(match.group("truncated")),
                        "terminated_from_env": _parse_bool(match.group("terminated_from_env")),
                        "pose_valid": _parse_bool(match.group("pose_valid")),
                        "invalid_points": _parse_literal(match.group("invalid_points")),
                        "first_invalid": _parse_literal(match.group("first_invalid")),
                        "reasons": _parse_literal(match.group("reasons")),
                    }
                )
                continue

        if current is not None:
            rows.append(current)
    return rows


def add_checks(rows, max_lp_dist: float, target_angle_deg_at_edge: float):
    for row in rows:
        clipped_dist = float(np.clip(row["dist"] / max_lp_dist, -1.0, 1.0))
        row["expected_target_angle_deg"] = -clipped_dist * target_angle_deg_at_edge
        row["target_angle_error"] = row["target_angle_deg"] - row["expected_target_angle_deg"]
        row["expected_final_reward"] = row["orientation_reward"] + row["velocity_reward"]
        row["final_reward_error"] = row["final_reward"] - row["expected_final_reward"]


def write_csv(rows, csv_path: Path):
    fieldnames = [
        "step",
        "x",
        "y",
        "z",
        "yaw",
        "tile_i",
        "tile_j",
        "kind",
        "heading",
        "wheels",
        "base_reward",
        "orientation_reward",
        "velocity_reward",
        "final_reward",
        "total_reward",
        "dist",
        "dot",
        "angle_deg",
        "target_angle_deg",
        "speed",
        "terminated",
        "truncated",
        "terminated_from_env",
        "pose_valid",
        "invalid_points",
        "first_invalid",
        "reasons",
        "expected_target_angle_deg",
        "target_angle_error",
        "expected_final_reward",
        "final_reward_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["heading"] = ",".join(f"{v:.6f}" for v in out["heading"])
            out["wheels"] = ",".join(f"{v:.6f}" for v in out["wheels"])
            writer.writerow(out)


def plot_rows(rows, out_path: Path):
    steps = np.array([row["step"] for row in rows], dtype=np.int32)
    xs = np.array([row["x"] for row in rows], dtype=np.float32)
    ys = np.array([row["y"] for row in rows], dtype=np.float32)
    base_reward = np.array([row["base_reward"] for row in rows], dtype=np.float32)
    orientation = np.array([row["orientation_reward"] for row in rows], dtype=np.float32)
    velocity = np.array([row["velocity_reward"] for row in rows], dtype=np.float32)
    final_reward = np.array([row["final_reward"] for row in rows], dtype=np.float32)
    dist = np.array([row["dist"] for row in rows], dtype=np.float32)
    dot = np.array([row["dot"] for row in rows], dtype=np.float32)
    angle_deg = np.array([row["angle_deg"] for row in rows], dtype=np.float32)
    target_angle_deg = np.array([row["target_angle_deg"] for row in rows], dtype=np.float32)
    speed = np.array([row["speed"] for row in rows], dtype=np.float32)
    target_angle_error = np.array([row["target_angle_error"] for row in rows], dtype=np.float32)
    final_reward_error = np.array([row["final_reward_error"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(xs, ys, color="0.75", linewidth=1.0, zorder=1)
    scatter = ax.scatter(xs, ys, c=final_reward, cmap="viridis", s=30, zorder=2)
    ax.scatter(xs[0], ys[0], color="tab:green", s=70, label="start", zorder=3)
    ax.scatter(xs[-1], ys[-1], color="tab:red", s=70, label="end", zorder=3)
    ax.set_title("Trajectory Colored by Final Reward")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="final_reward")

    ax = axes[0, 1]
    ax.plot(steps, base_reward, label="base", linewidth=1.2)
    ax.plot(steps, orientation, label="orientation", linewidth=1.2)
    ax.plot(steps, velocity, label="velocity", linewidth=1.2)
    ax.plot(steps, final_reward, label="final", linewidth=1.8)
    ax.set_title("Reward Components")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(steps, dist, label="dist", linewidth=1.4)
    ax.plot(steps, dot, label="dot", linewidth=1.4)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_title("Lane Position Terms")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(steps, angle_deg, label="angle_deg", linewidth=1.4)
    ax.plot(steps, target_angle_deg, label="target_angle_deg", linewidth=1.4)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_title("Angle vs Target Angle")
    ax.set_xlabel("step")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2, 0]
    ax.plot(steps, speed, label="speed", linewidth=1.4)
    ax.set_title("Estimated Speed")
    ax.set_xlabel("step")
    ax.set_ylabel("m/s")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2, 1]
    ax.plot(steps, final_reward_error, label="final_error", linewidth=1.4)
    ax.plot(steps, target_angle_error, label="target_angle_error", linewidth=1.4)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_title("Formula Consistency Check")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("lane_check_heading Log Summary", fontsize=15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_annotated_map(rows, out_path: Path, tile_size: float, annotate_every: int):
    xs = np.array([row["x"] for row in rows], dtype=np.float32)
    ys = np.array([row["y"] for row in rows], dtype=np.float32)
    final_reward = np.array([row["final_reward"] for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)

    tile_keys = sorted({(row["tile_i"], row["tile_j"], row["kind"]) for row in rows})
    if tile_keys:
        min_i = min(t[0] for t in tile_keys)
        max_i = max(t[0] for t in tile_keys)
        min_j = min(t[1] for t in tile_keys)
        max_j = max(t[1] for t in tile_keys)
    else:
        min_i = int(math.floor(float(np.min(xs)) / tile_size))
        max_i = int(math.floor(float(np.max(xs)) / tile_size))
        min_j = int(math.floor(float(np.min(ys)) / tile_size))
        max_j = int(math.floor(float(np.max(ys)) / tile_size))

    for i in range(min_i, max_i + 2):
        ax.axvline(i * tile_size, color="0.88", linewidth=1.0, zorder=0)
    for j in range(min_j, max_j + 2):
        ax.axhline(j * tile_size, color="0.88", linewidth=1.0, zorder=0)

    for tile_i, tile_j, kind in tile_keys:
        cx = (tile_i + 0.5) * tile_size
        cy = (tile_j + 0.5) * tile_size
        ax.text(
            cx,
            cy,
            f"({tile_i},{tile_j})\n{kind}",
            ha="center",
            va="center",
            fontsize=7,
            color="0.55",
            zorder=0,
        )

    ax.plot(xs, ys, color="0.65", linewidth=1.0, zorder=1)
    scatter = ax.scatter(xs, ys, c=final_reward, cmap="viridis", s=40, zorder=2)
    ax.scatter(xs[0], ys[0], color="tab:green", s=90, label="start", zorder=3)
    ax.scatter(xs[-1], ys[-1], color="tab:red", s=90, label="end", zorder=3)

    offsets = [
        (8, 8),
        (8, -8),
        (-8, 8),
        (-8, -8),
    ]
    for idx, row in enumerate(rows):
        if annotate_every > 1 and idx % annotate_every != 0:
            continue
        dx, dy = offsets[idx % len(offsets)]
        label = (
            f"{row['step']:05d}\n"
            f"r={row['final_reward']:.3f}\n"
            f"o={row['orientation_reward']:.3f}\n"
            f"v={row['velocity_reward']:.3f}\n"
            f"d={row['dist']:.3f}\n"
            f"dot={row['dot']:.3f}\n"
            f"a={row['angle_deg']:.1f}\n"
            f"t={row['target_angle_deg']:.1f}"
        )
        ax.annotate(
            label,
            xy=(row["x"], row["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.6},
            zorder=4,
        )

    ax.set_title("Annotated Lane Check Map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(False)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="final_reward")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _draw_simple_tile_grid(ax, rows, tile_size: float):
    xs = np.array([row["x"] for row in rows], dtype=np.float32)
    ys = np.array([row["y"] for row in rows], dtype=np.float32)
    tile_keys = sorted({(row["tile_i"], row["tile_j"], row["kind"]) for row in rows})
    if tile_keys:
        min_i = min(t[0] for t in tile_keys)
        max_i = max(t[0] for t in tile_keys)
        min_j = min(t[1] for t in tile_keys)
        max_j = max(t[1] for t in tile_keys)
    else:
        min_i = int(math.floor(float(np.min(xs)) / tile_size))
        max_i = int(math.floor(float(np.max(xs)) / tile_size))
        min_j = int(math.floor(float(np.min(ys)) / tile_size))
        max_j = int(math.floor(float(np.max(ys)) / tile_size))

    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            rect = Rectangle(
                (i * tile_size, j * tile_size),
                tile_size,
                tile_size,
                facecolor="#f8f8f8",
                edgecolor="#d0d0d0",
                linewidth=1.2,
                zorder=0,
            )
            ax.add_patch(rect)

    for tile_i, tile_j, kind in tile_keys:
        cx = (tile_i + 0.5) * tile_size
        cy = (tile_j + 0.5) * tile_size
        ax.text(
            cx,
            cy,
            f"({tile_i},{tile_j})\n{kind}",
            ha="center",
            va="center",
            fontsize=7,
            color="0.55",
            zorder=1,
        )


def _draw_map_background(ax, map_dir: Path):
    tiles, tile_size, _ = _load_map_tiles(map_dir)
    if not tiles:
        raise RuntimeError(f"No tiles loaded from {map_dir}")

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

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return tiles, tile_size, min_gx, max_gx, min_gy, max_gy, width, height, margin


def plot_annotated_map_with_background(
    rows,
    out_path: Path,
    annotate_every: int,
    map_dir: Path | None,
    tile_size: float,
):
    final_reward = np.array([row["final_reward"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)

    if map_dir is not None:
        (
            _tiles,
            _tile_size_loaded,
            min_gx,
            max_gx,
            min_gy,
            max_gy,
            width,
            height,
            margin,
        ) = _draw_map_background(ax, map_dir)

        def to_canvas(x, y):
            return _canvas_transform(x, y, min_gx, max_gx, min_gy, max_gy, width, height, margin)

        points_xy = [to_canvas(float(row["x"]), float(row["y"])) for row in rows]
        px = np.array([p[0] for p in points_xy], dtype=np.float32)
        py = np.array([p[1] for p in points_xy], dtype=np.float32)
        ax.set_title(f"Annotated Lane Check Map [{map_dir.name}]")
    else:
        _draw_simple_tile_grid(ax, rows, tile_size)
        px = np.array([row["x"] for row in rows], dtype=np.float32)
        py = np.array([row["y"] for row in rows], dtype=np.float32)
        ax.set_title("Annotated Lane Check Map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    ax.plot(px, py, color="0.55", linewidth=1.1, zorder=2)
    scatter = ax.scatter(px, py, c=final_reward, cmap="viridis", s=42, zorder=3)
    ax.scatter(px[0], py[0], color="tab:green", s=90, label="start", zorder=4)
    ax.scatter(px[-1], py[-1], color="tab:red", s=90, label="end", zorder=4)

    offsets = [(8, 8), (8, -8), (-8, 8), (-8, -8)]
    for idx, row in enumerate(rows):
        if annotate_every > 1 and idx % annotate_every != 0:
            continue
        dx, dy = offsets[idx % len(offsets)]
        label = (
            f"{row['step']:05d}\n"
            f"r={row['final_reward']:.3f}\n"
            f"o={row['orientation_reward']:.3f}\n"
            f"v={row['velocity_reward']:.3f}\n"
            f"d={row['dist']:.3f}\n"
            f"dot={row['dot']:.3f}\n"
            f"a={row['angle_deg']:.1f}\n"
            f"t={row['target_angle_deg']:.1f}"
        )
        ax.annotate(
            label,
            xy=(px[idx], py[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.6},
            zorder=5,
        )

    ax.grid(False)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="final_reward")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def best_reward_segment(rows):
    if not rows:
        raise ValueError("No rows for best_reward_segment")

    best_sum = -float("inf")
    best_start = 0
    best_end = 0

    cur_sum = 0.0
    cur_start = 0
    for idx, row in enumerate(rows):
        reward = float(row["final_reward"])
        if cur_sum <= 0.0:
            cur_sum = reward
            cur_start = idx
        else:
            cur_sum += reward

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start = cur_start
            best_end = idx

    segment = rows[best_start : best_end + 1]
    mean_reward = best_sum / float(len(segment))
    return {
        "start_idx": best_start,
        "end_idx": best_end,
        "sum_reward": float(best_sum),
        "mean_reward": float(mean_reward),
        "segment": segment,
    }


def plot_best_reward_route(
    rows,
    out_path: Path,
    map_dir: Path | None,
    tile_size: float,
):
    best = best_reward_segment(rows)
    segment = best["segment"]

    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)
    if map_dir is not None:
        (
            _tiles,
            _tile_size_loaded,
            min_gx,
            max_gx,
            min_gy,
            max_gy,
            width,
            height,
            margin,
        ) = _draw_map_background(ax, map_dir)

        def to_canvas(x, y):
            return _canvas_transform(x, y, min_gx, max_gx, min_gy, max_gy, width, height, margin)

        all_xy = [to_canvas(float(row["x"]), float(row["y"])) for row in rows]
        seg_xy = [to_canvas(float(row["x"]), float(row["y"])) for row in segment]
        title_suffix = f"[{map_dir.name}]"
    else:
        _draw_simple_tile_grid(ax, rows, tile_size)
        all_xy = [(float(row["x"]), float(row["y"])) for row in rows]
        seg_xy = [(float(row["x"]), float(row["y"])) for row in segment]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        title_suffix = ""

    ax.plot(
        [p[0] for p in all_xy],
        [p[1] for p in all_xy],
        color="#c7c7c7",
        linewidth=1.2,
        zorder=2,
        label="full trajectory",
    )
    seg_rewards = np.array([row["final_reward"] for row in segment], dtype=np.float32)
    scatter = ax.scatter(
        [p[0] for p in seg_xy],
        [p[1] for p in seg_xy],
        c=seg_rewards,
        cmap="plasma",
        s=70,
        zorder=4,
    )
    ax.plot(
        [p[0] for p in seg_xy],
        [p[1] for p in seg_xy],
        color="#dc2626",
        linewidth=3.0,
        zorder=3,
        label="best reward segment",
    )
    ax.scatter(seg_xy[0][0], seg_xy[0][1], color="tab:green", s=100, zorder=5, label="segment start")
    ax.scatter(seg_xy[-1][0], seg_xy[-1][1], color="tab:red", s=100, zorder=5, label="segment end")

    offsets = [(8, 8), (8, -8), (-8, 8), (-8, -8)]
    for local_idx, (row, pt) in enumerate(zip(segment, seg_xy)):
        dx, dy = offsets[local_idx % len(offsets)]
        label = (
            f"{row['step']:05d}\n"
            f"r={row['final_reward']:.3f}\n"
            f"d={row['dist']:.3f}\n"
            f"dot={row['dot']:.3f}\n"
            f"a={row['angle_deg']:.1f}"
        )
        ax.annotate(
            label,
            xy=pt,
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.6},
            zorder=6,
        )

    start_step = segment[0]["step"]
    end_step = segment[-1]["step"]
    ax.set_title(
        "Best Reward Route "
        f"{title_suffix}\n"
        f"steps={start_step}-{end_step} "
        f"sum_reward={best['sum_reward']:.3f} "
        f"mean_reward={best['mean_reward']:.3f}"
    )
    ax.grid(False)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="final_reward on best segment")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def print_summary(rows):
    terminated = sum(1 for row in rows if row.get("terminated"))
    truncated = sum(1 for row in rows if row.get("truncated"))
    max_final_error = max(abs(row["final_reward_error"]) for row in rows)
    max_target_error = max(abs(row["target_angle_error"]) for row in rows)
    final_rewards = np.array([row["final_reward"] for row in rows], dtype=np.float32)
    print(f"rows={len(rows)} terminated={terminated} truncated={truncated}")
    print(
        "final_reward: "
        f"min={float(np.min(final_rewards)):.3f} "
        f"max={float(np.max(final_rewards)):.3f} "
        f"mean={float(np.mean(final_rewards)):.3f}"
    )
    print(
        "formula_check: "
        f"max_abs(final - (orientation + velocity + collision))={max_final_error:.6f} "
        f"max_abs(target_angle_error)={max_target_error:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Parse and plot lane_check_heading logs.")
    parser.add_argument("log_path", help="Path to the captured lane_check_heading text log.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for outputs. Defaults to test/lane_check_plots/<log_stem>.",
    )
    parser.add_argument("--max-lp-dist", type=float, default=0.05)
    parser.add_argument("--target-angle-deg-at-edge", type=float, default=45.0)
    parser.add_argument("--tile-size", type=float, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--map", default=None, help="Map name under maps/ to draw the real map background.")
    parser.add_argument("--maps-dir", default=str(PROJECT_ROOT / "maps"), help="Directory that contains map folders.")
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=1,
        help="Annotate every Nth point on the map figure. Default 1 annotates all parsed points.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else DEFAULT_OUT_ROOT / log_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_log(log_path)
    if not rows:
        raise RuntimeError(f"No lane_check_heading entries parsed from {log_path}")

    add_checks(rows, max_lp_dist=float(args.max_lp_dist), target_angle_deg_at_edge=float(args.target_angle_deg_at_edge))
    csv_path = out_dir / "parsed_lane_check.csv"
    fig_path = out_dir / "lane_check_summary.png"
    map_fig_path = out_dir / "lane_check_annotated_map.png"
    best_route_fig_path = out_dir / "lane_check_best_reward_route.png"
    map_dir = None
    if args.map:
        candidate = Path(args.maps_dir).expanduser().resolve() / args.map
        if not candidate.exists():
            raise FileNotFoundError(f"Map directory not found: {candidate}")
        map_dir = candidate
    write_csv(rows, csv_path)
    plot_rows(rows, fig_path)
    plot_annotated_map_with_background(
        rows,
        map_fig_path,
        annotate_every=max(1, int(args.annotate_every)),
        map_dir=map_dir,
        tile_size=float(args.tile_size),
    )
    plot_best_reward_route(
        rows,
        best_route_fig_path,
        map_dir=map_dir,
        tile_size=float(args.tile_size),
    )
    print_summary(rows)
    print(f"csv={csv_path}")
    print(f"figure={fig_path}")
    print(f"annotated_map={map_fig_path}")
    print(f"best_reward_route={best_route_fig_path}")


if __name__ == "__main__":
    main()
