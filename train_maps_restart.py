#!/usr/bin/env python3
"""
Switch maps by segment and restart the training workflow for each segment.

Workflow per segment:
  1) Stop engine and renderer processes left by the previous segment.
  2) Start a fresh runtime on the target map.
  3) Run Main.py for the current segment budget.
  4) Save the model and continue to the next segment.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable


def parse_csv(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def parse_int_csv(values: str) -> list[int]:
    out = []
    for token in parse_csv(values):
        out.append(int(token))
    return out


def wait_url_nonempty(url: str, timeout_s: float, label: str) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            raw = urllib.request.urlopen(url, timeout=2.0).read()
            if raw:
                return
        except Exception as e:  # pragma: no cover
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"{label} not ready within {timeout_s:.1f}s at {url}. Last error: {last_err}")


def _pids_from_pattern(pattern: str) -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    pids = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _kill_pids(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def stop_renderer_processes(renderer_process_name: str) -> None:
    patterns = [
        renderer_process_name,
        "duckiematrix.x86_64",
        "/.duckietown/duckiematrix/releases/",
        "UnityCrashHandler",
    ]
    for pat in patterns:
        _kill_pids(_pids_from_pattern(pat), signal.SIGTERM)
    time.sleep(1.0)
    for pat in patterns:
        _kill_pids(_pids_from_pattern(pat), signal.SIGKILL)


def stop_duckiematrix(
    proc: subprocess.Popen | None,
    *,
    container_name: str,
    renderer_process_name: str,
) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=5)

    stop_renderer_processes(renderer_process_name)
    # Remove the external viewer container.
    run_quiet(["docker", "rm", "-f", "my-viewer"])
    # Remove the engine container created by the launcher.
    run_quiet(["docker", "rm", "-f", container_name])


def run_quiet(cmd: Iterable[str]) -> None:
    subprocess.run(list(cmd), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def hard_cleanup(container_name: str, renderer_process_name: str) -> None:
    stop_renderer_processes(renderer_process_name)
    run_quiet(["docker", "rm", "-f", "my-viewer"])
    run_quiet(["docker", "rm", "-f", container_name])


def build_schedule(maps: list[str], steps: list[int], cycles: int) -> list[tuple[str, int]]:
    if len(steps) == 1:
        steps = steps * len(maps)
    if len(steps) != len(maps):
        raise ValueError("`--timesteps-per-map` must have 1 value or the same count as `--maps`.")
    schedule: list[tuple[str, int]] = []
    for _ in range(cycles):
        for m, s in zip(maps, steps):
            schedule.append((m, s))
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=str, default="loop,sandbox", help="comma-separated map sequence")
    parser.add_argument(
        "--timesteps-per-map",
        type=str,
        default="200000",
        help="single integer or comma-separated integers aligned with --maps",
    )
    parser.add_argument("--cycles", type=int, default=1, help="repeat map sequence N times")
    parser.add_argument("--logdir", type=str, default="./runs_db21j_ppo_old", help="training output directory")
    parser.add_argument("--main-py", type=str, default="./Main.py", help="path to training entry script")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable for Main.py")
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="entity name")
    parser.add_argument("--renderer-version", type=str, default="0.7.0", help="renderer version for dts matrix run")
    parser.add_argument("--container-name", type=str, default="dts-matrix-engine", help="engine container name")
    parser.add_argument(
        "--renderer-process-name",
        type=str,
        default="duckiematrix.x86_64",
        help="process name pattern for renderer cleanup",
    )
    parser.add_argument("--engine-host", type=str, default="127.0.0.1", help="engine host")
    parser.add_argument("--engine-port", type=int, default=7501, help="engine port")
    parser.add_argument("--engine-wait-timeout", type=float, default=40.0)
    parser.add_argument("--camera-wait-timeout", type=float, default=40.0)
    parser.add_argument("--save-freq", type=int, default=200000, help="checkpoint frequency passed to Main.py")
    parser.add_argument("--device", type=str, default="auto", help="device passed to Main.py")
    parser.add_argument("--headless", action="store_true", help="pass --headless to Main.py")
    parser.add_argument(
        "--respawn-mode",
        type=str,
        default="random",
        choices=["random", "fixed"],
        help="shared respawn mode for the training wrapper and the patched engine image",
    )
    args = parser.parse_args()

    maps = parse_csv(args.maps)
    if not maps:
        raise ValueError("No maps provided.")
    steps = parse_int_csv(args.timesteps_per_map)
    schedule = build_schedule(maps, steps, args.cycles)

    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    proc_log_dir = logdir / "orchestrator_logs"
    proc_log_dir.mkdir(parents=True, exist_ok=True)

    pose_url = f"http://{args.engine_host}:{args.engine_port}/robot/{args.entity_name}/state/pose/"
    cam_url = (
        f"http://{args.engine_host}:{args.engine_port}/robot/"
        f"{args.entity_name}/sensor/camera/front_center/jpeg/"
    )

    model_path: Path | None = None
    matrix_proc: subprocess.Popen | None = None
    matrix_log = None

    def shutdown(*_):
        nonlocal matrix_proc, matrix_log
        stop_duckiematrix(
            matrix_proc,
            container_name=args.container_name,
            renderer_process_name=args.renderer_process_name,
        )
        if matrix_log:
            matrix_log.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        for seg_idx, (map_name, timesteps) in enumerate(schedule, start=1):
            print(f"\n[SEGMENT {seg_idx}/{len(schedule)}] map={map_name}, timesteps={timesteps}")
            hard_cleanup(args.container_name, args.renderer_process_name)

            matrix_log_path = proc_log_dir / f"matrix_seg{seg_idx:02d}_{map_name}.log"
            matrix_log = open(matrix_log_path, "w", encoding="utf-8")

            matrix_cmd = [
                "dts",
                "matrix",
                "run",
                "--standalone",
                "--embedded",
                "--map",
                map_name,
                "--version",
                args.renderer_version,
                "--no-tutorial",
                "--no-pull",
            ]
            print("[INFO] starting duckiematrix:", " ".join(matrix_cmd))
            matrix_env = os.environ.copy()
            matrix_env["DUCKIEMATRIX_RESPAWN_MODE"] = args.respawn_mode
            matrix_proc = subprocess.Popen(
                matrix_cmd,
                stdout=matrix_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=matrix_env,
            )
            wait_url_nonempty(pose_url, args.engine_wait_timeout, "engine pose")
            wait_url_nonempty(cam_url, args.camera_wait_timeout, "camera stream")

            save_name = f"ppo_db21j_seg{seg_idx:02d}_{map_name}"
            train_cmd = [
                args.python,
                args.main_py,
                "--timesteps",
                str(timesteps),
                "--logdir",
                str(logdir),
                "--entity-name",
                args.entity_name,
                "--save-freq",
                str(args.save_freq),
                "--save-name",
                save_name,
                "--device",
                args.device,
                "--respawn-mode",
                args.respawn_mode,
            ]
            if args.headless:
                train_cmd.append("--headless")
            if model_path is not None:
                train_cmd.extend(["--load-model", str(model_path)])

            print("[INFO] starting training:", " ".join(train_cmd))
            subprocess.run(train_cmd, check=True)

            candidate = logdir / f"{save_name}.zip"
            if not candidate.exists():
                raise FileNotFoundError(f"Segment model not found: {candidate}")
            model_path = candidate
            print(f"[INFO] segment done, model={model_path}")

            stop_duckiematrix(
                matrix_proc,
                container_name=args.container_name,
                renderer_process_name=args.renderer_process_name,
            )
            matrix_proc = None
            hard_cleanup(args.container_name, args.renderer_process_name)

            if matrix_log:
                matrix_log.close()
                matrix_log = None

        print("\n[DONE] all segments finished.")
        if model_path is not None:
            print(f"[DONE] final model: {model_path}")
        return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
