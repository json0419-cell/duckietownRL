#!/usr/bin/env python3
import argparse
import os
import random
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_IMAGE = "deeplearninglab/myduckiematrix:ran-respawn"
DEFAULT_RENDERER = Path.home() / ".duckietown/duckiematrix/releases/v0.7.0-linux/duckiematrix.x86_64"
DEFAULT_BILLBOARDS = Path.home() / ".duckietown/shell/databases/billboards.yaml"
DEFAULT_SECRETS = Path.home() / ".duckietown/shell/profiles/ente/databases/secrets.yaml"
DEFAULT_PORTS = (7501, 7502, 7503, 7504, 7505)


def normalize_map_name(map_arg: str) -> str:
    path = Path(map_arg)
    return path.name or str(map_arg)


def read_dt2_token(secrets_path: Path) -> str:
    if not secrets_path.exists():
        raise RuntimeError(f"secrets file not found: {secrets_path}")
    for line in secrets_path.read_text().splitlines():
        if line.strip().startswith("token/dt2:"):
            _, value = line.split(":", 1)
            token = value.strip()
            if token:
                return token
    raise RuntimeError(f"token/dt2 not found in {secrets_path}")


def read_billboard_names(billboards_path: Path) -> str:
    if not billboards_path.exists():
        return "{}"
    names: list[str] = []
    for line in billboards_path.read_text().splitlines():
        if line.startswith("    ") and not line.startswith("        ") and ":" in line:
            key = line.split(":", 1)[0].strip()
            if key:
                names.append(key)
    if not names:
        return "{}"
    parts = [f"\"{name}\": 1" for name in names]
    return "{" + ", ".join(parts) + "}"


def pick_billboard_text() -> str:
    choices = [
        "If it is not tested, treat it as broken!",
        "Eat and sleep!",
        "Testing first.",
        "Slow is smooth.",
    ]
    return random.choice(choices)


def _read_url(url: str, timeout_s: float = 2.0) -> bytes | None:
    try:
        return urllib.request.urlopen(url, timeout=timeout_s).read()
    except Exception:
        return None


def wait_url_nonempty(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = _read_url(url)
        if raw:
            return True
        time.sleep(0.5)
    return False


def wait_url_changing(url: str, timeout_s: float, *, min_changes: int = 1) -> bool:
    deadline = time.time() + timeout_s
    previous = None
    changes = 0
    while time.time() < deadline:
        raw = _read_url(url)
        if raw:
            if previous is not None and raw != previous:
                changes += 1
                if changes >= min_changes:
                    return True
            previous = raw
        time.sleep(0.5)
    return False


def wait_standalone_ready(pose_url: str, cam_url: str, timeout_s: float) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        if not wait_url_nonempty(pose_url, min(remaining, 5.0)):
            continue
        remaining = max(0.0, deadline - time.time())
        if remaining <= 0.0:
            break
        if not wait_url_nonempty(cam_url, min(remaining, 5.0)):
            continue
        remaining = max(0.0, deadline - time.time())
        if remaining <= 0.0:
            break
        if wait_url_changing(cam_url, min(remaining, 10.0), min_changes=1):
            return True, "camera stream is updating"
    return False, "camera stream is not updating"


def build_engine_cmd(args: argparse.Namespace) -> list[str]:
    map_name = normalize_map_name(args.map)
    return [
        "docker",
        "run",
        "--name",
        args.container_name,
        "--network",
        "host",
        "--rm",
        "-d",
        args.image,
        "--",
        "--mode",
        args.mode,
        "--map",
        map_name,
        "--renderers",
        "1",
        "--hostname",
        args.hostname,
        "--world-control-out-port",
        str(args.world_control_out_port),
        "--matrix-control-out-port",
        str(args.matrix_control_out_port),
        "--matrix-websocket-bridge-control-out-port",
        str(args.matrix_websocket_bridge_control_out_port),
        "--matrix-websocket-bridge-data-out-port",
        str(args.matrix_websocket_bridge_data_out_port),
        "--matrix-websocket-bridge-data-in-port",
        str(args.matrix_websocket_bridge_data_in_port),
    ]


def build_renderer_cmd(args: argparse.Namespace, token: str) -> list[str]:
    return [
        str(args.renderer_binary),
        "-logfile",
        "/dev/stdout",
        f"--force-{args.graphics_api}",
        "--token",
        token,
        "--billboard",
        args.billboard_text,
        "--billboards-path",
        str(args.billboards_path),
        "--billboard-names",
        args.billboard_names,
    ]


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start Duckiematrix engine + local renderer without using "
            "`dts matrix run --standalone`. "
            "The renderer path currently assumes the default standalone "
            "port set 7501-7505."
        )
    )
    parser.add_argument("--map", default="maps/loop")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-name", default="dts-matrix-engine-manual")
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--mode", default="realtime", choices=("realtime", "gym"))
    parser.add_argument("--world-control-out-port", type=int, default=7501)
    parser.add_argument("--matrix-control-out-port", type=int, default=7502)
    parser.add_argument("--matrix-websocket-bridge-control-out-port", type=int, default=7503)
    parser.add_argument("--matrix-websocket-bridge-data-out-port", type=int, default=7504)
    parser.add_argument("--matrix-websocket-bridge-data-in-port", type=int, default=7505)
    parser.add_argument("--entity-name", default="map_0/vehicle_0")
    parser.add_argument("--ready-timeout", type=float, default=40.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--renderer-binary", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--graphics-api", choices=("opengl", "vulkan"), default="opengl")
    parser.add_argument("--secrets-path", type=Path, default=DEFAULT_SECRETS)
    parser.add_argument("--billboards-path", type=Path, default=DEFAULT_BILLBOARDS)
    parser.add_argument("--billboard-text", default=pick_billboard_text())
    parser.add_argument("--billboard-names", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = (
        args.world_control_out_port,
        args.matrix_control_out_port,
        args.matrix_websocket_bridge_control_out_port,
        args.matrix_websocket_bridge_data_out_port,
        args.matrix_websocket_bridge_data_in_port,
    )
    if ports != DEFAULT_PORTS:
        raise RuntimeError(
            "manual renderer wiring is only validated for the default standalone "
            "ports 7501-7505. Custom-port renderer startup is not solved yet."
        )
    if not args.renderer_binary.exists():
        raise RuntimeError(f"renderer binary not found: {args.renderer_binary}")

    token = read_dt2_token(args.secrets_path)
    if args.billboard_names is None:
        args.billboard_names = read_billboard_names(args.billboards_path)

    engine_cmd = build_engine_cmd(args)
    renderer_cmd = build_renderer_cmd(args, token)

    print(shlex.join(engine_cmd))
    engine = subprocess.run(engine_cmd, check=False, capture_output=True, text=True)
    if engine.stdout:
        print(engine.stdout.strip())
    if engine.returncode != 0:
        if engine.stderr:
            print(engine.stderr.strip(), file=sys.stderr)
        return engine.returncode

    print(shlex.join(renderer_cmd))
    renderer_proc: subprocess.Popen | None = None
    try:
        renderer_proc = subprocess.Popen(renderer_cmd, start_new_session=True)
        pose_url = f"http://{args.hostname}:{args.world_control_out_port}/robot/{args.entity_name}/state/pose/"
        cam_url = (
            f"http://{args.hostname}:{args.world_control_out_port}/robot/"
            f"{args.entity_name}/sensor/camera/front_center/jpeg/"
        )
        ready, reason = wait_standalone_ready(pose_url, cam_url, args.ready_timeout)
        if not ready:
            raise RuntimeError(f"standalone not ready: {reason}")
        print(f"[INFO] standalone ready: {cam_url} ({reason})")
        if args.duration <= 0:
            input("Press Enter to stop...\n")
        else:
            print(f"[INFO] running for {args.duration:.1f}s ...")
            time.sleep(args.duration)
        return 0
    finally:
        print("[INFO] stopping standalone ...")
        stop_process(renderer_proc)
        subprocess.run(
            ["docker", "rm", "-f", args.container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[INFO] standalone stopped")


if __name__ == "__main__":
    raise SystemExit(main())
