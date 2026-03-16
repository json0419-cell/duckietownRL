#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Iterable, Optional


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
    # Include common local renderer process names so cleanup is more reliable.
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


def stop_engine(
    proc: subprocess.Popen | None,
    container_name: str,
    *,
    stop_renderer: bool,
    renderer_process_name: str,
) -> None:
    if proc is not None and proc.poll() is None:
        # Stop the launcher and its children as a process group when possible.
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
    if stop_renderer:
        stop_renderer_processes(renderer_process_name)
        # Remove the optional viewer container if it was started manually.
        subprocess.run(
            ["docker", "rm", "-f", "my-viewer"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # Remove any leftover engine container as a final cleanup step.
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_engine_cmd(
    map_name: str,
    *,
    no_pull: bool = True,
    engine_only: bool = False,
) -> list[str]:
    if engine_only:
        cmd = ["dts", "matrix", "engine", "run", "--map", map_name]
    else:
        cmd = ["dts", "matrix", "run", "--standalone", "--map", map_name, "--no-tutorial"]

    if no_pull:
        cmd.append("--no-pull")
    return cmd


def start_engine(
    map_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 7501,
    entity_name: str = "map_0/vehicle_0",
    ready_timeout: float = 40.0,
    container_name: str = "dts-matrix-engine",
    renderer_process_name: str = "duckiematrix.x86_64",
    no_pull: bool = True,
    engine_only: bool = False,
    env_overrides: Optional[dict[str, str]] = None,
    stdout=None,
    stderr=None,
) -> tuple[subprocess.Popen, list[str], str]:
    cmd = build_engine_cmd(
        map_name,
        no_pull=no_pull,
        engine_only=engine_only,
    )

    print("[INFO] starting engine:", " ".join(cmd))
    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    except FileNotFoundError as e:
        raise RuntimeError("`dts` command not found in PATH") from e

    pose_url = f"http://{host}:{port}/robot/{entity_name}/state/pose/"
    cam_url = f"http://{host}:{port}/robot/{entity_name}/sensor/camera/front_center/jpeg/"
    ready_url = pose_url if engine_only else cam_url
    ready_label = "engine pose" if engine_only else "camera stream"

    if engine_only:
        ready = wait_url_nonempty(ready_url, ready_timeout)
        ready_reason = "pose stream is reachable"
    else:
        ready, ready_reason = wait_standalone_ready(pose_url, cam_url, ready_timeout)
    if not ready:
        stop_engine(
            proc,
            container_name,
            stop_renderer=not engine_only,
            renderer_process_name=renderer_process_name,
        )
        raise RuntimeError(f"{ready_label} not ready within {ready_timeout:.1f}s: {ready_url} ({ready_reason})")

    if engine_only:
        print(f"[INFO] engine ready: {pose_url} ({ready_reason})")
    else:
        print(f"[INFO] standalone ready: {cam_url} ({ready_reason})")

    return proc, cmd, ready_url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="loop", help="map name, e.g. loop/sandbox")
    parser.add_argument("--host", default="127.0.0.1", help="engine host")
    parser.add_argument("--port", type=int, default=7501, help="engine DTPS port")
    parser.add_argument("--entity-name", default="map_0/vehicle_0", help="entity name for readiness check")
    parser.add_argument("--ready-timeout", type=float, default=40.0, help="wait engine readiness timeout")
    parser.add_argument("--duration", type=float, default=15.0, help="run this many seconds, <=0 means wait Enter")
    parser.add_argument("--container-name", default="dts-matrix-engine", help="engine docker container name")
    parser.add_argument(
        "--renderer-process-name",
        default="duckiematrix.x86_64",
        help="process name pattern for local renderer cleanup",
    )
    parser.add_argument("--pull", action="store_true", help="allow dts to pull instead of using --no-pull")
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="start engine only (`dts matrix engine run`); default is standalone (`dts matrix run --standalone`)",
    )
    args = parser.parse_args()

    try:
        proc, _, _ = start_engine(
            args.map,
            host=args.host,
            port=args.port,
            entity_name=args.entity_name,
            ready_timeout=args.ready_timeout,
            container_name=args.container_name,
            renderer_process_name=args.renderer_process_name,
            no_pull=not args.pull,
            engine_only=args.engine_only,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    try:
        if args.duration > 0:
            print(f"[INFO] running for {args.duration:.1f}s ...")
            time.sleep(args.duration)
        else:
            input("[INFO] press Enter to stop engine ...")
    finally:
        print("[INFO] stopping engine ...")
        stop_engine(
            proc,
            args.container_name,
            stop_renderer=not args.engine_only,
            renderer_process_name=args.renderer_process_name,
        )
        print("[INFO] engine stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
