#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request


def wait_url_nonempty(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            raw = urllib.request.urlopen(url, timeout=2.0).read()
            if raw:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


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
    # Common names for Duckiematrix Unity renderer on Linux.
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
        # Stop dts + children as a process group when possible.
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
        # optional renderer container if user started it manually
        subprocess.run(
            ["docker", "rm", "-f", "my-viewer"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # extra cleanup: if dts left docker container running
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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
    parser.add_argument("--no-pull", action="store_true", help="pass --no-pull to dts")
    parser.add_argument("--embedded", action="store_true", help="pass --embedded to dts")
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="start engine only (`dts matrix engine run`); default is standalone (`dts matrix run --standalone`)",
    )
    args = parser.parse_args()

    if args.engine_only:
        cmd = ["dts", "matrix", "engine", "run", "--map", args.map]
        if args.embedded:
            cmd.append("--embedded")
        if args.no_pull:
            cmd.append("--no-pull")
    else:
        cmd = ["dts", "matrix", "run", "--standalone", "--map", args.map, "--no-tutorial"]
        if args.embedded:
            cmd.append("--embedded")
        if args.no_pull:
            cmd.append("--no-pull")

    print("[INFO] starting engine:", " ".join(cmd))
    try:
        # New session => we can terminate whole process group reliably.
        proc = subprocess.Popen(cmd, start_new_session=True)
    except FileNotFoundError:
        print("[ERROR] `dts` command not found in PATH")
        return 1

    pose_url = f"http://{args.host}:{args.port}/robot/{args.entity_name}/state/pose/"
    cam_url = f"http://{args.host}:{args.port}/robot/{args.entity_name}/sensor/camera/front_center/jpeg/"
    ready_url = pose_url if args.engine_only else cam_url
    ready_label = "engine pose" if args.engine_only else "camera stream"
    ready = wait_url_nonempty(ready_url, args.ready_timeout)
    if not ready:
        print(f"[ERROR] {ready_label} not ready within {args.ready_timeout:.1f}s: {ready_url}")
        stop_engine(
            proc,
            args.container_name,
            stop_renderer=not args.engine_only,
            renderer_process_name=args.renderer_process_name,
        )
        return 2

    if args.engine_only:
        print(f"[INFO] engine ready: {pose_url}")
    else:
        print(f"[INFO] standalone ready (camera ok): {cam_url}")
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
