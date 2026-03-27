#!/usr/bin/env python3
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENGINE_NAME = "dts-matrix-engine"
DEFAULT_WORLD_PORT = 7501
DEFAULT_CONTROL_PORT = 7502
DEFAULT_WS_CONTROL_PORT = 7503


@dataclass
class InstanceConfig:
    index: int
    map_label: str
    map_arg: str
    engine_name: str
    port_offset: int
    world_port: int
    control_port: int
    ws_control_port: int
    cmd: list[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple `dts matrix run --standalone` instances with "
            "custom engine names and port offsets."
        )
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=None,
        help="comma-separated map names or paths; default is all maps under --maps-dir",
    )
    parser.add_argument(
        "--maps-dir",
        type=str,
        default="./maps",
        help="base directory for map names that are not already paths",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./multi_standalone_logs",
        help="directory for per-instance stdout/stderr logs",
    )
    parser.add_argument(
        "--engine-name-prefix",
        type=str,
        default=DEFAULT_ENGINE_NAME,
        help="engine container name prefix; first instance uses the bare prefix",
    )
    parser.add_argument(
        "--port-offsets",
        type=str,
        default=None,
        help="optional comma-separated port offsets; default is 0,20,30,40,...",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="pass --sandbox to each dts matrix run invocation",
    )
    parser.add_argument(
        "--graphics-api",
        type=str,
        default="opengl",
        choices=("opengl", "vulkan", "default"),
        help="renderer graphics API for each standalone instance",
    )
    parser.add_argument("--pull", action="store_true", help="allow image pull; default adds --no-pull")
    parser.add_argument("--dry-run", action="store_true", help="print commands without launching")
    return parser.parse_args()


def discover_maps(maps_root: Path) -> list[str]:
    maps = []
    for child in sorted(maps_root.iterdir()):
        if child.is_dir() and (child / "main.yaml").exists():
            maps.append(child.name)
    if not maps:
        raise RuntimeError(f"No maps found under {maps_root}")
    return maps


def resolve_map_arg(item: str, maps_root: Path) -> tuple[str, str]:
    raw = item.strip()
    if not raw:
        raise ValueError("empty map entry")

    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        return candidate.resolve().as_posix(), candidate.name

    candidate = maps_root / raw
    if candidate.is_dir():
        return candidate.resolve().as_posix(), candidate.name

    raise FileNotFoundError(f"Map not found: {raw}")


def default_offsets(count: int) -> list[int]:
    offsets = [0]
    for idx in range(1, count):
        offsets.append((idx + 1) * 10)
    return offsets


def parse_offsets(raw: str | None, count: int) -> list[int]:
    if raw is None:
        return default_offsets(count)
    offsets = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(offsets) != count:
        raise ValueError(
            f"--port-offsets expects {count} values for {count} maps, got {len(offsets)}"
        )
    return offsets


def build_instances(args: argparse.Namespace) -> list[InstanceConfig]:
    maps_root = Path(args.maps_dir).expanduser().resolve()
    if args.maps:
        map_items = [m.strip() for m in args.maps.split(",") if m.strip()]
    else:
        map_items = discover_maps(maps_root)
    if not map_items:
        raise ValueError("No maps given")

    offsets = parse_offsets(args.port_offsets, len(map_items))
    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    instances: list[InstanceConfig] = []
    for idx, raw_map in enumerate(map_items):
        map_arg, map_label = resolve_map_arg(raw_map, maps_root)
        offset = offsets[idx]
        engine_name = args.engine_name_prefix if idx == 0 else f"{args.engine_name_prefix}-{idx + 1}"
        cmd = [
            "dts",
            "matrix",
            "run",
            "--standalone",
            "--map",
            map_arg,
            "--verbose",
        ]
        if args.graphics_api == "opengl":
            cmd.append("--force-opengl")
        elif args.graphics_api == "vulkan":
            cmd.append("--force-vulkan")
        if args.sandbox:
            cmd.append("--sandbox")
        if not args.pull:
            cmd.append("--no-pull")
        if idx > 0 or offset != 0:
            cmd.extend(["--engine-name", engine_name, "--port-offset", str(offset)])

        instances.append(
            InstanceConfig(
                index=idx + 1,
                map_label=map_label,
                map_arg=map_arg,
                engine_name=engine_name,
                port_offset=offset,
                world_port=DEFAULT_WORLD_PORT + offset,
                control_port=DEFAULT_CONTROL_PORT + offset,
                ws_control_port=DEFAULT_WS_CONTROL_PORT + offset,
                cmd=cmd,
                log_path=logdir / f"{idx + 1:02d}_{map_label}.log",
            )
        )
    return instances


def terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def stop_engine_container(engine_name: str) -> None:
    subprocess.run(
        ["docker", "stop", engine_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    args = parse_args()
    instances = build_instances(args)

    for inst in instances:
        print(
            f"[PLAN] #{inst.index} map={inst.map_label} "
            f"engine={inst.engine_name} port_offset={inst.port_offset} "
            f"world={inst.world_port} control={inst.control_port} ws_control={inst.ws_control_port}"
        )
        print(f"[PLAN] log={inst.log_path}")
        print(f"[PLAN] cmd={shlex.join(inst.cmd)}")

    if args.dry_run:
        return 0

    procs: list[tuple[InstanceConfig, subprocess.Popen, object]] = []
    try:
        for inst in instances:
            log_file = open(inst.log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                inst.cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append((inst, proc, log_file))
            print(f"[STARTED] #{inst.index} pid={proc.pid} engine={inst.engine_name} map={inst.map_label}")
            time.sleep(2.0)
            if proc.poll() is not None:
                print(
                    f"[FAILED] #{inst.index} exited early with code {proc.returncode}. "
                    f"See {inst.log_path}",
                    file=sys.stderr,
                )
                return int(proc.returncode or 1)

        print("[READY] all standalone instances launched; press Ctrl-C to stop them")
        while True:
            dead = []
            for inst, proc, _ in procs:
                if proc.poll() is not None:
                    dead.append((inst, proc.returncode))
            if dead:
                for inst, rc in dead:
                    print(
                        f"[EXITED] #{inst.index} engine={inst.engine_name} "
                        f"map={inst.map_label} returncode={rc}. See {inst.log_path}",
                        file=sys.stderr,
                    )
                return 1
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[STOP] stopping standalone instances")
        return 0
    finally:
        for inst, proc, log_file in reversed(procs):
            terminate_process(proc)
            stop_engine_container(inst.engine_name)
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
