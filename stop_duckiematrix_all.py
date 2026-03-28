#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import time


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


def stop_matching_processes(patterns: list[str]) -> None:
    seen = set()
    for pat in patterns:
        for pid in _pids_from_pattern(pat):
            if pid != os.getpid():
                seen.add(pid)

    _kill_pids(sorted(seen), signal.SIGTERM)
    time.sleep(1.0)

    remaining = []
    for pid in sorted(seen):
        try:
            os.kill(pid, 0)
            remaining.append(pid)
        except ProcessLookupError:
            pass
        except Exception:
            remaining.append(pid)
    _kill_pids(remaining, signal.SIGKILL)


def stop_engine_containers(prefix: str) -> None:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = []
    for line in (proc.stdout or "").splitlines():
        name = line.strip()
        if not name:
            continue
        if name == prefix or name.startswith(prefix + "-"):
            names.append(name)

    if names:
        subprocess.run(
            ["docker", "rm", "-f", *names],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    subprocess.run(
        ["docker", "rm", "-f", "my-viewer"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Force-stop local Duckiematrix renderers, dts launcher processes, and engine containers."
    )
    parser.add_argument(
        "--engine-name-prefix",
        default="dts-matrix-engine",
        help="engine container name prefix to remove",
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help="also run `ray stop --force` after cleaning Duckiematrix processes",
    )
    args = parser.parse_args()

    patterns = [
        "duckiematrix.x86_64",
        "/.duckietown/duckiematrix/releases/",
        "UnityCrashHandler",
        "dts matrix run --standalone",
        "dts matrix run --browser",
        "dts matrix engine run",
    ]

    stop_matching_processes(patterns)
    stop_engine_containers(args.engine_name_prefix)

    if args.ray:
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("[DONE] stopped Duckiematrix processes and removed matching engine containers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
