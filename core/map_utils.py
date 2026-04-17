from pathlib import Path


def discover_maps(maps_root: Path) -> list[str]:
    maps = []
    for child in sorted(maps_root.iterdir()):
        if child.is_dir() and (child / "main.yaml").exists():
            maps.append(child.name)
    if not maps:
        raise RuntimeError(f"No maps found under {maps_root}")
    return maps


def map_engine_arg(maps_dir_arg: str, map_name: str) -> str:
    return str(Path(maps_dir_arg) / map_name)

