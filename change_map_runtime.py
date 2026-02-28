#!/usr/bin/env python3
"""
Runtime map switch client for Duckiematrix.

Usage:
  python change_map_runtime.py --map loop
  python change_map_runtime.py --map sandbox --host 127.0.0.1 --port 7501
"""

import argparse
import asyncio
import time

import dtps
from duckietown_messages.standard.dictionary import Dictionary
from duckietown_messages.standard.header import Header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, type=str, help="target map name")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="DTPS host")
    parser.add_argument("--port", default=7501, type=int, help="DTPS port")
    return parser.parse_args()


async def _publish_change_map(host: str, port: int, map_name: str) -> None:
    context = await dtps.context(urls=[f"http://{host}:{port}/"])
    queue = await (context / "robot" / "__engine__" / "change_map").queue_create()
    message = Dictionary(
        header=Header(timestamp=time.time()),
        data={"map": map_name},
    )
    await queue.publish(message.to_rawdata())


def main() -> None:
    args = parse_args()
    asyncio.run(_publish_change_map(args.host, args.port, args.map))
    print(
        f"change_map request sent: map='{args.map}' to "
        f"http://{args.host}:{args.port}/robot/__engine__/change_map",
    )


if __name__ == "__main__":
    main()
