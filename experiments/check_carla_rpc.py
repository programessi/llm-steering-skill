from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--town", default=None)
    args = parser.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout_s)
    if args.town:
        world = client.load_world(args.town)
    else:
        world = client.get_world()
    result = {
        "ok": True,
        "host": args.host,
        "port": args.port,
        "server_version": client.get_server_version(),
        "client_version": client.get_client_version(),
        "world": world.get_map().name,
        "actor_count": len(world.get_actors()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
