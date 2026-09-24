from __future__ import annotations

import argparse

from trajectory_dash.app import run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory Dash player")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash/Flask debug mode without the auto-reloader.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(debug=args.debug, host=args.host, port=args.port)
