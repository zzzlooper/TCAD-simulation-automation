#!/usr/bin/env python3
"""Dummy simulation executable used for orchestrator testing."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path


def parse_geometry_factor(geometry: str) -> float:
    numbers = [int(part) for part in re.findall(r"\d+", geometry)]
    if not numbers:
        return 1.0
    return 1.0 + (sum(numbers) % 9) * 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy TCAD-like simulator.")
    parser.add_argument("--voltage", type=float, required=True, help="Bias voltage (V)")
    parser.add_argument(
        "--temperature", type=float, required=True, help="Temperature (K)"
    )
    parser.add_argument("--geometry", type=str, required=True, help="Geometry tag")
    parser.add_argument(
        "--repeat-index", type=int, default=0, help="Repeat index for duplicated runs"
    )
    parser.add_argument(
        "--fail-probability",
        type=float,
        default=0.0,
        help="Injected random failure probability [0, 1].",
    )
    parser.add_argument(
        "--sleep-min", type=float, default=0.15, help="Minimum runtime delay (seconds)"
    )
    parser.add_argument(
        "--sleep-max", type=float, default=0.45, help="Maximum runtime delay (seconds)"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.015,
        help="Relative random noise applied to metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(
        f"{args.geometry}|{args.voltage}|{args.temperature}|{args.repeat_index}"
    )

    if args.temperature <= 0:
        print("Invalid temperature: must be > 0 K", file=sys.stderr)
        return 2

    if not (0.0 <= args.fail_probability <= 1.0):
        print("fail-probability must be between 0 and 1", file=sys.stderr)
        return 2

    if args.sleep_max < args.sleep_min:
        print("sleep-max must be >= sleep-min", file=sys.stderr)
        return 2

    time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    # Deterministic failure rule + probabilistic failure injection.
    if (args.voltage > 1.2 and args.temperature < 260) or (
        random.random() < args.fail_probability
    ):
        print("Numerical solver failed to converge.", file=sys.stderr)
        return 3

    geometry_factor = parse_geometry_factor(args.geometry)
    thermal_scale = math.exp(-320.0 / args.temperature)
    base_current = args.voltage * thermal_scale * geometry_factor * 1e-3
    base_leak = (args.voltage**2) * (args.temperature / 300.0) * 1e-9 / geometry_factor
    noise = 1.0 + random.uniform(-args.noise_level, args.noise_level)

    idsat_a = base_current * noise
    ileak_a = base_leak * (2.0 - noise)

    result = {
        "voltage_v": args.voltage,
        "temperature_k": args.temperature,
        "geometry": args.geometry,
        "repeat_index": args.repeat_index,
        "metrics": {
            "idsat_a": idsat_a,
            "ileak_a": ileak_a,
            "idsat_mA_per_um": idsat_a * 1e3 / max(1.0, geometry_factor),
        },
    }

    Path("result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"RUN OK | geometry={args.geometry} V={args.voltage:.3f} T={args.temperature:.1f}K "
        f"Idsat={idsat_a:.5e}A Ileak={ileak_a:.5e}A"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
