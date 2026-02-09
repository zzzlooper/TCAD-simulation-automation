#!/usr/bin/env python3
"""TCAD-style batch simulation orchestrator."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class RunSpec:
    run_id: int
    run_name: str
    run_dir: str
    params: Dict[str, Any]
    command: List[str]
    timeout_sec: Optional[float]


@dataclass
class RunResult:
    run_id: int
    run_name: str
    run_dir: str
    status: str
    return_code: Optional[int]
    duration_s: float
    error: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and orchestrate TCAD-style parameterized batch runs."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML or JSON config describing the sweep.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override max worker processes from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing them.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        config = json.loads(raw_text)
    elif suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            )
        config = yaml.safe_load(raw_text)
    else:
        raise ValueError(
            f"Unsupported config extension '{suffix}'. Use .json, .yaml, or .yml."
        )

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping/object.")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required = ["command", "parameters"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    if not isinstance(config["parameters"], dict) or not config["parameters"]:
        raise ValueError("'parameters' must be a non-empty mapping.")

    for name, values in config["parameters"].items():
        if isinstance(values, list):
            if not values:
                raise ValueError(f"Parameter '{name}' has an empty sweep list.")
        else:
            config["parameters"][name] = [values]

    repeats = int(config.get("repeats", 1))
    if repeats < 1:
        raise ValueError("'repeats' must be >= 1.")

    if "max_workers" in config and int(config["max_workers"]) < 1:
        raise ValueError("'max_workers' must be >= 1.")

    if "timeout_sec" in config and float(config["timeout_sec"]) <= 0:
        raise ValueError("'timeout_sec' must be > 0.")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("batch_orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_command(command_cfg: Any) -> List[str]:
    if isinstance(command_cfg, str):
        tokens = shlex.split(command_cfg)
    elif isinstance(command_cfg, list) and command_cfg:
        tokens = [str(token) for token in command_cfg]
    else:
        raise ValueError("'command' must be a non-empty string or list.")
    return tokens


def resolve_existing_paths(tokens: List[str], base_dir: Path) -> List[str]:
    resolved: List[str] = []
    for token in tokens:
        expanded = os.path.expandvars(token)
        candidate = Path(expanded)

        if candidate.is_absolute():
            resolved.append(str(candidate))
            continue

        maybe_path = (base_dir / candidate).resolve()
        if maybe_path.exists():
            resolved.append(str(maybe_path))
        else:
            resolved.append(expanded)
    return resolved


def parameter_grid(parameters: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(parameters.keys())
    values = [parameters[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def param_flag(param_name: str, arg_map: Mapping[str, Optional[str]]) -> Optional[str]:
    if param_name in arg_map:
        mapped = arg_map[param_name]
        if mapped is None:
            return None
        return str(mapped)
    return "--" + param_name.replace("_", "-")


def params_to_args(
    params: Dict[str, Any], arg_map: Mapping[str, Optional[str]]
) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        flag = param_flag(key, arg_map)
        if not flag:
            continue

        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue

        args.extend([flag, str(value)])
    return args


def flatten_result_dict(
    result_obj: Dict[str, Any], prefix: str = "result"
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in result_obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[full_key] = value
        elif isinstance(value, dict):
            flattened.update(flatten_result_dict(value, prefix=full_key))
        else:
            flattened[full_key] = json.dumps(value, sort_keys=True)
    return flattened


def run_simulation(spec: RunSpec) -> RunResult:
    run_dir = Path(spec.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    params_path = run_dir / "params.json"
    params_path.write_text(
        json.dumps(spec.params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    command_path = run_dir / "command.txt"
    command_path.write_text(
        " ".join(shlex.quote(part) for part in spec.command) + "\n", encoding="utf-8"
    )

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    started = time.perf_counter()
    return_code: Optional[int] = None
    status = "failed"
    error = ""
    metrics: Dict[str, Any] = {}

    try:
        with (
            stdout_path.open("w", encoding="utf-8") as out,
            stderr_path.open("w", encoding="utf-8") as err,
        ):
            completed = subprocess.run(
                spec.command,
                cwd=run_dir,
                stdout=out,
                stderr=err,
                check=False,
                timeout=spec.timeout_sec,
                text=True,
            )
        return_code = completed.returncode
        status = "success" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = (
            f"Simulation exceeded timeout ({spec.timeout_sec}s). "
            "Check stdout.log and stderr.log."
        )
    except Exception as exc:
        status = "error"
        error = str(exc)

    duration_s = time.perf_counter() - started

    result_json = run_dir / "result.json"
    if result_json.exists():
        try:
            obj = json.loads(result_json.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                metrics = flatten_result_dict(obj)
        except Exception as exc:
            metrics["result_parse_error"] = str(exc)

    status_payload = {
        "run_id": spec.run_id,
        "run_name": spec.run_name,
        "status": status,
        "return_code": return_code,
        "duration_s": round(duration_s, 6),
        "error": error,
    }
    (run_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return RunResult(
        run_id=spec.run_id,
        run_name=spec.run_name,
        run_dir=str(run_dir),
        status=status,
        return_code=return_code,
        duration_s=duration_s,
        error=error,
        params=spec.params,
        metrics=metrics,
    )


def write_summary_csv(results: List[RunResult], out_path: Path) -> None:
    static_fields = [
        "run_id",
        "run_name",
        "status",
        "return_code",
        "duration_s",
        "run_dir",
        "error",
    ]

    param_fields = sorted({key for row in results for key in row.params.keys()})
    metric_fields = sorted({key for row in results for key in row.metrics.keys()})
    fieldnames = static_fields + param_fields + metric_fields

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in sorted(results, key=lambda item: item.run_id):
            csv_row = {
                "run_id": row.run_id,
                "run_name": row.run_name,
                "status": row.status,
                "return_code": row.return_code if row.return_code is not None else "",
                "duration_s": f"{row.duration_s:.6f}",
                "run_dir": row.run_dir,
                "error": row.error,
            }
            for key in param_fields:
                csv_row[key] = row.params.get(key, "")
            for key in metric_fields:
                csv_row[key] = row.metrics.get(key, "")
            writer.writerow(csv_row)


def build_run_specs(
    config: Dict[str, Any], config_path: Path, campaign_dir: Path
) -> List[RunSpec]:
    command_tokens = normalize_command(config["command"])
    command_tokens = resolve_existing_paths(command_tokens, config_path.parent)

    static_args = [str(token) for token in config.get("static_args", [])]
    static_args = resolve_existing_paths(static_args, config_path.parent)

    arg_map_cfg = config.get("arg_map", {})
    if arg_map_cfg is None:
        arg_map_cfg = {}
    if not isinstance(arg_map_cfg, dict):
        raise ValueError("'arg_map' must be a mapping when provided.")
    arg_map = {str(k): (None if v is None else str(v)) for k, v in arg_map_cfg.items()}

    repeats = int(config.get("repeats", 1))
    timeout_sec = float(config["timeout_sec"]) if "timeout_sec" in config else None

    runs_dir = campaign_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    specs: List[RunSpec] = []
    run_id = 1
    for params in parameter_grid(config["parameters"]):
        for repeat in range(repeats):
            params_with_repeat = dict(params)
            if repeats > 1:
                params_with_repeat["repeat_index"] = repeat

            run_name = f"run_{run_id:04d}"
            run_dir = runs_dir / run_name
            full_command = (
                command_tokens
                + static_args
                + params_to_args(params_with_repeat, arg_map=arg_map)
            )

            specs.append(
                RunSpec(
                    run_id=run_id,
                    run_name=run_name,
                    run_dir=str(run_dir),
                    params=params_with_repeat,
                    command=full_command,
                    timeout_sec=timeout_sec,
                )
            )
            run_id += 1
    return specs


def run_campaign(
    config: Dict[str, Any],
    config_path: Path,
    max_workers_override: Optional[int],
    dry_run: bool,
) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_name = str(config.get("campaign_name", "tcad_batch"))
    output_root = Path(config.get("output_root", "output"))
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    campaign_dir = output_root / f"{campaign_name}_{timestamp}"
    campaign_dir.mkdir(parents=True, exist_ok=False)

    logger = configure_logger(campaign_dir / "orchestrator.log")
    logger.info("Campaign directory: %s", campaign_dir)
    logger.info("Loading sweep from: %s", config_path.resolve())

    snapshot_path = campaign_dir / "config_snapshot.json"
    snapshot_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_specs = build_run_specs(
        config=config, config_path=config_path, campaign_dir=campaign_dir
    )
    logger.info("Planned runs: %d", len(run_specs))
    if not run_specs:
        logger.error("No runs generated. Check your parameter sweep lists.")
        return 2

    if dry_run:
        for spec in run_specs[:10]:
            logger.info(
                "DRY RUN | %s | %s",
                spec.run_name,
                " ".join(shlex.quote(part) for part in spec.command),
            )
        if len(run_specs) > 10:
            logger.info("DRY RUN | ... %d additional runs omitted", len(run_specs) - 10)
        logger.info("Dry run complete. No simulations were executed.")
        return 0

    max_workers = (
        int(max_workers_override)
        if max_workers_override
        else int(config.get("max_workers", max(1, os.cpu_count() or 1)))
    )
    max_workers = max(1, max_workers)
    logger.info("Using worker processes: %d", max_workers)

    results: List[RunResult] = []
    started = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(run_simulation, spec): spec for spec in run_specs}
        completed_count = 0

        for future in as_completed(future_map):
            spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = RunResult(
                    run_id=spec.run_id,
                    run_name=spec.run_name,
                    run_dir=spec.run_dir,
                    status="orchestrator_error",
                    return_code=None,
                    duration_s=0.0,
                    error=str(exc),
                    params=spec.params,
                    metrics={},
                )
            results.append(result)
            completed_count += 1

            logger.info(
                "Completed %d/%d | %s | status=%s rc=%s duration=%.2fs",
                completed_count,
                len(run_specs),
                result.run_name,
                result.status,
                result.return_code,
                result.duration_s,
            )

    elapsed = time.perf_counter() - started

    summary_csv = campaign_dir / "summary.csv"
    write_summary_csv(results=results, out_path=summary_csv)

    failures = [res for res in results if res.status != "success"]
    failed_runs_file = campaign_dir / "failed_runs.txt"
    if failures:
        failed_runs_file.write_text(
            "\n".join(f"{res.run_name},{res.status},{res.error}" for res in failures)
            + "\n",
            encoding="utf-8",
        )
    else:
        failed_runs_file.write_text("", encoding="utf-8")

    logger.info("Campaign finished in %.2fs", elapsed)
    logger.info("Summary CSV: %s", summary_csv)
    logger.info("Failed runs: %d", len(failures))

    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()

    try:
        config = load_config(config_path)
        validate_config(config)
        return run_campaign(
            config=config,
            config_path=config_path,
            max_workers_override=args.max_workers,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
