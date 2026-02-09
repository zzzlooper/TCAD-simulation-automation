# TCAD-Style Batch Simulation Orchestrator

Python mini-project that mimics real Linux TCAD batch workflows:

- Parameterized sweep definitions in YAML or JSON
- One run directory per job
- Parallel execution via multiprocessing + subprocess
- Per-run stdout/stderr capture
- Failure detection and campaign summary CSV output

## Project Layout

```text
.
├── batch_orchestrator.py
├── dummy_simulator.py
├── configs/
│   ├── sweep_demo.yaml
│   └── sweep_demo.json
├── requirements.txt
└── output/   # generated at runtime
```

## Quick Start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Run a YAML campaign:

```bash
python3 batch_orchestrator.py --config configs/sweep_demo.yaml
```

3. Run a JSON campaign:

```bash
python3 batch_orchestrator.py --config configs/sweep_demo.json
```

4. Dry-run (plan only):

```bash
python3 batch_orchestrator.py --config configs/sweep_demo.yaml --dry-run
```

## Config Schema

Required fields:

- `command`: string or token list, executable to launch per run
- `parameters`: mapping of parameter name to list of sweep values

Optional fields:

- `campaign_name`: output campaign prefix
- `output_root`: output parent directory
- `repeats`: repeat count for each parameter point
- `max_workers`: number of parallel worker processes
- `timeout_sec`: per-run timeout
- `static_args`: extra arguments added before sweep arguments
- `arg_map`: custom mapping from parameter names to flags

By default, parameters map to CLI flags as `--<param-name>` with `_` converted to `-`.

## Output Structure

Each run creates a folder:

```text
output/<campaign>_<timestamp>/
├── config_snapshot.json
├── orchestrator.log
├── summary.csv
├── failed_runs.txt
└── runs/
    ├── run_0001/
    │   ├── command.txt
    │   ├── params.json
    │   ├── status.json
    │   ├── stdout.log
    │   ├── stderr.log
    │   └── result.json    # if produced by simulator
    └── ...
```

## Example CV Bullet

Developed Python-based batch automation framework for parameterized simulation runs, including job orchestration, logging, and result aggregation in Linux environments.

