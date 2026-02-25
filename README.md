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

## Demonstrated Results (Committed Sample)

The repository includes two committed campaign snapshots to demonstrate real output artifacts:

- `output/tcad_dummy_sweep_20260209_223728/` (YAML config)
- `output/tcad_dummy_sweep_json_20260209_223716/` (JSON config)

| Campaign | Total Runs | Success | Failed | Key Artifacts |
| --- | ---: | ---: | ---: | --- |
| `tcad_dummy_sweep_20260209_223728` | 48 | 44 | 4 | `summary.csv`, `failed_runs.txt`, `orchestrator.log`, `config_snapshot.json` |
| `tcad_dummy_sweep_json_20260209_223716` | 12 | 12 | 0 | `summary.csv`, `orchestrator.log`, `config_snapshot.json` |

Quick verification commands:

```bash
# YAML campaign stats
awk -F, 'NR>1{t++; if($3=="success") s++; else if($3=="failed") f++} END{printf("runs=%d success=%d failed=%d\n",t,s,f)}' \
  output/tcad_dummy_sweep_20260209_223728/summary.csv

# JSON campaign stats
awk -F, 'NR>1{t++; if($3=="success") s++; else if($3=="failed") f++} END{printf("runs=%d success=%d failed=%d\n",t,s,f)}' \
  output/tcad_dummy_sweep_json_20260209_223716/summary.csv

# Show failed run IDs (YAML campaign)
cat output/tcad_dummy_sweep_20260209_223728/failed_runs.txt
```

## Example CV Bullet

Developed Python-based batch automation framework for parameterized simulation runs, including job orchestration, logging, and result aggregation in Linux environments.
