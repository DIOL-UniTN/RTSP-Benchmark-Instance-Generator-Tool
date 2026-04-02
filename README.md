# RTSP Benchmark Instance Management Tool

A Python toolkit for generating, validating, and solving **Radiotherapy Scheduling Problem (RTSP)** benchmark instances.

---

## Project Structure

```
RTSP_management_tool/
├── main.py                    ← Unified entry point (start here)
├── generator.py               ← Instance generator (called by main.py)
├── validator.py               ← Solution feasibility checker
├── generator_parameter.json   ← Global generation parameters (protocols, machines, weights)
│
├── gen_input_params/          ← Per-dataset configuration files
│   ├── dataset_1_7_2.json
│   ├── dataset_6_10_4.json
│   └── ...
│
├── instances/                 ← Generated instances (.pk files, created at runtime)
├── solutions/                 ← Generated / computed solutions (.pk files, created at runtime)
│
├── data_structure/
│   ├── Input.py
│   ├── Solution.py
│   └── Fitness.py
│
└── solvers/
    ├── heuristics/
    │   └── heuristics.py      ← Greedy FirstFit heuristic
    ├── simulated_annealing/
    │   └── SimulatedAnnealing.py
    └── cplex_milp/
        └── batch_milp.py      ← Requires IBM CPLEX
```

---

## Requirements

```bash
pip install numpy
pip install pickle
# For the MILP solver only:
pip install cplex  # requires an IBM CPLEX installation
```

---

## Quick Start

All functionality is accessible through `main.py`. Run it from inside the project directory:

```bash
cd RTSP_management_tool
python main.py <command> [options]
```

---

## Commands

### 1. `generate` – Create benchmark instances

Reads parameters from JSON files in `gen_input_params/` and writes instances + initial solutions to `instances/` and `solutions/`.

```bash
# Generate instances for ALL datasets in gen_input_params/
python main.py generate

# Generate only a specific dataset
python main.py generate --params dataset_1_7_2.json

# Generate all datasets matching a pattern
python main.py generate --params "dataset_6_*.json"

# Use a custom global parameter file
python main.py generate --gen-params path/to/my_generator_parameter.json
```

Output directories follow the pattern:
```
instances/monthly/<dataset_title>/<arrival_mean>_<time_windows>/
solutions/FirstFit/monthly/<dataset_title>/<arrival_mean>_<time_windows>/
```

---

### 2. `validate` – Check solution feasibility

```bash
python main.py validate \
    --instance  instances/monthly/new_dataset_1/7_2/0.pk \
    --solution  solutions/FirstFit/monthly/new_dataset_1/7_2/0.pk
```

Prints `FEASIBLE` or `NOT feasible`.

---

### 3. `solve` – Run a solver on an existing instance

#### Greedy Heuristic (FirstFit) – fast, no dependencies
```bash
python main.py solve \
    --solver   heuristic \
    --instance instances/monthly/new_dataset_1/7_2/0.pk \
    --output   solutions/my_run/0.pk
```

#### Simulated Annealing – slower, higher quality
```bash
python main.py solve \
    --solver          sa \
    --instance        instances/monthly/new_dataset_1/7_2/0.pk \
    --solution        solutions/FirstFit/monthly/new_dataset_1/7_2/0.pk \
    --neighbour-sizes 5,5,5,5 \
    --output          solutions/sa_run/0.pk
```

#### MILP (CPLEX) – exact solver, requires IBM CPLEX
```bash
python main.py solve \
    --solver   milp \
    --instance instances/monthly/new_dataset_1/7_2/0.pk
```

**Solver names accepted:** `heuristic`, `sa`, `milp` (case-insensitive).

Common options:

| Option | Description |
|---|---|
| `--instance FILE` | Path to the input instance `.pk` file (**required**) |
| `--output FILE` | Where to save the solution (default: `solutions/<name>_solution.pk`) |
| `--obj N` | Objective weights index in `generator_parameter.json` (default: `1`) |

---

### 4. `list-datasets` – Inspect available configs

```bash
python main.py list-datasets
```

Prints a summary table of all JSON files in `gen_input_params/`.

---

### 5. `list-solvers` – See available solvers

```bash
python main.py list-solvers
```

---

## Dataset Config File Format

Each JSON file in `gen_input_params/` describes one dataset to generate. All fields are required unless marked optional.

```jsonc
{
    "dataset_title": "new_dataset_1",      // Folder name used when saving instances
    "instances_number": 50,                // How many instances to generate
    "seed": 198743,                        // Random seed for reproducibility
    "machines_number": 6,                  // Number of treatment machines
    "time_windows_number": 2,             // Time windows per machine per day
    "capacities": [240, 240, ...],        // Capacity per (machine × time_window); length = machines × TW
    "initial_occupation_percentage": [...],// Initial occupancy fraction per slot (0.35–0.90 recommended)
    "occupation_decay_velocity": [...],   // Decay speed per slot (0 = slow, 1 = fast)
    "arrival_mean": 7,                    // Poisson mean for daily patient arrivals
    "priorities_percentage": [0.3, 0.3, 0.4], // Fraction of patients per priority level (must sum to 1)
    "min_beam_matched_sets": 0,           // (optional) Minimum complete beam-matched machine sets
    "min_partial_beam_matched_sets": 0    // (optional) Minimum partial beam-matched machine sets
}
```

> **Tip:** `capacities`, `initial_occupation_percentage`, and `occupation_decay_velocity` must each have length `machines_number × time_windows_number`.

---

## Global Parameter File (`generator_parameter.json`)

Contains statistical parameters for machine occupancy modelling, patient protocol definitions, and objective function weights. Edit with care — these values are calibrated to produce realistic instances.

Key sections:

| Section | Description |
|---|---|
| `machines` | Regression coefficients for occupancy curve (`beta_1_min/max`, `beta_2_mean`, `k_d_median`) |
| `patients.protocols_info` | Protocol definitions: duration, fractions, allowed weekdays, priority |
| `patients.protocols_probabilities` | Per-priority protocol selection probabilities |
| `weightsAlpha` | Objective function weights indexed by integer key |

---

## Reproducibility

Set the same `seed` in the dataset config file to obtain identical instances across runs.

---

## Adding a New Dataset

1. Copy an existing file from `gen_input_params/`, e.g.:
   ```bash
   cp gen_input_params/dataset_1_7_2.json gen_input_params/my_experiment.json
   ```
2. Edit the fields as needed (remember to update `dataset_title` and `seed`).
3. Run:
   ```bash
   python main.py generate --params my_experiment.json
   ```
