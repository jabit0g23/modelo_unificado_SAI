# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Optimization modeling framework for container terminal operations at a Chilean port (SAI). Implements two interdependent MIP (Mixed-Integer Programming) models solved via **Pyomo + Gurobi**:

- **Magdalena model** (`modelo_coloracion.py`): container allocation/coloring across storage blocks over a week
- **Camila model** (`modelo_gruas_maxmin.py`): crane scheduling per shift (maxmin or minmax objective)

## Running the Code

All configuration is at the top of `main.py` (lines 10–46). Edit parameters there, then run:

```bash
python main.py
```

Key parameters in `main.py`:
- `SEMANAS`: list of ISO week start dates (e.g. `["2022-01-03"]`) or use `USAR_RANGO` with `ISO_WEEK_INI`/`ISO_WEEK_FIN`
- `CRITERIO`: `"criterio_ii"` or `"criterio_iii"` — flow classification criterion
- `CAP_MODE`: `"pila"` (stack) or `"bahia"` (bay) — storage capacity mode
- `OBJETIVO_GRUAS`: `"maxmin"` or `"minmax"` — crane objective
- `EJECUTAR`: dict of booleans toggling each pipeline phase on/off

Individual analysis scripts can be run directly:
```bash
python a_metrics.py
python a_calcular_distancias_reales.py
python a_analisis_seg.py
```

## Dependencies

No `requirements.txt` exists. Required packages:
- `pyomo` — optimization modeling
- `pandas`, `numpy` — data processing
- `gurobipy` — Gurobi Python API (requires Gurobi installation + `gurobi.lic` in repo root)

## Pipeline Architecture

`main.py` orchestrates 5 sequential phases:

1. **Magdalena instances** (`instancias_coloracion.py`) — calls `codigos/analisis_flujos.py` and `codigos/evolucion_turnos.py`, then `codigos/instancias.py` to write `Instancia_*.xlsx` files
2. **Magdalena solve** (`modelo_coloracion.py`) — reads instances, builds and solves Pyomo ConcreteModel, writes `resultado_*.xlsx` and metrics CSV
3. **Export feasible weeks** — writes `semanas_filtradas.csv` and `semanas_infactibles.csv`
4. **Camila instances** (`instancias_gruas.py`) — generates per-shift `Instancia_*_T{01..21}.xlsx` files from Magdalena results
5. **Camila solve** (`modelo_gruas_maxmin.py`) — reads shift instances, solves crane scheduling MIP, appends metrics

## Directory Structure

```
archivos_estaticos/         # Input data (not tracked by git)
  {year}/{criterion}/       # Flow classification CSVs
  Distancias_GranPatio.xlsx # Distance matrix
  Flujos.csv                # Historical flow data (~1GB)

codigos/                    # Shared preprocessing modules
  analisis_flujos.py        # Flow normalization and shift assignment
  evolucion_turnos.py       # Shift-based data aggregation
  instancias.py             # Instance generation (capacity, demand, inventory)

resultados_generados_{CAP_MODE}_{CRITERIO}_{variant}/  # Output (not tracked)
  instancias_magdalena/{week}/   # Intermediate instance files
  resultados_magdalena/{week}/   # Magdalena solution files
  instancias_camila/             # Per-shift Camila instance files
  resultados_camila/             # Crane scheduling results
  metrics/                       # metrics_magdalena.csv, metrics_gruas.csv
```

## Key Implementation Notes

- All code and comments are in **Spanish**
- `util.py` provides model statistics helpers used by both optimization models
- The `.gitignore` excludes all `.csv`, `.xlsx`, `.log`, `.lp`, `.ilp`, `.iis` files — instance and result files are never committed
- Gurobi solver parameters (time limit, threads, MIP gap) are set directly in the model files before `solver.solve()`
- `instancias_gruas.py` reads Magdalena results to build crane instances — the two models are coupled through the output of phase 2
