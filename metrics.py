#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

BASE_FOLDERS = [
    "resultados_generados_bahia_criterio_ii",
    #"resultados_generados_bahia_criterio_iii",
    #"resultados_generados_pila_criterio_ii",
    #"resultados_generados_pila_criterio_iii",
]

MODELS = [
    ("magdalena", "metrics_magdalena.csv"),
    ("gruas", "metrics_gruas.csv"),
]

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _normalize_term(x):
    if pd.isna(x):
        return None
    s = str(x)
    # toma el último token después de un punto si viene con prefijo tipo "TerminationCondition.optimal"
    return s.split(".")[-1].strip().lower()

def _row_has_solution(row: pd.Series) -> bool:
    term = _normalize_term(row.get("termination"))
    obj  = row.get("obj_value")
    fase_ok = True if "fase" not in row.index else (row["fase"] == "final")
    cond_term = term in ("optimal", "feasible")
    cond_obj  = pd.notna(obj)
    return bool(fase_ok and (cond_term or cond_obj))

def summarize_metrics(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()

    # Tipos numéricos
    numeric_cols = [
        "vars_total","vars_bin","vars_int","vars_cont",
        "constr_total","params_total","solve_seconds","obj_value"
    ]
    _coerce_numeric(df, numeric_cols)
    for c in [c for c in df.columns if c.startswith("size_")]:
        _coerce_numeric(df, [c])

    # Semana a datetime si existe
    if "semana" in df.columns:
        df["_semana_dt"] = pd.to_datetime(df["semana"], errors="coerce")
    else:
        df["_semana_dt"] = pd.NaT

    # Flag de solución
    df["has_solution"] = df.apply(_row_has_solution, axis=1)

    # Subconjunto "final" si existe la columna fase
    if "fase" in df.columns:
        finals = df[df["fase"] == "final"].copy()
        if finals.empty:
            finals = df.copy()
    else:
        finals = df.copy()

    def num_stat(col, func):
        if col in finals.columns:
            series = finals[col].dropna()
            return func(series) if not series.empty else np.nan
        return np.nan

    out = {}
    out["model"] = model_name
    out["runs_total"] = int(len(df))
    out["runs_final_rows"] = int(len(finals))
    out["runs_with_solution"] = int(df["has_solution"].sum()) if "has_solution" in df.columns else 0
    out["success_rate_pct"] = round(100.0 * out["runs_with_solution"] / out["runs_total"], 2) if out["runs_total"] else np.nan

    if "fase" in df.columns:
        out["infeasible_rows"]   = int((df["fase"] == "infeasible").sum())
        out["no_incumbent_rows"] = int((df["fase"] == "no_incumbent").sum())
    else:
        if "termination" in df.columns:
            out["infeasible_rows"] = int(df["termination"].astype(str).str.contains("infeas", case=False, na=False).sum())
        else:
            out["infeasible_rows"] = 0
        out["no_incumbent_rows"] = 0

    out["weeks_covered"] = int(df["semana"].nunique()) if "semana" in df.columns else np.nan
    out["last_week"] = (
        finals["_semana_dt"].max().date().isoformat()
        if "_semana_dt" in finals.columns and finals["_semana_dt"].notna().any()
        else None
    )

    # Tiempos
    out["solve_sec_mean"] = float(num_stat("solve_seconds", np.mean))
    out["solve_sec_p50"]  = float(num_stat("solve_seconds", lambda s: s.quantile(0.50)))
    out["solve_sec_p95"]  = float(num_stat("solve_seconds", lambda s: s.quantile(0.95)))

    # Tamaños del modelo
    for col in ["vars_total","vars_bin","vars_int","vars_cont","constr_total","params_total"]:
        out[f"{col}_mean"]   = float(num_stat(col, np.mean))
        out[f"{col}_median"] = float(num_stat(col, lambda s: s.quantile(0.50)))

    # Sets promedio
    for col in [c for c in finals.columns if c.startswith("size_")]:
        out[f"{col}_mean"] = float(num_stat(col, np.mean))

    # Objetivo
    out["obj_min"] = float(num_stat("obj_value", np.min))
    out["obj_p50"] = float(num_stat("obj_value", lambda s: s.quantile(0.50)))
    out["obj_max"] = float(num_stat("obj_value", np.max))

    # Top-3 de terminaciones
    if "termination" in df.columns:
        top_terms = df["termination"].astype(str).str.lower().value_counts().head(3)
        for i, (k, v) in enumerate(top_terms.items(), start=1):
            out[f"term_{i}_name"]  = k
            out[f"term_{i}_count"] = int(v)

    return pd.DataFrame([out])

def main():
    base_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    combined_rows = []

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Base: {base_root}")

    for folder in BASE_FOLDERS:
        base_path = os.path.join(base_root, folder)
        metrics_dir = os.path.join(base_path, "metrics")
        if not os.path.isdir(base_path):
            print(f"  - {folder}: no existe, salto.")
            continue
        if not os.path.isdir(metrics_dir):
            print(f"  - {folder}: no tiene 'metrics/', salto.")
            continue

        print(f"  > Procesando {folder}/metrics")

        for model_name, csv_name in MODELS:
            csv_path = os.path.join(metrics_dir, csv_name)
            if not os.path.exists(csv_path):
                print(f"    · {model_name}: {csv_name} no encontrado, salto.")
                continue

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    print(f"    · {model_name}: CSV vacío.")
                    continue
                summary_df = summarize_metrics(df, model_name)
                summary_df.insert(0, "carpeta", folder)

                # Guardar resumen por modelo dentro del metrics de la carpeta
                out_path = os.path.join(metrics_dir, f"summary_{model_name}.csv")
                summary_df.to_csv(out_path, index=False)
                print(f"    · {model_name}: resumen -> {out_path}")

                combined_rows.append(summary_df)
            except Exception as e:
                print(f"    · {model_name}: error leyendo/sumando {csv_path}: {e}")

    # Overview combinado
    if combined_rows:
        overview = pd.concat(combined_rows, ignore_index=True)
        overview_dir = os.path.join(base_root, "summary")
        os.makedirs(overview_dir, exist_ok=True)
        ov_path = os.path.join(overview_dir, "overview_metrics_por_modelo.csv")
        overview.to_csv(ov_path, index=False)
        print(f"\nOverview global -> {ov_path}")

        # También un pivot rápido carpeta x modelo con métricas básicas
        piv = overview.pivot_table(
            index=["carpeta", "model"],
            values=[
                "runs_total", "runs_with_solution", "success_rate_pct",
                "solve_sec_mean", "vars_total_mean", "constr_total_mean", "params_total_mean",
                "obj_min","obj_p50","obj_max"
            ],
            aggfunc="first"
        ).reset_index()
        piv_path = os.path.join(overview_dir, "overview_basico_pivot.csv")
        piv.to_csv(piv_path, index=False)
        print(f"Overview básico (pivot) -> {piv_path}")
    else:
        print("\nNo se generó overview porque no hubo métricas válidas en las carpetas señaladas.")

if __name__ == "__main__":
    main()
