import json
from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path("resultados_generados_bahia_criterio_ii_p68/metrics/metrics_magdalena.csv")

# Datos de hardware/software que NO vienen completos en el CSV
SYSTEM_INFO = {
    "cpu": "AMD Ryzen 7 7730U processor",
    "cores": 8,
    "threads_hw": 16,
    "clock": "2.00 GHz",
    "ram": "16 GB",
    "os": "Windows"
}

# Filtros recomendados para quedarte solo con el experimento final reportado
FILTERS = {
    "modelo": "magdalena",
    "fase": "final",
    "participacion": 68,   # usa None si no quieres filtrar por participación
}

# Si después de filtrar quedan filas repetidas por semana, conserva la última
KEEP_ONE_ROW_PER_WEEK = True

# Archivos de salida
OUT_DIR = CSV_PATH.parent
TXT_OUT = OUT_DIR / "computational_summary_and_paragraphs.txt"
JSON_OUT = OUT_DIR / "computational_summary_stats.json"
CSV_OUT = OUT_DIR / "computational_summary_by_week.csv"

# =========================================================
# HELPERS
# =========================================================
def fmt_int(x):
    if pd.isna(x):
        return "NA"
    return f"{int(round(x)):,}"

def fmt_float(x, nd=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{nd}f}"

def fmt_pct(x, nd=1):
    if pd.isna(x):
        return "NA"
    return f"{100*float(x):.{nd}f}%"

def mode_or_join(series):
    s = series.dropna().astype(str)
    if s.empty:
        return "NA"
    modes = s.mode()
    if len(modes) == 1:
        return modes.iloc[0]
    return ", ".join(sorted(s.unique()))

def series_stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {
            "min": None, "max": None, "mean": None, "median": None, "std": None,
            "q25": None, "q75": None, "sum": None
        }
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "sum": float(s.sum())
    }

def boolify(col):
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })

def build_range_text(s, decimals=0):
    st = series_stats(s)
    if st["min"] is None:
        return "NA"
    if decimals == 0:
        mn = fmt_int(st["min"])
        mx = fmt_int(st["max"])
        av = fmt_float(st["mean"], 1)
    else:
        mn = fmt_float(st["min"], decimals)
        mx = fmt_float(st["max"], decimals)
        av = fmt_float(st["mean"], decimals)
    return f"{mn} to {mx} (avg. {av})"

# =========================================================
# LOAD
# =========================================================
if not CSV_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Normalización de tipos
if "semana" in df.columns:
    df["semana"] = pd.to_datetime(df["semana"], errors="coerce")

for c in ["usar_cota_inferior", "usar_cota_superior"]:
    if c in df.columns:
        df[c] = boolify(df[c])

numeric_cols = [
    "participacion", "beta_alpha", "gamma_val", "build_seconds", "threads", "mip_gap",
    "node_count", "vars_total", "vars_bin", "vars_int", "vars_cont", "constr_total",
    "params_total", "size_B", "size_S", "size_T", "solve_seconds", "obj_value"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================================================
# FILTERS
# =========================================================
fdf = df.copy()

for col, val in FILTERS.items():
    if val is not None and col in fdf.columns:
        fdf = fdf[fdf[col] == val]

if KEEP_ONE_ROW_PER_WEEK and "semana" in fdf.columns:
    fdf = (
        fdf.sort_values(["semana"])
           .drop_duplicates(subset=["semana"], keep="last")
           .reset_index(drop=True)
    )

if fdf.empty:
    raise ValueError("Después de aplicar los filtros no quedaron filas. Revisa FILTERS.")

# =========================================================
# CORE STATS
# =========================================================
n_rows = len(fdf)
week_min = fdf["semana"].min() if "semana" in fdf.columns else None
week_max = fdf["semana"].max() if "semana" in fdf.columns else None

status_counts = fdf["solver_status"].value_counts(dropna=False).to_dict() if "solver_status" in fdf.columns else {}
term_counts = fdf["termination"].value_counts(dropna=False).to_dict() if "termination" in fdf.columns else {}

optimal_mask = (
    fdf["termination"].astype(str).str.lower().eq("optimal")
    if "termination" in fdf.columns else pd.Series([False] * len(fdf))
)
n_optimal = int(optimal_mask.sum())
optimal_rate = n_optimal / n_rows if n_rows > 0 else np.nan

# Tiempos
build_stats = series_stats(fdf["build_seconds"]) if "build_seconds" in fdf.columns else {}
solve_stats = series_stats(fdf["solve_seconds"]) if "solve_seconds" in fdf.columns else {}

# Tamaño de instancias
sizeB_stats = series_stats(fdf["size_B"]) if "size_B" in fdf.columns else {}
sizeS_stats = series_stats(fdf["size_S"]) if "size_S" in fdf.columns else {}
sizeT_stats = series_stats(fdf["size_T"]) if "size_T" in fdf.columns else {}

# Tamaño del modelo
vars_stats = series_stats(fdf["vars_total"]) if "vars_total" in fdf.columns else {}
bin_stats = series_stats(fdf["vars_bin"]) if "vars_bin" in fdf.columns else {}
int_stats = series_stats(fdf["vars_int"]) if "vars_int" in fdf.columns else {}
cont_stats = series_stats(fdf["vars_cont"]) if "vars_cont" in fdf.columns else {}
constr_stats = series_stats(fdf["constr_total"]) if "constr_total" in fdf.columns else {}
params_stats = series_stats(fdf["params_total"]) if "params_total" in fdf.columns else {}
nodes_stats = series_stats(fdf["node_count"]) if "node_count" in fdf.columns else {}
gap_stats = series_stats(fdf["mip_gap"]) if "mip_gap" in fdf.columns else {}
obj_stats = series_stats(fdf["obj_value"]) if "obj_value" in fdf.columns else {}

gurobi_versions = mode_or_join(fdf["gurobi_version"]) if "gurobi_version" in fdf.columns else "NA"
threads_used = mode_or_join(fdf["threads"]) if "threads" in fdf.columns else "NA"
participation_values = ", ".join(sorted(map(str, fdf["participacion"].dropna().unique()))) if "participacion" in fdf.columns else "NA"

# =========================================================
# RESUMEN TABULAR POR SEMANA
# =========================================================
cols_week = [
    "semana", "size_B", "size_S", "size_T", "vars_total", "constr_total",
    "build_seconds", "solve_seconds", "solver_status", "termination", "mip_gap", "obj_value"
]
cols_week = [c for c in cols_week if c in fdf.columns]
week_table = fdf[cols_week].copy()
week_table.to_csv(CSV_OUT, index=False)

# =========================================================
# TEXTO: RESUMEN Y PÁRRAFOS
# =========================================================
summary_lines = []

summary_lines.append("============================================================")
summary_lines.append("COMPUTATIONAL SUMMARY")
summary_lines.append("============================================================")
summary_lines.append(f"Instances analyzed: {n_rows}")
if week_min is not None and pd.notna(week_min):
    summary_lines.append(f"Weeks covered: {week_min.date()} to {week_max.date()}")
summary_lines.append(f"Model: {FILTERS.get('modelo')}")
summary_lines.append(f"Phase: {FILTERS.get('fase')}")
summary_lines.append(f"Participation filter: {participation_values}")
summary_lines.append("")
summary_lines.append("Solver execution")
summary_lines.append(f"  - Gurobi version(s): {gurobi_versions}")
summary_lines.append(f"  - Threads reported: {threads_used}")
summary_lines.append(f"  - Optimal solutions: {n_optimal}/{n_rows} ({fmt_pct(optimal_rate, 1)})")
summary_lines.append(f"  - Solver status counts: {status_counts}")
summary_lines.append(f"  - Termination counts: {term_counts}")
summary_lines.append("")
summary_lines.append("Time statistics [s]")
summary_lines.append(
    f"  - Build time: min {fmt_float(build_stats['min'],2)}, "
    f"avg {fmt_float(build_stats['mean'],2)}, median {fmt_float(build_stats['median'],2)}, "
    f"max {fmt_float(build_stats['max'],2)}, total {fmt_float(build_stats['sum'],2)}"
)
summary_lines.append(
    f"  - Solve time: min {fmt_float(solve_stats['min'],2)}, "
    f"avg {fmt_float(solve_stats['mean'],2)}, median {fmt_float(solve_stats['median'],2)}, "
    f"max {fmt_float(solve_stats['max'],2)}, total {fmt_float(solve_stats['sum'],2)}"
)
summary_lines.append("")
summary_lines.append("Instance size")
summary_lines.append(
    f"  - Blocks (size_B): min {fmt_int(sizeB_stats['min'])}, "
    f"avg {fmt_float(sizeB_stats['mean'],1)}, max {fmt_int(sizeB_stats['max'])}"
)
summary_lines.append(
    f"  - Segregations (size_S): min {fmt_int(sizeS_stats['min'])}, "
    f"avg {fmt_float(sizeS_stats['mean'],1)}, max {fmt_int(sizeS_stats['max'])}"
)
summary_lines.append(
    f"  - Time periods (size_T): min {fmt_int(sizeT_stats['min'])}, "
    f"avg {fmt_float(sizeT_stats['mean'],1)}, max {fmt_int(sizeT_stats['max'])}"
)
summary_lines.append("")
summary_lines.append("Model size")
summary_lines.append(
    f"  - Total variables: min {fmt_int(vars_stats['min'])}, "
    f"avg {fmt_float(vars_stats['mean'],1)}, max {fmt_int(vars_stats['max'])}"
)
summary_lines.append(
    f"  - Binary variables: min {fmt_int(bin_stats['min'])}, "
    f"avg {fmt_float(bin_stats['mean'],1)}, max {fmt_int(bin_stats['max'])}"
)
summary_lines.append(
    f"  - Integer variables: min {fmt_int(int_stats['min'])}, "
    f"avg {fmt_float(int_stats['mean'],1)}, max {fmt_int(int_stats['max'])}"
)
summary_lines.append(
    f"  - Continuous variables: min {fmt_int(cont_stats['min'])}, "
    f"avg {fmt_float(cont_stats['mean'],1)}, max {fmt_int(cont_stats['max'])}"
)
summary_lines.append(
    f"  - Constraints: min {fmt_int(constr_stats['min'])}, "
    f"avg {fmt_float(constr_stats['mean'],1)}, max {fmt_int(constr_stats['max'])}"
)
summary_lines.append(
    f"  - Parameters: min {fmt_int(params_stats['min'])}, "
    f"avg {fmt_float(params_stats['mean'],1)}, max {fmt_int(params_stats['max'])}"
)
summary_lines.append("")
summary_lines.append("Optimization quality")
summary_lines.append(
    f"  - MIP gap: min {fmt_pct(gap_stats['min'],4)}, "
    f"avg {fmt_pct(gap_stats['mean'],4)}, max {fmt_pct(gap_stats['max'],4)}"
)
summary_lines.append(
    f"  - Node count: min {fmt_int(nodes_stats['min'])}, "
    f"avg {fmt_float(nodes_stats['mean'],1)}, max {fmt_int(nodes_stats['max'])}"
)
summary_lines.append(
    f"  - Objective value: min {fmt_float(obj_stats['min'],1)}, "
    f"avg {fmt_float(obj_stats['mean'],1)}, max {fmt_float(obj_stats['max'],1)}"
)

# ---------------------------------------------------------
# Párrafo 1: Hardware + software + desempeño computacional
# ---------------------------------------------------------
paragraph_hw_sw = (
    f"The computational experiments were carried out on a {SYSTEM_INFO['os']}-based machine "
    f"equipped with an {SYSTEM_INFO['cpu']} ({SYSTEM_INFO['cores']} cores, "
    f"{SYSTEM_INFO['threads_hw']} hardware threads, {SYSTEM_INFO['clock']}) and "
    f"{SYSTEM_INFO['ram']} of RAM. The MILP model was implemented in Python and solved with "
    f"Gurobi {gurobi_versions}. For the final benchmark set of {n_rows} weekly instances, "
    f"the solver reported {threads_used} working threads and proved optimality in "
    f"{n_optimal} out of {n_rows} instances ({fmt_pct(optimal_rate,1)}). "
    f"Model-building times ranged from {fmt_float(build_stats['min'],2)} to {fmt_float(build_stats['max'],2)} s "
    f"(average {fmt_float(build_stats['mean'],2)} s), while solving times ranged from "
    f"{fmt_float(solve_stats['min'],2)} to {fmt_float(solve_stats['max'],2)} s "
    f"(average {fmt_float(solve_stats['mean'],2)} s, median {fmt_float(solve_stats['median'],2)} s)."
)

# ---------------------------------------------------------
# Párrafo 2: escala computacional / tamaño de instancias
# ---------------------------------------------------------
paragraph_scale = (
    f"The reported instances correspond to {n_rows} weekly scenarios covering the period from "
    f"{week_min.date()} to {week_max.date()}. After the selection procedure with participation "
    f"level {participation_values}%, each instance contained between {fmt_int(sizeS_stats['min'])} and "
    f"{fmt_int(sizeS_stats['max'])} segregations (average {fmt_float(sizeS_stats['mean'],1)}), "
    f"with {fmt_int(sizeB_stats['min'])} to {fmt_int(sizeB_stats['max'])} storage blocks and "
    f"{fmt_int(sizeT_stats['min'])} to {fmt_int(sizeT_stats['max'])} time periods. "
    f"These instances produced MILP formulations with between {fmt_int(vars_stats['min'])} and "
    f"{fmt_int(vars_stats['max'])} variables (average {fmt_float(vars_stats['mean'],1)}) and between "
    f"{fmt_int(constr_stats['min'])} and {fmt_int(constr_stats['max'])} constraints "
    f"(average {fmt_float(constr_stats['mean'],1)}), which shows that the proposed approach remains "
    f"computationally tractable for year-long weekly experimentation under realistic operating conditions."
)

# ---------------------------------------------------------
# Párrafo 3: versión más compacta, por si ITOR te aprieta espacio
# ---------------------------------------------------------
paragraph_compact = (
    f"The model was implemented in Python and solved with Gurobi {gurobi_versions} on a "
    f"{SYSTEM_INFO['os']} machine with an {SYSTEM_INFO['cpu']} "
    f"({SYSTEM_INFO['cores']} cores, {SYSTEM_INFO['threads_hw']} threads, {SYSTEM_INFO['clock']}) and "
    f"{SYSTEM_INFO['ram']} of RAM. The final experiment set comprises {n_rows} weekly real-data instances, "
    f"each with {fmt_int(sizeB_stats['min'])}-{fmt_int(sizeB_stats['max'])} blocks, "
    f"{fmt_int(sizeT_stats['min'])}-{fmt_int(sizeT_stats['max'])} time periods, and "
    f"{fmt_int(sizeS_stats['min'])}-{fmt_int(sizeS_stats['max'])} segregations "
    f"(average {fmt_float(sizeS_stats['mean'],1)}). The corresponding MILP models contain "
    f"{fmt_int(vars_stats['min'])}-{fmt_int(vars_stats['max'])} variables and "
    f"{fmt_int(constr_stats['min'])}-{fmt_int(constr_stats['max'])} constraints, and were solved in "
    f"{fmt_float(solve_stats['mean'],2)} s on average (maximum {fmt_float(solve_stats['max'],2)} s), "
    f"with {n_optimal}/{n_rows} instances solved to proven optimality."
)

# =========================================================
# SAVE OUTPUTS
# =========================================================
stats_json = {
    "filters": FILTERS,
    "system_info": SYSTEM_INFO,
    "n_instances": n_rows,
    "week_min": None if pd.isna(week_min) else str(week_min.date()),
    "week_max": None if pd.isna(week_max) else str(week_max.date()),
    "gurobi_versions": gurobi_versions,
    "threads_used": threads_used,
    "optimal_instances": n_optimal,
    "optimal_rate": optimal_rate,
    "status_counts": status_counts,
    "termination_counts": term_counts,
    "build_seconds": build_stats,
    "solve_seconds": solve_stats,
    "size_B": sizeB_stats,
    "size_S": sizeS_stats,
    "size_T": sizeT_stats,
    "vars_total": vars_stats,
    "vars_bin": bin_stats,
    "vars_int": int_stats,
    "vars_cont": cont_stats,
    "constr_total": constr_stats,
    "params_total": params_stats,
    "mip_gap": gap_stats,
    "node_count": nodes_stats,
    "obj_value": obj_stats,
    "paragraph_hw_sw": paragraph_hw_sw,
    "paragraph_scale": paragraph_scale,
    "paragraph_compact": paragraph_compact
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(stats_json, f, indent=2, ensure_ascii=False)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
    f.write("\n\n============================================================\n")
    f.write("SUGGESTED PAPER PARAGRAPHS\n")
    f.write("============================================================\n\n")
    f.write("[Paragraph A: Hardware and Software]\n")
    f.write(paragraph_hw_sw + "\n\n")
    f.write("[Paragraph B: Computational scale / instance generation]\n")
    f.write(paragraph_scale + "\n\n")
    f.write("[Paragraph C: Compact version]\n")
    f.write(paragraph_compact + "\n")

# =========================================================
# PRINT TO CONSOLE
# =========================================================
print("\n".join(summary_lines))
print("\n" + "="*60)
print("PARAGRAPH A: HARDWARE AND SOFTWARE")
print("="*60)
print(paragraph_hw_sw)

print("\n" + "="*60)
print("PARAGRAPH B: COMPUTATIONAL SCALE / INSTANCE GENERATION")
print("="*60)
print(paragraph_scale)

print("\n" + "="*60)
print("PARAGRAPH C: COMPACT VERSION")
print("="*60)
print(paragraph_compact)

print("\nArchivos generados:")
print(f" - {TXT_OUT}")
print(f" - {JSON_OUT}")
print(f" - {CSV_OUT}")