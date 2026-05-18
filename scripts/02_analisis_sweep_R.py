"""
Analiza los archivos R_Sweep_*.xlsx de un directorio de resultados
y encuentra la combinacion de (R_C, R_H, R_TI) que minimiza la
distancia total agregada entre todas las semanas.
"""

import pandas as pd
import glob
import os
import sys

RESULTADOS_DIR = "resultados/pipeline_pila_criterio_iii_7d_theta1.2_test1R/resultados_coloracion"

# ── carga ──────────────────────────────────────────────────────────────────────

archivos = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "**/R_Sweep_*.xlsx"), recursive=True))

if not archivos:
    print(f"No se encontraron archivos R_Sweep_*.xlsx en {RESULTADOS_DIR}")
    sys.exit(1)

dfs = []
for ruta in archivos:
    df = pd.read_excel(ruta)
    dfs.append(df)

datos = pd.concat(dfs, ignore_index=True)
semanas = datos["Semana"].unique()
n_semanas = len(semanas)
print(f"Semanas cargadas: {n_semanas}  ({', '.join(sorted(semanas))})")
print(f"Combinaciones posibles por semana: {datos.groupby('Semana').size().iloc[0]}")
print()

# ── analisis por combinacion ────────────────────────────────────────────────────

factibles = datos[datos["Status"] == "optimal"].copy()

resumen = (
    factibles
    .groupby(["R_C", "R_H", "R_TI"])
    .agg(
        semanas_factibles=("Distancia_Total", "count"),
        distancia_total=("Distancia_Total", "sum"),
        distancia_media=("Distancia_Total", "mean"),
        distancia_max=("Distancia_Total", "max"),
    )
    .reset_index()
)

# solo combinaciones factibles en TODAS las semanas
todas_factibles = resumen[resumen["semanas_factibles"] == n_semanas].copy()

print(f"Combinaciones factibles en todas las semanas: {len(todas_factibles)} / {len(resumen)}")
print()

# ── ranking ─────────────────────────────────────────────────────────────────────

todas_factibles = todas_factibles.sort_values("distancia_total").reset_index(drop=True)
todas_factibles.index += 1  # rank desde 1

print("=== Top 10 combinaciones (factibles en todas las semanas, por distancia total) ===")
cols = ["R_C", "R_H", "R_TI", "distancia_total", "distancia_media", "distancia_max"]
print(todas_factibles[cols].head(10).to_string(index=True))
print()

mejor = todas_factibles.iloc[0]
print(f">>> MEJOR combinacion global: R_C={int(mejor.R_C)}  R_H={int(mejor.R_H)}  R_TI={int(mejor.R_TI)}")
print(f"    Distancia total: {mejor.distancia_total:,.0f}")
print(f"    Distancia media por semana: {mejor.distancia_media:,.0f}")
print()

# ── detalle por semana para la mejor combinacion ────────────────────────────────

print("=== Detalle por semana (mejor combinacion global) ===")
mascara = (
    (factibles["R_C"] == mejor.R_C) &
    (factibles["R_H"] == mejor.R_H) &
    (factibles["R_TI"] == mejor.R_TI)
)
detalle = factibles[mascara][["Semana", "Distancia_Total"]].sort_values("Semana")
print(detalle.to_string(index=False))
print()

# ── mejor por semana (referencia) ───────────────────────────────────────────────

print("=== Mejor combinacion por semana (referencia, puede variar) ===")
idx_min = factibles.groupby("Semana")["Distancia_Total"].idxmin()
mejor_por_semana = factibles.loc[idx_min, ["Semana", "R_C", "R_H", "R_TI", "Distancia_Total"]].sort_values("Semana")
print(mejor_por_semana.to_string(index=False))
print()

# ── diferencia vs optimo por semana ─────────────────────────────────────────────

merged = mejor_por_semana.rename(columns={"Distancia_Total": "dist_optima_semana"}).merge(
    detalle.rename(columns={"Distancia_Total": "dist_global"}),
    on="Semana"
)
merged["diferencia"] = merged["dist_global"] - merged["dist_optima_semana"]
merged["diferencia_%"] = (merged["diferencia"] / merged["dist_optima_semana"] * 100).round(2)
print("=== Gap entre mejor global y optimo por semana ===")
print(merged[["Semana", "dist_optima_semana", "dist_global", "diferencia", "diferencia_%"]].to_string(index=False))

# ── exportar excel ───────────────────────────────────────────────────────────────

pivot = factibles.pivot_table(
    index=["R_C", "R_H", "R_TI"], columns="Semana", values="Distancia_Total"
).reset_index()
pivot.columns.name = None

todos = resumen.merge(pivot, on=["R_C", "R_H", "R_TI"]).sort_values("distancia_media").reset_index(drop=True)
todos.insert(0, "rank", todos.index + 1)
todos["factible_todas_semanas"] = todos["semanas_factibles"] == n_semanas

output_path = os.path.join(os.path.dirname(__file__), "analisis_sweep_R.xlsx")
todos.to_excel(output_path, index=False)
print(f"\nExcel guardado en: {output_path}")
