import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "resultados"

EXPERIMENTOS = [
    "pipeline_bahia_criterio_iii_7d_theta1.2_test20",
    "pipeline_bahia_criterio_iii_7d_theta1.4_test20",
    "pipeline_bahia_criterio_iii_7d_theta1.6_test20",
]

dfs = []
for exp in EXPERIMENTOS:
    csv_path = BASE / exp / "metrics" / "metrics_coloracion.csv"
    if not csv_path.exists():
        print(f"No encontrado: {csv_path}")
        continue
    df = pd.read_csv(csv_path, usecols=["semana", "solve_seconds"])
    df = df.rename(columns={"solve_seconds": exp})
    dfs.append(df.set_index("semana"))

if not dfs:
    print("Ningún archivo encontrado.")
    exit(1)

result = pd.concat(dfs, axis=1)
result.index.name = "semana"

avg_row = result.mean(axis=0).rename("PROMEDIO")
result = pd.concat([result, avg_row.to_frame().T])

out = Path(__file__).parent / "tiempo_ejecucion_coloracion.xlsx"
result.to_excel(out)
print(f"Guardado en {out}")
print(result.tail(3))
