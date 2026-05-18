import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "resultados"

EXPERIMENTOS = [
    "pipeline_bahia_criterio_iii_7d_theta1.2_test20",
    "pipeline_bahia_criterio_iii_7d_theta1.4_test20",
    "pipeline_bahia_criterio_iii_7d_theta1.6_test20",
]

def cargar_conteos(exp_path):
    """Devuelve Series con frecuencia de cada n_bloques para todo el experimento."""
    semanas = sorted((exp_path / "resultados_coloracion").glob("*/"))
    all_values = []
    for semana in semanas:
        fecha = semana.name
        xlsx = semana / f"resultado_{fecha}_K.xlsx"
        if not xlsx.exists():
            continue
        df = pd.read_excel(xlsx, sheet_name="Total bloques")
        col = df.columns[1]  # "Total bloques asignadas"
        all_values.extend(df[col].tolist())

    if not all_values:
        return pd.Series(dtype=int)

    counts = pd.Series(all_values).value_counts().sort_index()
    counts.index.name = "n_bloques"
    return counts


out = Path(__file__).parent / "frecuencia_bloques_coloracion.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    resumen_frecs = {}

    for exp in EXPERIMENTOS:
        exp_path = BASE / exp
        if not exp_path.exists():
            print(f"No encontrado: {exp_path}")
            continue

        counts = cargar_conteos(exp_path)
        if counts.empty:
            print(f"Sin datos: {exp}")
            continue

        total = counts.sum()
        df_exp = pd.DataFrame({
            "n_bloques": counts.index,
            "frecuencia": counts.values,
            "porcentaje": (counts.values / total * 100).round(2),
        })

        # Una hoja por experimento (nombre truncado a 31 chars)
        sheet = exp[-31:]
        df_exp.to_excel(writer, sheet_name=sheet, index=False)
        resumen_frecs[exp] = counts

    # Hoja resumen: frecuencia relativa (%) por experimento lado a lado
    if resumen_frecs:
        all_idx = sorted(set().union(*[c.index for c in resumen_frecs.values()]))
        rows = []
        for n in all_idx:
            row = {"n_bloques": n}
            for exp, counts in resumen_frecs.items():
                total = counts.sum()
                row[exp] = round(counts.get(n, 0) / total * 100, 2)
            rows.append(row)
        df_resumen = pd.DataFrame(rows)
        df_resumen.to_excel(writer, sheet_name="Resumen_%", index=False)

        # Hoja con estadisticas generales por experimento
        stats_rows = []
        for exp, counts in resumen_frecs.items():
            vals = pd.Series(counts.index.repeat(counts.values))  # n_bloques expandido
            stats_rows.append({
                "experimento": exp,
                "media": round(vals.mean(), 3),
                "mediana": vals.median(),
                "moda": vals.mode().iloc[0],
                "max": vals.max(),
                "p25": vals.quantile(0.25),
                "p75": vals.quantile(0.75),
                "total_segregaciones": len(vals),
            })
        pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Estadisticas", index=False)

print(f"Guardado en {out}")
