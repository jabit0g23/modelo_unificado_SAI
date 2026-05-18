"""
Estadísticas sobre la agrupación RUMA para el paper.
Computa, sobre las 51 instancias finales:
- % segregaciones que cayeron en RUMA vs individuales
- % movimientos cubiertos por segs RUMA
- Distribución de bahías ocupadas por RUMA vs individuales
- Justificación del umbral=20
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent / "resultados" / "pipeline_bahia_criterio_iii_7d_theta1.4_test20"
INST_DIR = BASE / "instancias_coloracion"

records = []
for week_dir in sorted(INST_DIR.iterdir()):
    fecha = week_dir.name
    inst_file = week_dir / f"Instancia_{fecha}_K.xlsx"
    if not inst_file.exists():
        continue

    xl = pd.ExcelFile(inst_file)
    seg_det = xl.parse("Segregaciones_Detalle")
    s_sheet = xl.parse("S")

    # Segs activas
    seg = seg_det[seg_det["En_S"] == True].copy()

    # Cruce con S para distinguir RUMA de individuales (los nombres RUMA están en S)
    ruma_set = set(s_sheet[s_sheet["Segregacion"].str.endswith("-RUMA", na=False)]["Segregacion"])
    indiv_set = set(s_sheet["Segregacion"]) - ruma_set

    seg["es_ruma"] = seg["Segregacion"].isin(ruma_set)
    seg["movs"] = seg[["RECV", "LOAD", "DSCH", "DLVR"]].sum(axis=1)

    n_total = len(seg)
    n_ruma = int(seg["es_ruma"].sum())
    n_indiv = n_total - n_ruma

    movs_total = int(seg["movs"].sum())
    movs_ruma = int(seg.loc[seg["es_ruma"], "movs"].sum())
    movs_indiv = movs_total - movs_ruma

    inv_ruma = int(seg.loc[seg["es_ruma"], "Inventario_Inicial"].sum())
    inv_total = int(seg["Inventario_Inicial"].sum())

    # cuántas segs ORIGINALES se colapsaron en RUMA
    # n_ruma_grupos = cantidad de grupos RUMA
    # n_originales_en_ruma = no lo tengo directamente; pero del RUMA analysis xlsx ya está
    records.append({
        "fecha": fecha,
        "n_total_modelo": n_total,
        "n_indiv": n_indiv,
        "n_grupos_ruma": n_ruma,
        "movs_total": movs_total,
        "movs_ruma": movs_ruma,
        "movs_indiv": movs_indiv,
        "pct_movs_ruma": round(100 * movs_ruma / movs_total, 2) if movs_total > 0 else 0,
        "inv_ruma": inv_ruma,
        "inv_total": inv_total,
        "pct_inv_ruma": round(100 * inv_ruma / inv_total, 2) if inv_total > 0 else 0,
    })

df = pd.DataFrame(records)
print(f"Semanas procesadas: {len(df)}\n")

# ── Datos del análisis RUMA bahías ya generado ─────────────────────────────────
ruma_xlsx = pd.read_excel(Path(__file__).parent / "analisis_ruma_bahias.xlsx", sheet_name="Detalle")

# n_segs_en_ruma_avg por semana (cuántas segs originales fueron absorbidas por RUMA)
agg_orig = ruma_xlsx.groupby("semana").agg(
    n_originales_en_ruma=("n_segs_en_ruma", "sum"),
    n_grupos_ruma=("seg_ruma", "count"),
    n_indiv_total=("n_segs_indiv", "sum"),  # ojo: este es por prefijo, suma sobre prefijos
).reset_index()

print("=== AGRUPACIÓN RUMA (umbral=20) ===")
print(f"Grupos RUMA por semana: avg={df.n_grupos_ruma.mean():.1f}, min={df.n_grupos_ruma.min()}, max={df.n_grupos_ruma.max()}")
print(f"Segs originales absorbidas por RUMA: avg={agg_orig.n_originales_en_ruma.mean():.1f}, "
      f"min={agg_orig.n_originales_en_ruma.min()}, max={agg_orig.n_originales_en_ruma.max()}")

# % colapso de segregaciones
df = df.merge(agg_orig[["semana", "n_originales_en_ruma"]], left_on="fecha", right_on="semana", how="left").drop(columns=["semana"])
df["n_originales_total"] = df["n_indiv"] + df["n_originales_en_ruma"]
df["pct_segs_a_ruma"] = round(100 * df["n_originales_en_ruma"] / df["n_originales_total"], 1)

print(f"\nSegs originales pre-agrupación: avg={df.n_originales_total.mean():.1f} "
      f"(min={df.n_originales_total.min()}, max={df.n_originales_total.max()})")
print(f"Segs en modelo post-agrupación: avg={df.n_total_modelo.mean():.1f} "
      f"(min={df.n_total_modelo.min()}, max={df.n_total_modelo.max()})")
print(f"% segs originales que cayeron en RUMA: avg={df.pct_segs_a_ruma.mean():.1f}%, "
      f"min={df.pct_segs_a_ruma.min()}%, max={df.pct_segs_a_ruma.max()}%")

print(f"\n=== IMPACTO EN MOVIMIENTOS ===")
print(f"Movs en grupos RUMA: avg={df.movs_ruma.mean():.0f} de {df.movs_total.mean():.0f} totales")
print(f"% movs en RUMA: avg={df.pct_movs_ruma.mean():.2f}%, "
      f"min={df.pct_movs_ruma.min()}%, max={df.pct_movs_ruma.max()}%")
print(f"% inventario inicial en RUMA: avg={df.pct_inv_ruma.mean():.2f}%, "
      f"min={df.pct_inv_ruma.min()}%, max={df.pct_inv_ruma.max()}%")

# ── Bahías ocupadas: comparación RUMA vs individuales ─────────────────────────
print(f"\n=== OCUPACIÓN DE BAHÍAS (de scripts/analisis_ruma_bahias.xlsx) ===")
res = pd.read_excel(Path(__file__).parent / "analisis_ruma_bahias.xlsx", sheet_name="Resumen")
print(f"Bahías ocupadas avg por grupo RUMA (mean): {res.bah_ocu_avg_mean.mean():.2f}")
print(f"Bahías ocupadas avg por seg INDIVIDUAL (p50 reales): {res.p50_reales_avg.iloc[0]:.2f}")
print(f"Pico bahías por RUMA (p95): {res.bah_ocu_max_p95.mean():.2f}")

# ── Tabla por semana ──────────────────────────────────────────────────────────
print("\n=== TABLA RESUMIDA POR SEMANA ===")
print(df[["fecha", "n_originales_total", "n_total_modelo", "n_originales_en_ruma",
         "n_grupos_ruma", "pct_segs_a_ruma", "pct_movs_ruma", "pct_inv_ruma"]].to_string(index=False))
