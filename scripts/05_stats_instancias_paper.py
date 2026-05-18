"""
Calcula estadísticas de instancias para el paper.
Usa: resultados/pipeline_bahia_criterio_iii_7d_theta1.4_test20/
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent / "resultados" / "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20_90_base"
INST_DIR = BASE / "instancias_coloracion"
METRICS_FILE = BASE / "metrics" / "metrics_coloracion.csv"


def normalize_impo_segregation(seg_name):
    """
    Para segregaciones impo, remueve el 4to componente (ID de visita).
    expo-dry-20-EU236-ANR → expo-dry-20-EU236-ANR (sin cambio)
    impo-reefer-40-STI1328-INDIRECTO → impo-reefer-40-INDIRECTO
    impo-imo-20-EU236-DIRECTO → impo-imo-20-DIRECTO
    """
    if seg_name.startswith("expo-"):
        return seg_name
    elif seg_name.startswith("impo-"):
        parts = seg_name.split("-")
        if len(parts) >= 5:
            # Estructura: impo, tipo, tamaño, VISIT_ID, dirección
            # Queremos: impo, tipo, tamaño, dirección (remover VISIT_ID que es índice 3)
            return "-".join([parts[0], parts[1], parts[2], parts[4]])
        else:
            return seg_name
    else:
        return seg_name

# ── Cargar métricas del solver ─────────────────────────────────────────────────
metrics = pd.read_csv(METRICS_FILE)
metrics["semana"] = pd.to_datetime(metrics["semana"]).dt.strftime("%Y-%m-%d")

# ── Cargar instancias ─────────────────────────────────────────────────────────
records = []
for week_dir in sorted(INST_DIR.iterdir()):
    fecha = week_dir.name
    inst_file = week_dir / f"Instancia_{fecha}_K.xlsx"
    if not inst_file.exists():
        print(f"  [SKIP] {fecha}: no instancia K")
        continue

    # Leer segregaciones raw desde analisis_flujos (pre-agrupación)
    flujos_file = week_dir / f"analisis_flujos_w{fecha}_0.xlsx"
    if flujos_file.exists():
        flujos = pd.read_excel(flujos_file, sheet_name="FlujosAll_sbt")
        raw_segs_list = flujos["criterio"].unique()
        # Normalizar: remover visit ID de impo
        normalized_segs = set(normalize_impo_segregation(seg) for seg in raw_segs_list)
        n_raw_segs = len(normalized_segs)
    else:
        n_raw_segs = None

    xl = pd.ExcelFile(inst_file)
    seg = xl.parse("Segregaciones_Detalle")
    r_s = xl.parse("R_s")

    seg = seg[seg["En_S"] == True].copy()
    n_seg = len(seg)

    reefer_segs = set(r_s[r_s["R"] == 1]["Segregacion"])
    n_reefer = seg["Segregacion"].isin(reefer_segs).sum()

    # Primaria: tiene operación de nave (DSCH > 0 o LOAD > 0)
    seg["es_primaria"] = (seg["DSCH"] > 0) | (seg["LOAD"] > 0)
    n_primary = int(seg["es_primaria"].sum())
    n_secondary = n_seg - n_primary

    records.append({
        "fecha": fecha,
        "n_raw_segs": n_raw_segs,
        "n_seg": n_seg,
        "n_reefer": int(n_reefer),
        "pct_reefer": round(100 * n_reefer / n_seg, 1) if n_seg > 0 else 0,
        "n_primary": n_primary,
        "n_secondary": n_secondary,
        "RECV": int(seg["RECV"].sum()),
        "LOAD": int(seg["LOAD"].sum()),
        "DSCH": int(seg["DSCH"].sum()),
        "DLVR": int(seg["DLVR"].sum()),
        "total_mov": int(seg[["RECV", "LOAD", "DSCH", "DLVR"]].sum().sum()),
    })

df = pd.DataFrame(records)
print(f"Semanas con instancia K: {len(df)}")

# ── SEGREGACIONES RAW (pre-agrupación) ──────────────────────────────────────────
df_raw = df[df["n_raw_segs"].notna()]
if len(df_raw) > 0:
    print("\n=== SEGREGACIONES RAW (pre-agrupación, impo normalizadas) ===")
    print(f"  Total: min={int(df_raw.n_raw_segs.min())}, max={int(df_raw.n_raw_segs.max())}, avg={df_raw.n_raw_segs.mean():.1f}")

# ── SEGREGACIONES (post-agrupación) ────────────────────────────────────────────
print("\n=== SEGREGACIONES (post-agrupación) ===")
print(f"  Total: min={df.n_seg.min()}, max={df.n_seg.max()}, avg={df.n_seg.mean():.1f}")
print(f"  Reefer: min={df.n_reefer.min()}, max={df.n_reefer.max()}, avg_pct={df.pct_reefer.mean():.1f}%")
print(f"  Primarias:   min={df.n_primary.min()}, max={df.n_primary.max()}, avg={df.n_primary.mean():.1f}")
print(f"  Secundarias: min={df.n_secondary.min()}, max={df.n_secondary.max()}, avg={df.n_secondary.mean():.1f}")
print(f"  Semana con más segregaciones:   {df.loc[df.n_seg.idxmax(), 'fecha']} ({df.n_seg.max()})")
print(f"  Semana con menos segregaciones: {df.loc[df.n_seg.idxmin(), 'fecha']} ({df.n_seg.min()})")

# ── FLUJOS ─────────────────────────────────────────────────────────────────────
print("\n=== FLUJOS POR TIPO (boxes/semana) ===")
for col in ["RECV", "LOAD", "DSCH", "DLVR"]:
    mn, mx, av = df[col].min(), df[col].max(), df[col].mean()
    week_max = df.loc[df[col].idxmax(), "fecha"]
    week_min = df.loc[df[col].idxmin(), "fecha"]
    print(f"  {col}: min={mn} ({week_min}), max={mx} ({week_max}), avg={av:.0f}")

# ── ACTIVIDAD TOTAL ────────────────────────────────────────────────────────────
print("\n=== ACTIVIDAD TOTAL ===")
print(f"  Semana más activa:   {df.loc[df.total_mov.idxmax(), 'fecha']} ({df.total_mov.max()} movs)")
print(f"  Semana menos activa: {df.loc[df.total_mov.idxmin(), 'fecha']} ({df.total_mov.min()} movs)")

# ── MILP SCALE (de metrics) ────────────────────────────────────────────────────
print("\n=== MILP SCALE ===")
print(f"  Variables:    min={metrics.vars_total.min():.0f}, max={metrics.vars_total.max():.0f}, avg={metrics.vars_total.mean():.0f}")
print(f"  Restricciones: min={metrics.constr_total.min():.0f}, max={metrics.constr_total.max():.0f}, avg={metrics.constr_total.mean():.0f}")
print(f"  size_B: {metrics.size_B.min()} – {metrics.size_B.max()}")
print(f"  size_S: min={metrics.size_S.min()}, max={metrics.size_S.max()}, avg={metrics.size_S.mean():.1f}")
print(f"  size_T: {metrics.size_T.min()} – {metrics.size_T.max()}")

# ── SOLVER ─────────────────────────────────────────────────────────────────────
print("\n=== SOLVER STATS ===")
n_optimal = (metrics.termination == "optimal").sum()
n_total = len(metrics)
print(f"  Semanas resueltas: {n_total}")
print(f"  Óptimas: {n_optimal} / {n_total} ({100*n_optimal/n_total:.1f}%)")
print(f"  Build time (s): min={metrics.build_seconds.min():.2f}, max={metrics.build_seconds.max():.2f}, avg={metrics.build_seconds.mean():.2f}")
print(f"  Solve time (s): min={metrics.solve_seconds.min():.2f}, max={metrics.solve_seconds.max():.2f}, avg={metrics.solve_seconds.mean():.2f}, median={metrics.solve_seconds.median():.2f}")
print(f"  Threads (típico): {metrics.threads.mode()[0]}")

# ── TABLA COMPLETA ─────────────────────────────────────────────────────────────
print("\n=== TABLA COMPLETA ===")
print(df.to_string(index=False))
