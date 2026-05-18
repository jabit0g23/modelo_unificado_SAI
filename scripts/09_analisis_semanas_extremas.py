"""
Identifica las semanas con mayor y menor reducción de distancia del modelo
respecto a la operación real, excluyendo movimientos de reubicación (YARD).

Comparación justa:
    dist_modelo  vs  dist_real_LOAD + dist_real_DLVR

Para las dos semanas extremas calcula también:
  - Resumen de contexto (segregaciones, flujos, KL_s, ocupación)
  - Tabla de distancias desglosada por LOAD / DLVR
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT         = SCRIPT_DIR.parent
RESULTADOS   = ROOT / "resultados" / "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20_90_base"
ANALISIS_SRC = SCRIPT_DIR / "analisis_resultados_dist.xlsx"
OUT_XLSX     = SCRIPT_DIR / "09_semanas_extremas.xlsx"
OUT_FIG_DIR  = SCRIPT_DIR / "09_figuras"


# ── helpers ────────────────────────────────────────────────────────────────

def contexto_semana(semana: str) -> dict:
    inst_dir = RESULTADOS / "instancias_coloracion" / semana
    res_dir  = RESULTADOS / "resultados_coloracion" / semana

    inst = pd.ExcelFile(inst_dir / f"Instancia_{semana}_K.xlsx")
    dp   = inst.parse("D_params")
    ki   = inst.parse("KI_s")
    vp   = inst.parse("VP_b")

    # segregaciones activas (con algún flujo o inventario)
    i0       = inst.parse("I0_sb")
    flujo_s  = dp.groupby("Segregacion")[["DR","DC","DD","DE"]].sum()
    flujo_s["total"] = flujo_s.sum(axis=1)
    i0_s     = i0.groupby("S")["I0"].sum()
    ki_full  = ki.merge(flujo_s.reset_index(), on="Segregacion", how="left").fillna(0)
    ki_full  = ki_full.merge(i0_s.reset_index().rename(columns={"S":"S"}), on="S", how="left").fillna(0)
    activas  = ki_full[(ki_full["total"] > 0) | (ki_full["I0"] > 0)]

    # flujos totales
    recv  = int(dp["DR"].sum())
    load  = int(dp["DC"].sum())
    dsch  = int(dp["DD"].sum())
    dlvr  = int(dp["DE"].sum())

    # distribución KL_s (solo segs activas, sin rumas)
    activas_no_ruma = activas[~activas["Segregacion"].str.contains("RUMA|GRUPO", case=False, na=False)]
    ki_dist = activas_no_ruma["KI"].value_counts().sort_index().to_dict()
    n_segs  = len(activas_no_ruma)
    pct_ki1 = round(ki_dist.get(1, 0) / n_segs * 100, 1) if n_segs else 0

    # ocupación media del patio (bahías ocupadas / bahías totales por bloque, promedio sobre periodos)
    res = pd.read_excel(res_dir / f"resultado_{semana}_K.xlsx")
    occ_per_bloque_periodo = (
        res.groupby(["Periodo", "Bloque"])
           .agg(bahias_usadas=("Bahías Ocupadas", "sum"), bahias_total=("Bahías", "first"))
           .reset_index()
    )
    occ_per_bloque_periodo["occ"] = (
        occ_per_bloque_periodo["bahias_usadas"] / occ_per_bloque_periodo["bahias_total"]
    )
    occ_media = round(occ_per_bloque_periodo["occ"].mean() * 100, 1)

    return {
        "semana":    semana,
        "n_segs":    n_segs,
        "recv":      recv,
        "load_mov":  load,
        "dsch":      dsch,
        "dlvr_mov":  dlvr,
        "ki_dist":   ki_dist,
        "pct_ki1":   pct_ki1,
        "occ_media": occ_media,
    }


def distancias_semana(semana: str, real_row, modelo_row) -> dict:
    return {
        "semana":         semana,
        "real_load":      int(real_row["Distancia_LOAD"]),
        "real_dlvr":      int(real_row["Distancia_DLVR"]),
        "real_yard":      int(real_row["Distancia_YARD"]),
        "real_sin_yard":  int(real_row["Distancia_LOAD"] + real_row["Distancia_DLVR"]),
        "mod_load":       int(modelo_row["Distancia_LOAD"]),
        "mod_dlvr":       int(modelo_row["Distancia_DLVR"]),
        "mod_total":      int(modelo_row["Distancia_Total"]),
        "red_load":       round((real_row["Distancia_LOAD"]  - modelo_row["Distancia_LOAD"])  / real_row["Distancia_LOAD"]  * 100, 1),
        "red_dlvr":       round((real_row["Distancia_DLVR"]  - modelo_row["Distancia_DLVR"])  / real_row["Distancia_DLVR"]  * 100, 1),
        "red_total":      round((real_row["Distancia_LOAD"] + real_row["Distancia_DLVR"] - modelo_row["Distancia_Total"])
                                / (real_row["Distancia_LOAD"] + real_row["Distancia_DLVR"]) * 100, 1),
        "pct_yard":       round(real_row["Distancia_YARD"] / real_row["Distancia_Total"] * 100, 1),
    }


# ── ranking ────────────────────────────────────────────────────────────────

real   = pd.read_excel(ANALISIS_SRC, sheet_name="Distancias reales")
modelo = pd.read_excel(ANALISIS_SRC, sheet_name="Distancias modelo").rename(columns={
    "Distancia Total": "Distancia_Total",
    "Distancia LOAD":  "Distancia_LOAD",
    "Distancia DLVR":  "Distancia_DLVR",
})

df = real[["Semana","Distancia_LOAD","Distancia_DLVR","Distancia_YARD",
           "Movimientos_LOAD","Movimientos_DLVR","Movimientos_YARD"]].merge(
    modelo[["Semana","Distancia_Total","Distancia_LOAD","Distancia_DLVR",
            "Movimientos_LOAD","Movimientos_DLVR"]],
    on="Semana", suffixes=("_real","_modelo")
)

df["dist_real_sin_yard"] = df["Distancia_LOAD_real"] + df["Distancia_DLVR_real"]
df["reduccion_pct"] = (
    (df["dist_real_sin_yard"] - df["Distancia_Total"]) / df["dist_real_sin_yard"] * 100
).round(2)
df["pct_yard"] = (
    df["Movimientos_YARD"] /
    (df["Movimientos_LOAD_real"] + df["Movimientos_DLVR_real"] + df["Movimientos_YARD"]) * 100
).round(1)

ranking = df[["Semana","dist_real_sin_yard","Distancia_Total","Distancia_YARD",
              "pct_yard","reduccion_pct"]].sort_values("reduccion_pct", ascending=False).reset_index(drop=True)
ranking.columns = ["Semana","dist_real_sin_yard","dist_modelo_total","dist_yard_real","pct_yard","reduccion_pct"]

sem_mejor = ranking.iloc[0]["Semana"]
sem_peor  = ranking.iloc[-1]["Semana"]

# ── detalle de las dos semanas extremas ───────────────────────────────────

resultados = {}
for semana in [sem_mejor, sem_peor]:
    real_row   = real[real["Semana"] == semana].iloc[0]
    modelo_row = modelo[modelo["Semana"] == semana].iloc[0]
    ctx  = contexto_semana(semana)
    dist = distancias_semana(semana, real_row, modelo_row)
    resultados[semana] = {**ctx, **dist}

# ── impresión ─────────────────────────────────────────────────────────────

for label, semana in [("MEJOR CASO", sem_mejor), ("PEOR CASO", sem_peor)]:
    r = resultados[semana]
    print(f"\n{'='*60}")
    print(f"  {label}: {semana}")
    print(f"{'='*60}")
    print(f"  Segregaciones activas (sin rumas): {r['n_segs']}")
    print(f"  Flujos: RECV={r['recv']:,}  LOAD={r['load_mov']:,}  DSCH={r['dsch']:,}  DLVR={r['dlvr_mov']:,}")
    print(f"  Ocupación media patio: {r['occ_media']}%")
    print(f"  KL_s dist: {r['ki_dist']}  (KL=1: {r['pct_ki1']}%)")
    print(f"  {'─'*50}")
    print(f"  Distancias    Real (m)      Modelo (m)   Reducción")
    print(f"  LOAD         {r['real_load']:>12,}  {r['mod_load']:>12,}   {r['red_load']:+.1f}%")
    print(f"  DLVR         {r['real_dlvr']:>12,}  {r['mod_dlvr']:>12,}   {r['red_dlvr']:+.1f}%")
    print(f"  Total        {r['real_sin_yard']:>12,}  {r['mod_total']:>12,}   {r['red_total']:+.1f}%")
    print(f"  YARD (real)  {r['real_yard']:>12,}  {'—':>12}   ({r['pct_yard']:.1f}% del total real)")

# ── guardar ────────────────────────────────────────────────────────────────

rows_ctx  = [{"semana": s, **{k: v for k,v in d.items() if k != "ki_dist"}}
             for s, d in resultados.items()]
rows_dist = rows_ctx  # mismo dict

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    ranking.to_excel(w, sheet_name="Ranking", index=False)
    pd.DataFrame(rows_ctx)[["semana","n_segs","recv","load_mov","dsch","dlvr_mov",
                             "occ_media","pct_ki1"]].to_excel(w, sheet_name="Contexto", index=False)
    pd.DataFrame(rows_dist)[["semana","real_load","real_dlvr","real_yard","real_sin_yard",
                              "mod_load","mod_dlvr","mod_total",
                              "red_load","red_dlvr","red_total","pct_yard"]].to_excel(w, sheet_name="Distancias", index=False)

print(f"\nGuardado en: {OUT_XLSX}")

# ── visualización: flujos por bloque → código LaTeX/pgfplots ──────────────

OUT_FIG_DIR.mkdir(exist_ok=True)


def flujos_reales_por_bloque(semana: str) -> pd.DataFrame:
    flujos_path = RESULTADOS / "instancias_coloracion" / semana / f"analisis_flujos_w{semana}_0.xlsx"
    df = pd.read_excel(flujos_path, sheet_name="FlujosAll_sbt")
    load = (df[df["ime_to"].str.startswith("Y-SAI", na=False)]
              .groupby("ime_fm")["LOAD"].sum().rename("real_load"))
    dlvr = (df[df["ime_to"] == "GATE"]
              .groupby("ime_fm")["DLVR"].sum().rename("real_dlvr"))
    return (pd.DataFrame({"real_load": load, "real_dlvr": dlvr})
              .fillna(0).reset_index().rename(columns={"ime_fm": "Bloque"}))


def flujos_modelo_por_bloque(semana: str) -> pd.DataFrame:
    res_path = RESULTADOS / "resultados_coloracion" / semana / f"resultado_{semana}_K.xlsx"
    df = pd.read_excel(res_path, sheet_name="General")
    agg = df.groupby("Bloque")[["Carga", "Entrega"]].sum().reset_index()
    agg.columns = ["Bloque", "mod_load", "mod_dlvr"]
    return agg


def distancias_por_bloque(semana: str) -> pd.DataFrame:
    inst_path = RESULTADOS / "instancias_coloracion" / semana / f"Instancia_{semana}_K.xlsx"
    xl = pd.ExcelFile(inst_path)
    le = xl.parse("LE_b").rename(columns={"B": "Bloque"})
    lc_df = xl.parse("LC_sb")
    lc = (lc_df[lc_df["LC"] > 0].groupby("B")["LC"].mean()
            .reset_index().rename(columns={"B": "Bloque", "LC": "LC_mean"}))
    return le.merge(lc, on="Bloque")


def _pgfplots_subfig(sub: pd.DataFrame, flow: str, dist_col: str,
                     xlabel: str, title: str) -> str:
    """Genera el bloque pgfplots para un subplot (LOAD o DLVR)."""
    sub = sub.sort_values(dist_col).reset_index(drop=True)

    total_real = sub[f"real_{flow}"].sum()
    total_mod  = sub[f"mod_{flow}"].sum()
    if total_real == 0 or total_mod == 0:
        return f"% sin datos para {flow}\n"

    sub["pct_real"] = (sub[f"real_{flow}"] / total_real * 100).round(2)
    sub["pct_mod"]  = (sub[f"mod_{flow}"]  / total_mod  * 100).round(2)

    # etiquetas anonimizadas: B1, B2, ... con distancia
    n = len(sub)
    labels = [f"B{i+1}\\\\{int(sub[dist_col].iloc[i])}" for i in range(n)]
    coords_real = " ".join(f"({i},{sub['pct_real'].iloc[i]:.2f})" for i in range(n))
    coords_mod  = " ".join(f"({i},{sub['pct_mod'].iloc[i]:.2f})"  for i in range(n))
    xticklabels = ",".join(f"{{{l}}}" for l in labels)

    lines = [
        r"\begin{axis}[",
        r"  ybar, bar width=7pt, width=\textwidth, height=5.5cm,",
        f"  title={{{title}}},",
        f"  xlabel={{{xlabel}}},",
        r"  ylabel={Porcentaje (\%)},",
        r"  xtick={" + ",".join(str(i) for i in range(n)) + "},",
        r"  xticklabels={" + xticklabels + "},",
        r"  xticklabel style={font=\tiny, align=center},",
        r"  ymin=0,",
        r"  legend style={at={(0.98,0.98)}, anchor=north east, font=\small},",
        r"  ymajorgrids=true, grid style={dashed,gray!40},",
        r"  enlarge x limits=0.025,",
        r"]",
        r"\addplot[fill=blue!50, draw=blue!70] coordinates {" + coords_real + "};",
        r"\addplot[fill=orange!60, draw=orange!80] coordinates {" + coords_mod + "};",
        r"\legend{Real, Modelo}",
        r"\end{axis}",
    ]
    return "\n".join(lines)


def generar_latex_semana(semana: str, label: str, fig_label: str, caption: str) -> str:
    real  = flujos_reales_por_bloque(semana)
    mod   = flujos_modelo_por_bloque(semana)
    dists = distancias_por_bloque(semana)
    df = dists.merge(real, on="Bloque", how="left").merge(mod, on="Bloque", how="left").fillna(0)

    sub_load = _pgfplots_subfig(df, "load", "LC_mean",
                                 r"Bloque (B$i$ = $i$-ésimo más cercano al muelle)",
                                 "LOAD")
    sub_dlvr = _pgfplots_subfig(df, "dlvr", "LE",
                                 r"Bloque (B$i$ = $i$-ésimo más cercano al gate)",
                                 "DLVR")

    tex = "\n".join([
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        sub_load,
        r"\end{tikzpicture}",
        r"\vspace{4pt}",
        r"\begin{tikzpicture}",
        sub_dlvr,
        r"\end{tikzpicture}",
        r"\caption{" + caption + "}",
        r"\label{" + fig_label + "}",
        r"\end{figure}",
    ])
    return tex


for label, semana, fig_label, caption in [
    ("Mejor caso", sem_mejor,
     "fig:flujos-sem3",
     r"Distribución porcentual de flujos por bloque, semana~3 (2022-01-17). "
     r"Bloques ordenados por distancia creciente al destino. "
     r"Barras azules: operación real; barras naranjas: modelo."),
    ("Peor caso",  sem_peor,
     "fig:flujos-sem12",
     r"Distribución porcentual de flujos por bloque, semana~12 (2022-03-21). "
     r"Bloques ordenados por distancia creciente al destino."),
]:
    tex = generar_latex_semana(semana, label, fig_label, caption)
    out_path = OUT_FIG_DIR / f"flujos_{semana}.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"LaTeX guardado: {out_path}")
