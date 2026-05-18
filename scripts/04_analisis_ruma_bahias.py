"""
04_analisis_ruma_bahias.py

Analiza los grupos RUMA en los resultados del modelo de coloración:
- Bahías ocupadas promedio y pico por tipo RUMA
- Comparación percentil vs segregaciones reales
- Composición: cuántas segs originales se agruparon y cuántas hay en total de ese prefijo

Salida: scripts/analisis_ruma_bahias.xlsx con 2 hojas:
  - Resumen   : una fila por tipo RUMA, agregado sobre todas las semanas
  - Detalle   : una fila por (semana × tipo RUMA)

Uso:
    python scripts/04_analisis_ruma_bahias.py
    python scripts/04_analisis_ruma_bahias.py --resultado_dir resultados/otro_dir
"""

import os
import argparse
import pandas as pd
import numpy as np

DEFAULT_RESULTADO_DIR = "resultados/pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20_90"
OUTPUT_FILE = "scripts/analisis_ruma_bahias.xlsx"


def _prefix3(name):
    parts = str(name).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else name


def cargar_semana(semana, res_dir, inst_dir):
    """
    Lee resultado + instancia de una semana.
    Devuelve (df_general, df_s, df_flujos) o None si falta algún archivo.
    """
    res_files = [
        f for f in os.listdir(res_dir)
        if f.startswith("resultado_") and f.endswith(".xlsx")
    ] if os.path.isdir(res_dir) else []
    inst_path = os.path.join(inst_dir, f"Instancia_{semana}.xlsx")
    analisis_path = os.path.join(inst_dir, f"analisis_flujos_w{semana}_0.xlsx")

    if not res_files or not os.path.exists(inst_path) or not os.path.exists(analisis_path):
        return None

    df_gen = pd.read_excel(os.path.join(res_dir, res_files[0]), sheet_name="General")
    df_s   = pd.read_excel(inst_path, sheet_name="S")
    df_f   = pd.read_excel(analisis_path, sheet_name="FlujosAll_sb_P")
    return df_gen, df_s, df_f


def procesar_semana(semana, df_gen, df_s, df_f):
    """
    Calcula métricas para cada grupo RUMA en una semana.
    Devuelve lista de dicts (uno por tipo RUMA presente).
    """
    seg_map = df_s.set_index("S")["Segregacion"].to_dict()
    df_gen["seg_nombre"] = df_gen["Segregación"].map(seg_map)

    # Segs individuales (no-RUMA) en el modelo
    segs_individuales = set(df_s["Segregacion"][~df_s["Segregacion"].str.endswith("-RUMA", na=False)])

    # Candidatas originales (todas las que aparecen en los flujos)
    todas_en_flujos = set(df_f["criterio"].dropna().unique())

    # Bahías ocupadas por periodo, sumadas sobre bloques (aditivo)
    periodo_col = "Período" if "Período" in df_gen.columns else "Periodo"
    agg = (
        df_gen.groupby(["seg_nombre", periodo_col])[["Bahías Ocupadas", "Volumen (TEUs)"]]
        .sum()
        .reset_index()
        .rename(columns={periodo_col: "Periodo"})
    )

    # Stats de segs reales para el percentil
    reales = agg[~agg["seg_nombre"].str.endswith("-RUMA", na=False)]
    bah_reales_por_seg = reales.groupby("seg_nombre")["Bahías Ocupadas"].mean()
    p25_r, p50_r, p75_r = bah_reales_por_seg.quantile([0.25, 0.5, 0.75]).values

    ruma_segs = df_s[df_s["Segregacion"].str.endswith("-RUMA", na=False)]["Segregacion"].tolist()

    rows = []
    for seg_ruma in ruma_segs:
        prefijo = _prefix3(seg_ruma)  # e.g. "expo-dry-40"

        # Segs de este prefijo que quedaron individuales en el modelo
        indiv_mismo_prefijo = {s for s in segs_individuales if _prefix3(s) == prefijo}

        # Segs de este prefijo que existían en los flujos (candidatas)
        todas_mismo_prefijo = {s for s in todas_en_flujos if _prefix3(s) == prefijo}

        # Las que fueron a RUMA = todas del prefijo que NO quedaron individuales
        n_segs_en_ruma  = len(todas_mismo_prefijo - indiv_mismo_prefijo)
        n_segs_indiv    = len(indiv_mismo_prefijo)
        n_segs_total    = len(todas_mismo_prefijo)

        # Bahías ocupadas de este RUMA (suma sobre bloques por periodo)
        datos_ruma = agg[agg["seg_nombre"] == seg_ruma]["Bahías Ocupadas"]
        bah_avg = datos_ruma.mean()
        bah_max = datos_ruma.max()

        pctil = (bah_reales_por_seg < bah_avg).mean() * 100 if len(bah_reales_por_seg) > 0 else None

        rows.append({
            "semana":           semana,
            "seg_ruma":         seg_ruma,
            "bah_ocu_avg":      round(bah_avg, 2),
            "bah_ocu_max":      int(bah_max),
            "pctil_vs_reales":  round(pctil, 1) if pctil is not None else None,
            "p25_reales":       round(p25_r, 2),
            "p50_reales":       round(p50_r, 2),
            "p75_reales":       round(p75_r, 2),
            "n_segs_en_ruma":   n_segs_en_ruma,
            "n_segs_indiv":     n_segs_indiv,
            "n_segs_total_tipo": n_segs_total,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultado_dir", default=DEFAULT_RESULTADO_DIR)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resultado_dir = args.resultado_dir if os.path.isabs(args.resultado_dir) \
        else os.path.join(base, args.resultado_dir)

    res_col  = os.path.join(resultado_dir, "resultados_coloracion")
    inst_col = os.path.join(resultado_dir, "instancias_coloracion")

    semanas = sorted(os.listdir(res_col)) if os.path.isdir(res_col) else []
    print(f"Leyendo resultados desde: {resultado_dir}")

    todos = []
    ok = 0
    for semana in semanas:
        datos = cargar_semana(
            semana,
            os.path.join(res_col, semana),
            os.path.join(inst_col, semana),
        )
        if datos is None:
            continue
        filas = procesar_semana(semana, *datos)
        todos.extend(filas)
        ok += 1

    print(f"Semanas procesadas: {ok} / {len(semanas)}")

    if not todos:
        print("Sin datos.")
        return

    # ── Detalle: una fila por semana × tipo RUMA ──────────────────────────
    df_det = pd.DataFrame(todos).sort_values(["semana", "seg_ruma"]).reset_index(drop=True)

    col_order_det = [
        "semana", "seg_ruma",
        "bah_ocu_avg", "bah_ocu_max",
        "pctil_vs_reales", "p25_reales", "p50_reales", "p75_reales",
        "n_segs_en_ruma", "n_segs_indiv", "n_segs_total_tipo",
    ]
    df_det = df_det[col_order_det]

    # ── Resumen: agregar sobre semanas ────────────────────────────────────
    agg_fns = {
        "bah_ocu_avg":       ["mean", "median"],
        "bah_ocu_max":       ["mean", lambda x: x.quantile(0.95)],
        "pctil_vs_reales":   ["mean"],
        "p50_reales":        ["mean"],
        "n_segs_en_ruma":    ["mean", "median", "max"],
        "n_segs_indiv":      ["mean"],
        "n_segs_total_tipo": ["mean", "median", "max"],
    }
    df_res = df_det.groupby("seg_ruma").agg(agg_fns)
    df_res.columns = [
        "bah_ocu_avg_mean", "bah_ocu_avg_p50",
        "bah_ocu_max_mean", "bah_ocu_max_p95",
        "pctil_vs_reales_avg",
        "p50_reales_avg",
        "n_segs_en_ruma_avg", "n_segs_en_ruma_p50", "n_segs_en_ruma_max",
        "n_segs_indiv_avg",
        "n_segs_total_tipo_avg", "n_segs_total_tipo_p50", "n_segs_total_tipo_max",
    ]
    df_res["n_semanas"] = df_det.groupby("seg_ruma")["semana"].nunique()
    df_res = df_res.reset_index().round(2)

    col_order_res = [
        "seg_ruma", "n_semanas",
        "bah_ocu_avg_mean", "bah_ocu_avg_p50",
        "bah_ocu_max_mean", "bah_ocu_max_p95",
        "pctil_vs_reales_avg", "p50_reales_avg",
        "n_segs_en_ruma_avg", "n_segs_en_ruma_p50", "n_segs_en_ruma_max",
        "n_segs_indiv_avg",
        "n_segs_total_tipo_avg", "n_segs_total_tipo_p50", "n_segs_total_tipo_max",
    ]
    df_res = df_res[col_order_res].sort_values("seg_ruma").reset_index(drop=True)

    # ── Imprimir resumen en consola ───────────────────────────────────────
    print("\n" + "="*100)
    print("RESUMEN POR TIPO RUMA")
    print("="*100)
    print(df_res.to_string(index=False))

    # ── Exportar ──────────────────────────────────────────────────────────
    out = os.path.join(base, OUTPUT_FILE)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_res.to_excel(w, sheet_name="Resumen", index=False)
        df_det.to_excel(w, sheet_name="Detalle", index=False)
    print(f"\nExportado: {out}")


if __name__ == "__main__":
    main()
