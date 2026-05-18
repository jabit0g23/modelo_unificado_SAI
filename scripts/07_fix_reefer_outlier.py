"""
Fix reefer overload para la semana 2022-03-28 (outlier).

El problema es triple:
  1) VSR por bloque: cada bloque tiene un limite de bahias reefer (VSR[b]).
     La restriccion del modelo es sum(v[s,b,t]) <= VSR[b] por bloque y periodo,
     donde v[s,b,t] = ceil(i[s,b,t] / C[b]).  Con 15 segs reefer activas
     simultaneamente y bloques pequenos (VSR=4), escalar por contenedores no
     garantiza que la suma de ceilings quepa.
  2) BigM negativo: DC supera I0+DR en algunos segs (constraint_9_inv).
  3) Peak global de bahias: sum_s ceil(I_s[t]/C) > total_VSR en algun periodo t.

Este script aplica tres pasos en orden:
  1) I0 por bloque  : escala I0 para que sum(ceil) <= floor(PERCENTILE*VSR[b])
  2) DR agregado    : escala DR para que peak_bays <= floor(PERCENTILE*total_VSR)
  3) DC+DE por seg  : cap greedy exacto para que BigM >= 0 en todo momento

Sobreescribe ambos archivos de instancia (.xlsx y _K.xlsx).
Si existe .bak, lee desde ahi (idempotente).

Uso:
    python scripts/07_fix_reefer_outlier.py
"""

import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

PERCENTILE = 0.90
C_BAY = 35          # contenedores por bahia (uniforme en todos los bloques reefer)
SEMANA = "2022-03-28"
INST_DIR = (
    Path(__file__).parent.parent
    / "resultados"
    / "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20_Rmas"
    / "instancias_coloracion"
    / SEMANA
)


# ── utilidades ─────────────────────────────────────────────────────────────────

def _scale_col(df: pd.DataFrame, segs: set, col: str, scale: float) -> pd.DataFrame:
    df = df.copy()
    mask = df["S"].isin(segs)
    df.loc[mask, col] = np.floor(df.loc[mask, col] * scale).astype(int)
    return df


def _bays_for_block(i0_sb: pd.DataFrame, reefer_segs: set, b: str) -> int:
    """Suma de ceil(I0[s,b] / C_BAY) para los segs reefer con I0>0 en bloque b."""
    mask = i0_sb["S"].isin(reefer_segs) & (i0_sb["B"] == b) & (i0_sb["I0"] > 0)
    return int(sum(math.ceil(float(v) / C_BAY) for v in i0_sb.loc[mask, "I0"]))


def _bays_for_block_scaled(i0_sb: pd.DataFrame, reefer_segs: set, b: str, scale: float) -> int:
    mask = i0_sb["S"].isin(reefer_segs) & (i0_sb["B"] == b) & (i0_sb["I0"] > 0)
    return int(sum(
        math.ceil(math.floor(float(v) * scale) / C_BAY)
        for v in i0_sb.loc[mask, "I0"]
        if math.floor(float(v) * scale) > 0
    ))


def _compute_peak_bays(d_params: pd.DataFrame, reefer_segs: set, i0_by_seg: pd.Series) -> int:
    """
    Max sobre todos los periodos de: sum_s ceil(I_s[t] / C_BAY).
    Simula el inventario total de cada seg a lo largo del horizonte.
    """
    T_list = sorted(d_params["T"].unique())
    inv = {s: float(i0_by_seg.get(s, 0)) for s in reefer_segs}

    # inventario inicial
    peak = sum(math.ceil(v / C_BAY) for v in inv.values() if v > 0)

    # agrupar D_params por (S, T) para velocidad
    d_r = d_params[d_params["S"].isin(reefer_segs)].set_index(["S", "T"])

    for t in T_list:
        for s in reefer_segs:
            try:
                row = d_r.loc[(s, t)]
                inv[s] = max(0.0, inv[s]
                             + float(row["DR"]) + float(row["DD"])
                             - float(row["DC"]) - float(row["DE"]))
            except KeyError:
                pass
        bays = sum(math.ceil(v / C_BAY) for v in inv.values() if v > 0)
        peak = max(peak, bays)

    return peak


def _cap_dc_greedy(d_params: pd.DataFrame, i0_sb: pd.DataFrame, reefer_segs: set) -> tuple[pd.DataFrame, list]:
    """
    Por cada seg reefer, limita DC y DE periodo a periodo para que BigM >= 0.
    Retorna el df modificado y lista de cambios (seg, t, dc_orig, dc_new).
    """
    df = d_params.copy()
    cambios = []
    for seg in reefer_segs:
        i0_s = int(round(float(i0_sb.loc[i0_sb["S"] == seg, "I0"].sum())))
        seg_idx = df[df["S"] == seg].sort_values("T").index.tolist()
        cum_in = 0
        cum_out = 0
        for idx in seg_idx:
            dr = int(round(float(df.at[idx, "DR"])))
            dd = int(round(float(df.at[idx, "DD"])))
            dc = int(round(float(df.at[idx, "DC"])))
            de = int(round(float(df.at[idx, "DE"])))
            cum_in += dr + dd
            disponible = i0_s + cum_in - cum_out
            if dc + de > disponible:
                dc_new = min(dc, max(0, disponible))
                de_new = min(de, max(0, disponible - dc_new))
                if dc != dc_new:
                    cambios.append((seg, int(df.at[idx, "T"]), dc, dc_new))
                df.at[idx, "DC"] = dc_new
                df.at[idx, "DE"] = de_new
                cum_out += dc_new + de_new
            else:
                cum_out += dc + de
    return df, cambios


# ── logica principal ────────────────────────────────────────────────────────────

def compute_scaled_sheets(
    sheets: dict,
    reefer_segs: set,
    vsr_per_block: pd.Series,
) -> tuple[dict, dict]:
    """
    Aplica los tres pasos de escalado.  Retorna (sheets_escalados, info).
    """
    total_vsr = int(vsr_per_block.sum())
    target_bays_global = int(PERCENTILE * total_vsr)   # floor

    i0_sb = sheets["I0_sb"].copy()

    # ── Paso 1: escalar I0 por bloque en bahias ────────────────────────────────
    for b in vsr_per_block.index:
        vsr_b = int(vsr_per_block[b])
        if vsr_b == 0:
            continue
        target_b = int(PERCENTILE * vsr_b)   # bahias objetivo en bloque b

        current_bays = _bays_for_block(i0_sb, reefer_segs, b)
        if current_bays <= target_b:
            continue

        # busqueda binaria de scale_b
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if _bays_for_block_scaled(i0_sb, reefer_segs, b, mid) <= target_b:
                lo = mid
            else:
                hi = mid
        scale_b = lo
        mask = i0_sb["S"].isin(reefer_segs) & (i0_sb["B"] == b)
        i0_sb.loc[mask, "I0"] = np.floor(i0_sb.loc[mask, "I0"] * scale_b).astype(int)

    # ── Paso 2: escalar DR si el peak de bahias supera el target global ────────
    i0_by_seg = i0_sb[i0_sb["S"].isin(reefer_segs)].groupby("S")["I0"].sum()
    d_params = sheets["D_params"].copy()
    scale_dr = 1.0

    if _compute_peak_bays(d_params, reefer_segs, i0_by_seg) > target_bays_global:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            d_mid = _scale_col(d_params, reefer_segs, "DR", mid)
            if _compute_peak_bays(d_mid, reefer_segs, i0_by_seg) <= target_bays_global:
                lo = mid
            else:
                hi = mid
        if lo < 0.9999:
            scale_dr = lo
            d_params = _scale_col(d_params, reefer_segs, "DR", scale_dr)

    # ── Paso 3: cap DC+DE por seg con BigM >= 0 ────────────────────────────────
    d_params, cambios_dc = _cap_dc_greedy(d_params, i0_sb, reefer_segs)

    out = dict(sheets)
    out["I0_sb"] = i0_sb
    out["D_params"] = d_params
    if "D_params_168h" in out:
        d168 = sheets["D_params_168h"].copy()
        d168 = _scale_col(d168, reefer_segs, "DR", scale_dr)
        d168, _ = _cap_dc_greedy(d168, i0_sb, reefer_segs)
        out["D_params_168h"] = d168

    return out, {"cambios_dc": cambios_dc, "scale_dr": scale_dr}


def write_excel(path: Path, sheets: dict):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)


def main():
    files = [
        INST_DIR / f"Instancia_{SEMANA}.xlsx",
        INST_DIR / f"Instancia_{SEMANA}_K.xlsx",
    ]

    # leer desde .bak si existe (idempotente)
    src_main = files[0].with_suffix(".bak") if files[0].with_suffix(".bak").exists() else files[0]
    xl_main = pd.ExcelFile(src_main)
    r_s      = xl_main.parse("R_s")
    vsr      = xl_main.parse("VSR_b").set_index("B")["VSR"]
    cb       = xl_main.parse("C_b").set_index("B")["C"]
    reefer_segs = set(r_s[r_s["R"] == 1]["S"])
    vsr_reefer  = vsr[vsr > 0]            # solo bloques con capacidad reefer
    total_vsr   = int(vsr_reefer.sum())   # bahias totales
    total_cap   = int((vsr * cb).sum())   # contenedores totales (para referencia)

    sheets_src = {name: xl_main.parse(name) for name in xl_main.sheet_names}
    i0_by_seg_orig = sheets_src["I0_sb"][sheets_src["I0_sb"]["S"].isin(reefer_segs)].groupby("S")["I0"].sum()
    i0_orig      = int(i0_by_seg_orig.sum())
    peak_bays_orig = _compute_peak_bays(sheets_src["D_params"], reefer_segs, i0_by_seg_orig)

    print(f"Semana: {SEMANA}  |  percentil objetivo: {PERCENTILE*100:.0f}%")
    print(f"Total bahias reefer (VSR): {total_vsr}  |  target bahias: {int(PERCENTILE*total_vsr)}")
    print(f"Total cap contenedores: {total_cap}")
    print(f"I0 reefer original: {i0_orig}  |  peak bahias original: {peak_bays_orig}  ({100*peak_bays_orig/total_vsr:.1f}%)")

    _, info = compute_scaled_sheets(sheets_src, reefer_segs, vsr_reefer)
    if info["cambios_dc"]:
        segs_dc = sorted({s for s, *_ in info["cambios_dc"]})
        print(f"\nPaso 3 – DC cappado ({len(segs_dc)} segs): {segs_dc}")
    if info["scale_dr"] < 1.0:
        print(f"\nPaso 2 – DR escalado: x{info['scale_dr']:.4f}")

    # aplicar a cada archivo preservando su KI_s propio
    for path in files:
        src = path.with_suffix(".bak") if path.with_suffix(".bak").exists() else path
        xl  = pd.ExcelFile(src)
        sheets = {name: xl.parse(name) for name in xl.sheet_names}
        ki_s = sheets["KI_s"].copy()

        scaled, _ = compute_scaled_sheets(sheets, reefer_segs, vsr_reefer)
        scaled["KI_s"] = ki_s

        if not path.with_suffix(".bak").exists():
            shutil.copy2(path, path.with_suffix(".bak"))

        write_excel(path, scaled)
        print(f"  Actualizado: {path.name}")

    # verificar resultado final
    xl_check    = pd.ExcelFile(files[0])
    i0_check    = xl_check.parse("I0_sb")
    d_check     = xl_check.parse("D_params")
    i0_by_seg_new = i0_check[i0_check["S"].isin(reefer_segs)].groupby("S")["I0"].sum()
    i0_new      = int(i0_by_seg_new.sum())
    peak_bays_new = _compute_peak_bays(d_check, reefer_segs, i0_by_seg_new)

    # verificar BigM por seg
    min_bigm = float("inf")
    for seg in reefer_segs:
        i0_s = int(i0_by_seg_new.get(seg, 0))
        s_data = d_check[d_check["S"] == seg].sort_values("T")
        cum_in, cum_out = 0, 0
        for _, row in s_data.iterrows():
            cum_in  += int(round(float(row["DR"]))) + int(round(float(row["DD"])))
            cum_out += int(round(float(row["DC"]))) + int(round(float(row["DE"])))
            min_bigm = min(min_bigm, i0_s + cum_in - cum_out)

    # verificar bahias por bloque
    print(f"\nI0 nuevo: {i0_new}  |  peak bahias nuevo: {peak_bays_new}  ({100*peak_bays_new/total_vsr:.1f}%)")
    print(f"BigM minimo entre todos los segs reefer: {min_bigm:.0f}  ({'OK' if min_bigm >= 0 else 'NEGATIVO - revisar'})")
    print("\nBahias por bloque (debe ser <= VSR[b]):")
    for b in sorted(vsr_reefer.index):
        bays = _bays_for_block(i0_check, reefer_segs, b)
        print(f"  {b}: {bays}/{int(vsr_reefer[b])}  {'OK' if bays <= int(vsr_reefer[b]) else 'OVERFLOW'}")


if __name__ == "__main__":
    main()
