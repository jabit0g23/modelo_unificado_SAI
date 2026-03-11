import os
import re
import glob
import math
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
ROOT_SEMANAS = r"resultados_generados_bahia_criterio_ii_p68/instancias_magdalena"
OUTPUT = r"resultados_generados_bahia_criterio_ii_p68/metrics"

FLOW_GLOB = "analisis_flujos*.xlsx"
FLOW_SHEET = "FlujosAll_sbt"

INSTANCIA_GLOB = "Instancia_*_K.xlsx"
INSTANCIA_SHEET_S = "S"

OUT_XLSX = os.path.join(OUTPUT, "Resumen_Estadistico_Instancias.xlsx")

# Si quieres intentar calcular ocupación desde inventario inicial, define una capacidad total.
# Si no la defines, la ocupación quedará vacía salvo que exista una hoja/columna explícita.
TOTAL_PATIO_CAPACITY_TEU = None
# Ejemplo:
# TOTAL_PATIO_CAPACITY_TEU = 9100

# Para exportar con formato más limpio
ROUND_DECIMALS = 2


# =========================================================
# HELPERS
# =========================================================
def _slug(x) -> str:
    if x is None:
        return ""
    txt = str(x).strip()
    txt = "".join(
        c for c in unicodedata.normalize("NFKD", txt)
        if not unicodedata.combining(c)
    )
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9]+", "", txt)
    return txt


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _upper_clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _to_num(s: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(s, errors="coerce")
    if s1.notna().any():
        return s1
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s2, errors="coerce")


def _find_best_file(folder: str, pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(folder, pattern)))
    if not matches:
        return None
    non_debug = [m for m in matches if "debug" not in os.path.basename(m).lower()]
    candidates = non_debug if non_debug else matches
    prefer_0 = [m for m in candidates if "_0" in os.path.basename(m)]
    if prefer_0:
        candidates = prefer_0
    candidates = sorted(candidates, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return candidates[0]


def _read_excel_sheet(path: str, preferred_sheet: Optional[str] = None) -> pd.DataFrame:
    if preferred_sheet is None:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        first = list(xls.keys())[0]
        return _norm_cols(xls[first])

    try:
        return _norm_cols(pd.read_excel(path, sheet_name=preferred_sheet, engine="openpyxl"))
    except Exception:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        first = list(xls.keys())[0]
        return _norm_cols(xls[first])


def _get_sheet_names(path: str) -> List[str]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    return list(xls.sheet_names)


def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    slug_to_col = {_slug(c): c for c in df.columns}
    for cand in candidates:
        if _slug(cand) in slug_to_col:
            return slug_to_col[_slug(cand)]
    if required:
        raise ValueError(f"No encontré ninguna de estas columnas: {candidates}. Disponibles: {list(df.columns)}")
    return None


def _extract_visit_code(text: str) -> Optional[str]:
    """
    Busca algo tipo EU236, MK566, HAP353, etc.
    """
    if text is None:
        return None
    txt = str(text).upper()
    m = re.search(r"\b([A-Z]{2,4}\d{2,4})\b", txt)
    return m.group(1) if m else None


def _split_visit(code: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not code:
        return None, None
    m = re.match(r"([A-Z]{2,4})(\d{2,4})$", code.upper())
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _week_from_folder_name(week_str: str) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    dt = pd.to_datetime(week_str, errors="coerce")
    if pd.isna(dt):
        return None, None
    return dt, int(dt.isocalendar().week)


def _safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan


def _safe_std(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.std(ddof=0)) if not s.empty else np.nan


def _fmt_num(x, decimals=1):
    if pd.isna(x):
        return "NA"
    if decimals == 0:
        return f"{int(round(float(x), 0))}"
    return f"{float(x):.{decimals}f}"


def _fmt_pct(x, decimals=1):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{decimals}f}%"


# =========================================================
# LECTURA DE FLUJOS
# =========================================================
def read_flujos_aggregated(path: str) -> pd.DataFrame:
    df = _read_excel_sheet(path, FLOW_SHEET)
    df = _norm_cols(df)

    c_seg = _find_col(df, ["criterio", "segregacion", "segregación"])
    c_recv = _find_col(df, ["RECV"])
    c_load = _find_col(df, ["LOAD"])
    c_dsch = _find_col(df, ["DSCH"])
    c_dlvr = _find_col(df, ["DLVR"])
    c_yard = _find_col(df, ["YARD"], required=False)

    df[c_seg] = _upper_clean(df[c_seg])
    for c in [c_recv, c_load, c_dsch, c_dlvr]:
        df[c] = _to_num(df[c]).fillna(0)

    if c_yard is None:
        df["YARD"] = 0
        c_yard = "YARD"
    else:
        df[c_yard] = _to_num(df[c_yard]).fillna(0)

    agg = (
        df.groupby(c_seg, as_index=False)[[c_recv, c_load, c_dsch, c_dlvr, c_yard]]
        .sum()
        .rename(columns={
            c_seg: "Segregacion",
            c_recv: "RECV",
            c_load: "LOAD",
            c_dsch: "DSCH",
            c_dlvr: "DLVR",
            c_yard: "YARD",
        })
    )

    for c in ["RECV", "LOAD", "DSCH", "DLVR", "YARD"]:
        agg[c] = agg[c].round(0).astype(int)

    return agg


# =========================================================
# DETECCIÓN DE REEFER
# =========================================================
def detect_reefer(base: pd.DataFrame, seg_col: str) -> Tuple[pd.Series, str]:
    # 1) Buscar columna explícita
    explicit_candidates = [
        "reefer", "es_reefer", "es reefer", "tipo_carga", "tipo carga",
        "tipo", "clase_carga", "clase carga"
    ]
    explicit = _find_col(base, explicit_candidates, required=False)

    if explicit is not None:
        s = base[explicit]

        # caso booleano / binario
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any() and set(s_num.dropna().unique()).issubset({0, 1}):
            return s_num.fillna(0).astype(int).astype(bool), f"columna:{explicit}"

        # caso texto
        s_txt = s.astype(str).str.upper().str.strip()
        is_rf = s_txt.str.contains(r"REEFER|REFRIG", regex=True, na=False)
        if is_rf.any():
            return is_rf, f"columna:{explicit}"

    # 2) Inferir desde nombre de segregación
    seg_txt = base[seg_col].astype(str).str.upper()
    is_rf = seg_txt.str.contains(r"REEFER|REFRIG", regex=True, na=False)
    return is_rf, "heuristica:segregacion"


# =========================================================
# DETECCIÓN DE PRIMARY / SECONDARY
# =========================================================
def detect_temporal_class(
    base: pd.DataFrame,
    seg_col: str,
    flows_by_seg: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, str]:
    """
    Devuelve serie con valores:
    - PRIMARY
    - SECONDARY
    - <NA>

    Orden de prioridad:
    1) columna explícita en hoja S
    2) columna categórica con valores tipo normal/anormal
    3) heurística desde visitas + flujos
    """

    # ---------- 1) columna explícita por nombre ----------
    candidate_names = [
        "primaria_secundaria", "primary_secondary", "temporalidad", "tipo_temporal",
        "clasificacion_temporal", "categoria_temporal", "tipo_segregacion",
        "tipo segregacion", "tipo", "normal_anormal", "normal anormal"
    ]
    c_temp = _find_col(base, candidate_names, required=False)

    if c_temp is not None:
        out = _map_temporal_from_text(base[c_temp])
        if out.notna().any():
            return out, f"columna:{c_temp}"

    # ---------- 2) scan de columnas categóricas ----------
    for c in base.columns:
        if c == seg_col:
            continue
        s = base[c]
        if s.dtype == "O":
            uniq = pd.Series(s.dropna().astype(str).str.strip().unique())
            if len(uniq) <= 12 and len(uniq) > 0:
                mapped = _map_temporal_from_text(s)
                if mapped.notna().any():
                    return mapped, f"scan_categorico:{c}"

    # ---------- 3) heurística con visitas ----------
    if flows_by_seg is not None and not flows_by_seg.empty:
        merged = base[[seg_col]].copy()
        merged = merged.merge(flows_by_seg, left_on=seg_col, right_on="Segregacion", how="left")
        for c in ["RECV", "LOAD", "DSCH", "DLVR", "YARD"]:
            if c not in merged.columns:
                merged[c] = 0
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

        # visita "actual" por servicio = mayor visita con LOAD > 0
        current_by_service: Dict[str, int] = {}
        for _, r in merged.iterrows():
            seg = r[seg_col]
            visit = _extract_visit_code(seg)
            pref, num = _split_visit(visit)
            if pref is None or num is None:
                continue
            if r["LOAD"] > 0:
                current_by_service[pref] = max(current_by_service.get(pref, -10**9), num)

        cls = pd.Series(pd.NA, index=merged.index, dtype="object")

        for i, r in merged.iterrows():
            seg = str(r[seg_col]).upper()

            # si ya viene importación sin visita explícita, no inventamos demasiado
            # si tiene DSCH positivo, muy probablemente es current-week => PRIMARY
            # si no, lo dejamos NA para no contaminar el análisis
            if seg.startswith("IMPO") and r["DSCH"] > 0:
                cls.loc[i] = "PRIMARY"
                continue

            visit = _extract_visit_code(seg)
            pref, num = _split_visit(visit)

            if pref is None or num is None:
                continue

            current_num = current_by_service.get(pref)
            if current_num is None:
                continue

            # Heurística simple:
            # current o current+1 => PRIMARY
            # más lejos => SECONDARY
            if num in {current_num, current_num + 1}:
                cls.loc[i] = "PRIMARY"
            else:
                cls.loc[i] = "SECONDARY"

        if cls.notna().any():
            return cls, "heuristica:visitas_y_load"

    return pd.Series(pd.NA, index=base.index, dtype="object"), "no_detectado"


def _map_temporal_from_text(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.upper().str.strip()

    out = pd.Series(pd.NA, index=s.index, dtype="object")

    mask_primary = txt.str.contains(r"PRIMARY|PRIMARIA|NORMAL|ACTUAL", regex=True, na=False)
    mask_secondary = txt.str.contains(r"SECONDARY|SECUNDARIA|ANORMAL", regex=True, na=False)

    out.loc[mask_primary] = "PRIMARY"
    out.loc[mask_secondary] = "SECONDARY"
    return out


# =========================================================
# OCUPACIÓN (OPCIONAL)
# =========================================================
def try_read_occupancy_from_instance(path: str) -> Tuple[Optional[float], str]:
    """
    Intenta sacar una ocupación semanal desde el archivo de instancia.
    Orden:
    1) hoja/campo explícito con "ocupacion"
    2) suma de inventario inicial / TOTAL_PATIO_CAPACITY_TEU
    """
    try:
        sheet_names = _get_sheet_names(path)
    except Exception:
        return pd.NA, "error_abriendo_archivo"

    # 1) buscar algo explícito
    for sh in sheet_names:
        if "ocup" in _slug(sh):
            try:
                df = _read_excel_sheet(path, sh)
                for c in df.columns:
                    if "ocup" in _slug(c):
                        vals = _to_num(df[c]).dropna()
                        if not vals.empty:
                            val = float(vals.iloc[0])
                            # si ya viene como 0-1, pasar a %
                            if 0 <= val <= 1:
                                val *= 100
                            return val, f"hoja:{sh}|col:{c}"
            except Exception:
                pass

    # 2) inferir desde inventario inicial
    if TOTAL_PATIO_CAPACITY_TEU is None or TOTAL_PATIO_CAPACITY_TEU <= 0:
        return pd.NA, "no_configurado"

    inv_sheet_candidates = ["i0", "inventarioinicial", "inventario inicial", "inventario"]
    for sh in sheet_names:
        if _slug(sh) in {_slug(x) for x in inv_sheet_candidates} or "invent" in _slug(sh) or _slug(sh) == "i0":
            try:
                df = _read_excel_sheet(path, sh)
                # buscar columna tipo inventario/cantidad/stock
                num_candidates = [
                    "I0", "inventario", "inventario_inicial", "cantidad", "stock", "contenedores", "valor"
                ]
                c_val = _find_col(df, num_candidates, required=False)

                if c_val is None:
                    # fallback: tomar todas las numéricas y sumar solo si parece razonable
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(_to_num(df[c]))]
                    if len(num_cols) == 1:
                        c_val = num_cols[0]

                if c_val is not None:
                    tot = _to_num(df[c_val]).fillna(0).sum()
                    occ = (tot / TOTAL_PATIO_CAPACITY_TEU) * 100.0
                    return float(occ), f"inferido:{sh}|col:{c_val}"
            except Exception:
                pass

    return pd.NA, "no_detectado"


# =========================================================
# ESTADÍSTICAS
# =========================================================
def build_stats_table(df: pd.DataFrame, metrics: List[str], week_col: str = "Semana_ISO") -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        s = pd.to_numeric(df[metric], errors="coerce")
        s_valid = s.dropna()
        if s_valid.empty:
            rows.append({
                "Metrica": metric,
                "Min": pd.NA,
                "Semana_Min": pd.NA,
                "Max": pd.NA,
                "Semana_Max": pd.NA,
                "Promedio": pd.NA,
                "Std": pd.NA
            })
            continue

        idx_min = s.idxmin()
        idx_max = s.idxmax()

        rows.append({
            "Metrica": metric,
            "Min": float(s.loc[idx_min]),
            "Semana_Min": df.loc[idx_min, week_col],
            "Max": float(s.loc[idx_max]),
            "Semana_Max": df.loc[idx_max, week_col],
            "Promedio": float(s.mean()),
            "Std": float(s.std(ddof=0))
        })

    return pd.DataFrame(rows)


def build_critical_weeks(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "N_Seg_Instancia", "N_Reefer_Instancia", "N_Primary_Instancia", "N_Secondary_Instancia",
        "Ocupacion_Instancia_pct",
        "RECV_Instancia", "LOAD_Instancia", "DSCH_Instancia", "DLVR_Instancia", "YARD_Instancia",
        "RECV_All", "LOAD_All", "DSCH_All", "DLVR_All", "YARD_All",
        "Actividad_Total_Instancia", "Actividad_Total_All"
    ]
    rows = []
    for metric in wanted:
        if metric not in df.columns:
            continue
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        if s.empty:
            continue

        idx_max = pd.to_numeric(df[metric], errors="coerce").idxmax()
        idx_min = pd.to_numeric(df[metric], errors="coerce").idxmin()

        rows.append({
            "Metrica": metric,
            "Tipo": "MAX",
            "Semana_Fecha": df.loc[idx_max, "Semana"],
            "Semana_ISO": df.loc[idx_max, "Semana_ISO"],
            "Valor": df.loc[idx_max, metric]
        })
        rows.append({
            "Metrica": metric,
            "Tipo": "MIN",
            "Semana_Fecha": df.loc[idx_min, "Semana"],
            "Semana_ISO": df.loc[idx_min, "Semana_ISO"],
            "Valor": df.loc[idx_min, metric]
        })

    return pd.DataFrame(rows)


def build_paragraph_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # ---------- Párrafo 1 ----------
    seg_min = pd.to_numeric(df["N_Seg_Instancia"], errors="coerce").min()
    seg_max = pd.to_numeric(df["N_Seg_Instancia"], errors="coerce").max()
    seg_avg = pd.to_numeric(df["N_Seg_Instancia"], errors="coerce").mean()

    reefer_min = pd.to_numeric(df["N_Reefer_Instancia"], errors="coerce").min()
    reefer_max = pd.to_numeric(df["N_Reefer_Instancia"], errors="coerce").max()
    reefer_avg = pd.to_numeric(df["N_Reefer_Instancia"], errors="coerce").mean()

    share_reefer = np.nan
    if not pd.isna(seg_avg) and seg_avg != 0 and not pd.isna(reefer_avg):
        share_reefer = reefer_avg / seg_avg

    primary_ok = "N_Primary_Instancia" in df.columns and pd.to_numeric(df["N_Primary_Instancia"], errors="coerce").notna().any()
    secondary_ok = "N_Secondary_Instancia" in df.columns and pd.to_numeric(df["N_Secondary_Instancia"], errors="coerce").notna().any()

    if primary_ok and secondary_ok:
        p_avg = pd.to_numeric(df["N_Primary_Instancia"], errors="coerce").mean()
        s_avg = pd.to_numeric(df["N_Secondary_Instancia"], errors="coerce").mean()
        txt1 = (
            f"A crucial aspect of our proposed approach is the allocation of storage space to container groups, "
            f"or segregations. In the analyzed instances, the number of selected segregations ranged from "
            f"{_fmt_num(seg_min, 0)} to {_fmt_num(seg_max, 0)} per week, with an average of {_fmt_num(seg_avg, 1)}. "
            f"Reefer segregations accounted for {_fmt_num(share_reefer * 100, 1)}% of the total on average, "
            f"ranging from {_fmt_num(reefer_min, 0)} to {_fmt_num(reefer_max, 0)} per week. "
            f"Primary segregations averaged {_fmt_num(p_avg, 1)} per week, while secondary segregations averaged "
            f"{_fmt_num(s_avg, 1)}."
        )
    else:
        txt1 = (
            f"A crucial aspect of our proposed approach is the allocation of storage space to container groups, "
            f"or segregations. In the analyzed instances, the number of selected segregations ranged from "
            f"{_fmt_num(seg_min, 0)} to {_fmt_num(seg_max, 0)} per week, with an average of {_fmt_num(seg_avg, 1)}. "
            f"Reefer segregations accounted for {_fmt_num(share_reefer * 100, 1)}% of the total on average, "
            f"ranging from {_fmt_num(reefer_min, 0)} to {_fmt_num(reefer_max, 0)} per week."
        )

    rows.append({"Parrafo": "P1", "Texto": txt1})

    # ---------- Párrafo 2 ----------
    def _max_row(metric):
        s = pd.to_numeric(df[metric], errors="coerce")
        idx = s.idxmax()
        return df.loc[idx]

    def _min_row(metric):
        s = pd.to_numeric(df[metric], errors="coerce")
        idx = s.idxmin()
        return df.loc[idx]

    r_seg = _max_row("N_Seg_Instancia")
    r_recv = _max_row("RECV_Instancia")
    r_load = _max_row("LOAD_Instancia")
    r_dsch = _max_row("DSCH_Instancia")
    r_dlvr = _max_row("DLVR_Instancia")
    r_min = _min_row("Actividad_Total_Instancia")

    occ_clause = ""
    if "Ocupacion_Instancia_pct" in df.columns and pd.to_numeric(df["Ocupacion_Instancia_pct"], errors="coerce").notna().any():
        r_occ = _max_row("Ocupacion_Instancia_pct")
        occ_clause = (
            f" while Week {int(r_occ['Semana_ISO'])} exhibited the maximum yard occupancy "
            f"({_fmt_pct(r_occ['Ocupacion_Instancia_pct'], 1)})"
        )

    txt2 = (
        f"Several critical operational weeks were identified throughout the year. "
        f"Week {int(r_seg['Semana_ISO'])} recorded the highest number of selected segregations "
        f"({_fmt_num(r_seg['N_Seg_Instancia'], 0)}){occ_clause}. "
        f"Week {int(r_recv['Semana_ISO'])} reached the maximum reception volume "
        f"({_fmt_num(r_recv['RECV_Instancia'], 0)} boxes), "
        f"Week {int(r_load['Semana_ISO'])} the maximum loading volume "
        f"({_fmt_num(r_load['LOAD_Instancia'], 0)} boxes), "
        f"Week {int(r_dsch['Semana_ISO'])} the maximum unloading volume "
        f"({_fmt_num(r_dsch['DSCH_Instancia'], 0)} boxes), and "
        f"Week {int(r_dlvr['Semana_ISO'])} the maximum delivery volume "
        f"({_fmt_num(r_dlvr['DLVR_Instancia'], 0)} boxes). "
        f"In contrast, Week {int(r_min['Semana_ISO'])} represented the lowest-activity period, "
        f"with a total of {_fmt_num(r_min['Actividad_Total_Instancia'], 0)} movements across the four main flows."
    )

    rows.append({"Parrafo": "P2", "Texto": txt2})

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    root = Path(ROOT_SEMANAS)
    out_dir = Path(OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    semanas = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-"
    ])
    if not semanas:
        raise SystemExit(f"No encontré carpetas de semanas en {ROOT_SEMANAS}")

    resumen_rows = []
    detalle_temporal_rows = []
    advertencias = []

    for semana in semanas:
        carpeta = root / semana

        instancia_path = _find_best_file(str(carpeta), INSTANCIA_GLOB)
        flow_path = _find_best_file(str(carpeta), FLOW_GLOB)

        if not instancia_path:
            advertencias.append({"Semana": semana, "Tipo": "MISSING_INSTANCIA", "Detalle": INSTANCIA_GLOB})
            continue
        if not flow_path:
            advertencias.append({"Semana": semana, "Tipo": "MISSING_FLUJOS", "Detalle": FLOW_GLOB})
            continue

        # -------------------------
        # Hoja S
        # -------------------------
        try:
            s_df = _read_excel_sheet(instancia_path, INSTANCIA_SHEET_S)
        except Exception as e:
            advertencias.append({"Semana": semana, "Tipo": "ERROR_LEYENDO_S", "Detalle": str(e)})
            continue

        try:
            seg_col = _find_col(s_df, ["Segregacion", "Segregación", "criterio"])
        except Exception as e:
            advertencias.append({"Semana": semana, "Tipo": "SIN_COLUMNA_SEG", "Detalle": str(e)})
            continue

        s_df[seg_col] = _upper_clean(s_df[seg_col])
        s_df = s_df[s_df[seg_col].notna() & (s_df[seg_col] != "")]
        if s_df.empty:
            advertencias.append({"Semana": semana, "Tipo": "HOJA_S_VACIA", "Detalle": os.path.basename(instancia_path)})
            continue

        # dejamos una fila por segregación
        base = s_df.groupby(seg_col, as_index=False).first()

        # -------------------------
        # Flujos agregados
        # -------------------------
        try:
            flows_by_seg = read_flujos_aggregated(flow_path)
        except Exception as e:
            advertencias.append({"Semana": semana, "Tipo": "ERROR_LEYENDO_FLUJOS", "Detalle": str(e)})
            continue

        seg_inst = set(base[seg_col].astype(str).str.upper().tolist())
        seg_all = set(flows_by_seg["Segregacion"].astype(str).str.upper().tolist())

        flows_inst = flows_by_seg[flows_by_seg["Segregacion"].isin(seg_inst)].copy()

        # -------------------------
        # Reefer
        # -------------------------
        reefer_mask, reefer_source = detect_reefer(base, seg_col)

        # -------------------------
        # Temporalidad
        # -------------------------
        temporal_cls, temporal_source = detect_temporal_class(base, seg_col, flows_by_seg=flows_inst)

        detail_temp = base[[seg_col]].copy()
        detail_temp["Semana"] = semana
        detail_temp["Reefer"] = reefer_mask.astype("boolean")
        detail_temp["Temporalidad"] = temporal_cls
        detail_temp["Fuente_Reefer"] = reefer_source
        detail_temp["Fuente_Temporalidad"] = temporal_source
        detalle_temporal_rows.append(detail_temp)

        n_primary = pd.to_numeric(pd.Series((temporal_cls == "PRIMARY").astype(int)), errors="coerce").sum() if temporal_cls.notna().any() else pd.NA
        n_secondary = pd.to_numeric(pd.Series((temporal_cls == "SECONDARY").astype(int)), errors="coerce").sum() if temporal_cls.notna().any() else pd.NA
        n_unknown_temp = int(temporal_cls.isna().sum())

        # -------------------------
        # Ocupación (opcional)
        # -------------------------
        occ_val, occ_source = try_read_occupancy_from_instance(instancia_path)

        # -------------------------
        # Resumen de movimientos
        # -------------------------
        def _sumcol(df_, c):
            if c not in df_.columns or df_.empty:
                return 0
            return int(pd.to_numeric(df_[c], errors="coerce").fillna(0).sum())

        recv_all = _sumcol(flows_by_seg, "RECV")
        load_all = _sumcol(flows_by_seg, "LOAD")
        dsch_all = _sumcol(flows_by_seg, "DSCH")
        dlvr_all = _sumcol(flows_by_seg, "DLVR")
        yard_all = _sumcol(flows_by_seg, "YARD")

        recv_inst = _sumcol(flows_inst, "RECV")
        load_inst = _sumcol(flows_inst, "LOAD")
        dsch_inst = _sumcol(flows_inst, "DSCH")
        dlvr_inst = _sumcol(flows_inst, "DLVR")
        yard_inst = _sumcol(flows_inst, "YARD")

        dt_sem, semana_iso = _week_from_folder_name(semana)

        resumen_rows.append({
            "Semana": semana,
            "Fecha_Semana": dt_sem.strftime("%Y-%m-%d") if dt_sem is not None else semana,
            "Semana_ISO": semana_iso,

            "Archivo_Instancia": os.path.basename(instancia_path),
            "Archivo_Flujos": os.path.basename(flow_path),

            "N_Seg_All": len(seg_all),
            "N_Seg_Instancia": len(seg_inst),
            "N_Reefer_Instancia": int(reefer_mask.sum()),
            "N_Primary_Instancia": n_primary,
            "N_Secondary_Instancia": n_secondary,
            "N_Temporal_Unknown": n_unknown_temp,

            "Fuente_Reefer": reefer_source,
            "Fuente_Temporalidad": temporal_source,

            "Ocupacion_Instancia_pct": occ_val,
            "Fuente_Ocupacion": occ_source,

            "RECV_All": recv_all,
            "LOAD_All": load_all,
            "DSCH_All": dsch_all,
            "DLVR_All": dlvr_all,
            "YARD_All": yard_all,
            "Actividad_Total_All": recv_all + load_all + dsch_all + dlvr_all,

            "RECV_Instancia": recv_inst,
            "LOAD_Instancia": load_inst,
            "DSCH_Instancia": dsch_inst,
            "DLVR_Instancia": dlvr_inst,
            "YARD_Instancia": yard_inst,
            "Actividad_Total_Instancia": recv_inst + load_inst + dsch_inst + dlvr_inst,
        })

        print(f"[OK] {semana} | seg_inst={len(seg_inst)} | flows_inst={recv_inst + load_inst + dsch_inst + dlvr_inst}")

    # =====================================================
    # DataFrames finales
    # =====================================================
    df_resumen = pd.DataFrame(resumen_rows)
    if df_resumen.empty:
        raise SystemExit("No se generó ningún resultado. Revisa rutas, patrones o nombres de hojas.")

    df_resumen = df_resumen.sort_values(by=["Fecha_Semana"], kind="stable").reset_index(drop=True)

    df_temp = pd.concat(detalle_temporal_rows, ignore_index=True) if detalle_temporal_rows else pd.DataFrame()
    df_warn = pd.DataFrame(advertencias) if advertencias else pd.DataFrame()

    stats_segs = build_stats_table(
        df_resumen,
        metrics=[
            "N_Seg_Instancia", "N_Reefer_Instancia",
            "N_Primary_Instancia", "N_Secondary_Instancia",
            "Ocupacion_Instancia_pct"
        ]
    )

    stats_mov_inst = build_stats_table(
        df_resumen,
        metrics=[
            "RECV_Instancia", "LOAD_Instancia", "DSCH_Instancia", "DLVR_Instancia",
            "YARD_Instancia", "Actividad_Total_Instancia"
        ]
    )

    stats_mov_all = build_stats_table(
        df_resumen,
        metrics=[
            "RECV_All", "LOAD_All", "DSCH_All", "DLVR_All",
            "YARD_All", "Actividad_Total_All"
        ]
    )

    critical = build_critical_weeks(df_resumen)
    paragraphs = build_paragraph_suggestions(df_resumen)

    # Redondeo
    for df_ in [df_resumen, stats_segs, stats_mov_inst, stats_mov_all, critical]:
        for c in df_.columns:
            if c == "Semana_ISO":
                continue
            if pd.api.types.is_float_dtype(df_[c]):
                df_[c] = df_[c].round(ROUND_DECIMALS)

    # =====================================================
    # Exportar
    # =====================================================
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="Resumen_Semanal", index=False)
        stats_segs.to_excel(writer, sheet_name="Stats_Segregaciones", index=False)
        stats_mov_inst.to_excel(writer, sheet_name="Stats_Mov_Instancia", index=False)
        stats_mov_all.to_excel(writer, sheet_name="Stats_Mov_Terminal", index=False)
        critical.to_excel(writer, sheet_name="Semanas_Criticas", index=False)
        paragraphs.to_excel(writer, sheet_name="Parrafos_Sugeridos", index=False)

        if not df_temp.empty:
            df_temp.to_excel(writer, sheet_name="Detalle_Temporalidad", index=False)
        if not df_warn.empty:
            df_warn.to_excel(writer, sheet_name="Advertencias", index=False)

    print("\n===================================================")
    print(f"Listo. Archivo guardado en: {OUT_XLSX}")
    print("Revisa especialmente estas hojas:")
    print("- Resumen_Semanal")
    print("- Stats_Segregaciones")
    print("- Stats_Mov_Instancia")
    print("- Semanas_Criticas")
    print("- Parrafos_Sugeridos")
    print("===================================================")


if __name__ == "__main__":
    main()