import os
import glob
import pandas as pd
from pathlib import Path
from collections import defaultdict

# =========================
# CONFIG
# =========================

ROOT_SEMANAS = r"resultados_generados_pila_criterio_iii/instancias_magdalena"
OUTPUT = r"resultados_generados_pila_criterio_iii_test_año_nofiltro/metrics"

ROOT_RESULTADOS_MAGDALENA = r"resultados_generados_pila_criterio_iii/resultados_magdalena"
DIST_MODELO_FILENAME_FMT = "Distancias_Modelo_{semana}_68.xlsx"
DIST_MODELO_GLOB = "Distancias_Modelo_*_68.xlsx"
DIST_MODELO_SHEET = None
DIST_MODELO_COLS = [
    "Semana",
    "Distancia Total",
    "Distancia LOAD",
    "Distancia DLVR",
    "Movimientos_DLVR",
    "Movimientos_LOAD",
]

FLOW_GLOB = "analisis_flujos*.xlsx"
FLOW_SHEET = "FlujosAll_sbt"   # hoja usada para distancias
FLOW_SHEET_SB = "FlujosAll_sb_P"  # hoja usada para filtro participación

DISTANCIAS_XLSX = r"archivos_estaticos/Distancias_GranPatio.xlsx"
DIST_SHEET = "Distancias"
DIST_COL_FROM = "ime_fm"
DIST_COL_TO   = "ime_to"
DIST_COL_VAL  = "Distancia[m]"

# Instancia (para filtro)
USAR_INSTANCIA = True
INSTANCIA_GLOB = "Instancia_*_K.xlsx"
INSTANCIA_SHEET_S = "S"
INSTANCIA_COL_SEG = "Segregacion"

# =====  filtro tipo instancia (participación + mínimos) =====
MODO_FILTRO_PART = True
PARTICIPACION_C = 0  # % Patio

# mínimos
MIN_LOAD = 15
MIN_RECV = 25
MIN_DLVR = 35
MIN_DSCH = 1

BLOQUES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8","C9","H1","H2","H3","H4","H5","T1","T2","T3","T4", "I1","I2"]
SITIOS_MUELLE = ['Y-SAI-1', 'Y-SAI-2']
GATE = 'GATE'

COUNT_YARD_ONLY_IF_HAS_LOAD_DLVR = True
DEFAULT_DIST_IF_MISSING = 0.0

EXPORTAR_EXCLUIDOS = True
MAX_EXCLUIDOS = None

OUT_XLSX = os.path.join(OUTPUT, "Distancias.xlsx")


# =========================
# HELPERS
# =========================

def _sanitize_numeric_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if str(c).lower() in {
            "modo", "semana", "segregacion", "archivo_flujos", "archivo_origen",
            "tipo", "origen", "destino", "ime_fm", "ime_to", "motivo",
            "__rowid__", "carrier", "shift", "criterio",
            "ime_fm_norm", "ime_to_norm", "ruta_flujos"
        }:
            continue
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        s_txt = s.astype(str).str.strip()
        s_clean = s_txt.str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        s_num = pd.to_numeric(s_clean, errors="coerce")
        if s_num.notna().any():
            out[c] = s_num
    return out

def _write_sheet_no_grouping(writer, df: pd.DataFrame, sheet_name: str):
    if df is None or df.empty:
        return
    df_out = _sanitize_numeric_display_df(df)
    sname = sheet_name[:31]
    df_out.to_excel(writer, sheet_name=sname, index=False)
    ws = writer.sheets[sname]
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            v = cell.value
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                cell.number_format = "0"

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _upper_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _read_excel_sheet(path: str, sheet_name=None) -> pd.DataFrame:
    if sheet_name is None:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        first_name = list(xls.keys())[0]
        return xls[first_name]
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

def _to_numeric_flexible(s: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(s, errors="coerce")
    if s1.notna().any():
        return s1
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s2, errors="coerce")

def norm_node(x) -> str:
    return str(x).strip().upper()

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

def unique_preserve(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_dist_map(dist_xlsx: str) -> dict:
    dist = pd.read_excel(dist_xlsx, sheet_name=DIST_SHEET, engine="openpyxl")
    dist = _norm_cols(dist)

    needed = {DIST_COL_FROM, DIST_COL_TO, DIST_COL_VAL}
    missing = needed - set(dist.columns)
    if missing:
        raise ValueError(f"Faltan columnas en distancias: {missing}. Disponibles: {list(dist.columns)}")

    dist[DIST_COL_FROM] = _upper_series(dist[DIST_COL_FROM])
    dist[DIST_COL_TO]   = _upper_series(dist[DIST_COL_TO])
    dist[DIST_COL_VAL]  = pd.to_numeric(dist[DIST_COL_VAL], errors="coerce").fillna(0.0)

    m = {}
    for _, r in dist.iterrows():
        m[(r[DIST_COL_FROM], r[DIST_COL_TO])] = float(r[DIST_COL_VAL])
    return m

def find_best_file(folder: str, pattern: str):
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

def read_flujos(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        return _norm_cols(df)

    if FLOW_SHEET is None:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        first_name = list(xls.keys())[0]
        df = xls[first_name]
    else:
        df = pd.read_excel(path, sheet_name=FLOW_SHEET, engine="openpyxl")

    df = _norm_cols(df)
    df["__rowid__"] = range(len(df))
    return df

def get_dist(dist_map: dict, fm: str, to: str) -> float:
    fm = norm_node(fm)
    to = norm_node(to)
    if (fm, to) in dist_map:
        return float(dist_map[(fm, to)])
    if (to, fm) in dist_map:
        return float(dist_map[(to, fm)])
    return float(DEFAULT_DIST_IF_MISSING)

def read_distancias_modelo_68(path: str, semana_folder: str) -> pd.DataFrame:
    df = _read_excel_sheet(path, DIST_MODELO_SHEET)
    df = _norm_cols(df)
    df = df.dropna(how="all").copy()
    if df.empty:
        return pd.DataFrame(columns=DIST_MODELO_COLS + ["Archivo_origen"])

    rename_map = {
        "Distancia_Total": "Distancia Total",
        "Distancia_LOAD": "Distancia LOAD",
        "Distancia_DLVR": "Distancia DLVR",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = set(DIST_MODELO_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"En {path} faltan columnas {missing}. Disponibles: {list(df.columns)}")

    out = df[DIST_MODELO_COLS].copy()
    semana_parse = pd.to_datetime(out["Semana"], errors="coerce")
    out["Semana"] = semana_parse.dt.strftime("%Y-%m-%d")
    out["Semana"] = out["Semana"].fillna(str(semana_folder))

    for c in ["Distancia Total", "Distancia LOAD", "Distancia DLVR", "Movimientos_DLVR", "Movimientos_LOAD"]:
        out[c] = _to_numeric_flexible(out[c])

    for c in ["Distancia Total", "Distancia LOAD", "Distancia DLVR", "Movimientos_DLVR", "Movimientos_LOAD"]:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(0).astype("Int64")

    if len(out) > 1:
        matched = out[out["Semana"] == semana_folder].copy()
        if not matched.empty:
            out = matched
        print(f"[WARN] {os.path.basename(path)} tiene {len(df)} filas. Se usarán {len(out)} fila(s).")

    out["Archivo_origen"] = os.path.basename(path)
    return out

def build_segs_por_participacion(df_sb: pd.DataFrame,
                                 participacion_pct: float,
                                 min_load=15, min_recv=25, min_dlvr=35, min_dsch=0) -> list[str]:
    """
    Replica el filtro del generador de instancias usando FlujosAll_sb_P (que trae 'Patio').
    - LOAD/DLVR: ratio con ime_fm == PATIO
    - RECV/DSCH: ratio con ime_to == PATIO
    - además total >= min_value
    """
    df = _norm_cols(df_sb)
    df = _ensure_cols(df, ["criterio", "ime_fm", "ime_to", "LOAD", "RECV", "DLVR", "DSCH"])

    df["criterio"] = df["criterio"].astype(str).str.strip()
    df["ime_fm"] = df["ime_fm"].astype(str).str.strip().str.upper()
    df["ime_to"] = df["ime_to"].astype(str).str.strip().str.upper()

    for c in ["LOAD", "RECV", "DLVR", "DSCH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    thr = float(participacion_pct) / 100.0
    PATIO = "PATIO"

    def analyze(mov_col: str, loc_col: str, min_value: int) -> list[str]:
        tot = df.groupby("criterio")[mov_col].sum()
        ok_tot = tot[tot >= min_value].index
        if len(ok_tot) == 0:
            return []
        d2 = df[df["criterio"].isin(ok_tot)]
        patio = d2[d2[loc_col] == PATIO].groupby("criterio")[mov_col].sum()
        total = d2.groupby("criterio")[mov_col].sum()
        ratio = (patio / total).fillna(0.0)
        return ratio[ratio >= thr].index.tolist()

    load_ok = analyze("LOAD", "ime_fm", min_load)
    dsch_ok = analyze("DSCH", "ime_to", min_dsch)
    recv_ok = analyze("RECV", "ime_to", min_recv)
    dlvr_ok = analyze("DLVR", "ime_fm", min_dlvr)

    return unique_preserve(load_ok + dsch_ok + recv_ok + dlvr_ok)


# =========================
# MAIN
# =========================

def main():
    BLOQUES_SET = {b.upper() for b in BLOQUES}
    MUELLE_SET  = {m.upper() for m in SITIOS_MUELLE}
    GATE_UP     = GATE.upper()

    dist_map = load_dist_map(DISTANCIAS_XLSX)

    root_semanas = Path(ROOT_SEMANAS)
    out_dir = Path(OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    semanas = sorted([
        d.name for d in root_semanas.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-"
    ])
    if not semanas:
        raise SystemExit(f"No encontré subcarpetas de semanas en {ROOT_SEMANAS}")

    # Outputs globales (con columna Modo)
    resultados = []
    resumen_semanal = []
    detalle_mov = []
    missing_dist_rows = []
    resumen_modelo_68 = []

    excluidos_rows = []
    excluidos_resumen = defaultdict(int)

    seg_fuera_instancia_rows = []
    seg_fuera_filtro_rows = []
    reconciliacion_rows = []

    exportar_excluidos = EXPORTAR_EXCLUIDOS

    def _sum_excluidos_semana(modo: str, semana_str: str, tipo: str) -> int:
        total = 0
        for (m, sem, t, _motivo), mv in excluidos_resumen.items():
            if m == modo and sem == semana_str and t == tipo:
                total += int(mv)
        return total

    def procesar_modo(modo: str, semana: str, flujos: pd.DataFrame, segregaciones: list,
                      c_criterio, c_fm, c_to, c_dlvr, c_load, c_yard,
                      c_carrier, c_shift,
                      raw_totales: dict):
        """Procesa un subconjunto de segregaciones y agrega filas a las tablas globales."""
        nonlocal exportar_excluidos

        dist_sem = {"DLVR": 0.0, "LOAD": 0.0, "YARD": 0.0}
        mov_sem  = {"DLVR": 0,   "LOAD": 0,   "YARD": 0}

        # raw del modo (solo segs procesadas)
        raw_mode_dlvr = int(flujos.loc[flujos[c_criterio].isin(segregaciones), c_dlvr].sum()) if segregaciones else 0
        raw_mode_load = int(flujos.loc[flujos[c_criterio].isin(segregaciones), c_load].sum()) if segregaciones else 0
        raw_mode_yard = int(flujos.loc[flujos[c_criterio].isin(segregaciones), c_yard].sum()) if segregaciones else 0

        # Contadores
        n_seg_total = len(segregaciones)
        if segregaciones:
            tmp_mov = (
                flujos.loc[flujos[c_criterio].isin(segregaciones)]
                     .groupby(c_criterio)[[c_dlvr, c_load, c_yard]]
                     .sum()
            )
            n_seg_con_mov = int((tmp_mov.sum(axis=1) > 0).sum()) if not tmp_mov.empty else 0
        else:
            n_seg_con_mov = 0

        for seg in segregaciones:
            fseg = flujos[flujos[c_criterio] == seg].copy()
            if fseg.empty:
                continue

            tiene_load_o_dlvr = ((fseg[c_load] > 0) | (fseg[c_dlvr] > 0)).any()

            dist_seg = {"DLVR": 0.0, "LOAD": 0.0, "YARD": 0.0}
            mov_seg  = {"DLVR": 0,   "LOAD": 0,   "YARD": 0}

            for r in fseg.itertuples(index=False):
                row = r._asdict()

                rowid = row.get("__rowid__", None)
                fm = row.get(c_fm)
                to = row.get(c_to)

                fm_norm = norm_node(fm)
                to_norm = norm_node(to)

                dlvr_v = int(row.get(c_dlvr, 0) or 0)
                load_v = int(row.get(c_load, 0) or 0)
                yard_v = int(row.get(c_yard, 0) or 0)

                carrier = row.get(c_carrier) if c_carrier else None
                shift   = row.get(c_shift) if c_shift else None

                d = get_dist(dist_map, fm_norm, to_norm)

                if d == DEFAULT_DIST_IF_MISSING and (fm_norm, to_norm) not in dist_map and (to_norm, fm_norm) not in dist_map:
                    missing_dist_rows.append({
                        "Modo": modo, "Semana": semana, "Segregacion": seg,
                        "ime_fm": fm, "ime_to": to,
                        "ime_fm_norm": fm_norm, "ime_to_norm": to_norm,
                        "__rowid__": rowid, "carrier": carrier, "shift": shift,
                    })

                # DLVR
                if dlvr_v > 0:
                    ok_dlvr = (fm_norm in BLOQUES_SET) and (to_norm == GATE_UP)
                    if ok_dlvr:
                        mov_seg["DLVR"] += dlvr_v
                        dist_seg["DLVR"] += dlvr_v * d
                        detalle_mov.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "DLVR",
                            "Origen": fm_norm, "Destino": to_norm,
                            "Movimientos": dlvr_v, "Distancia_unit_m": d, "Distancia_total_m": dlvr_v * d,
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                        })
                    elif exportar_excluidos:
                        motivo = "DLVR_fm_no_bloque" if (fm_norm not in BLOQUES_SET) else "DLVR_to_no_gate"
                        excluidos_rows.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "DLVR",
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                            "ime_fm": fm, "ime_to": to, "ime_fm_norm": fm_norm, "ime_to_norm": to_norm,
                            "Movimientos": dlvr_v, "Distancia_unit_m": d, "Motivo": motivo,
                        })
                        excluidos_resumen[(modo, semana, "DLVR", motivo)] += dlvr_v

                # LOAD
                if load_v > 0:
                    ok_load = (fm_norm in BLOQUES_SET) and (to_norm in MUELLE_SET)
                    if ok_load:
                        mov_seg["LOAD"] += load_v
                        dist_seg["LOAD"] += load_v * d
                        detalle_mov.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "LOAD",
                            "Origen": fm_norm, "Destino": to_norm,
                            "Movimientos": load_v, "Distancia_unit_m": d, "Distancia_total_m": load_v * d,
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                        })
                    elif exportar_excluidos:
                        motivo = "LOAD_fm_no_bloque" if (fm_norm not in BLOQUES_SET) else "LOAD_to_no_muelle"
                        excluidos_rows.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "LOAD",
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                            "ime_fm": fm, "ime_to": to, "ime_fm_norm": fm_norm, "ime_to_norm": to_norm,
                            "Movimientos": load_v, "Distancia_unit_m": d, "Motivo": motivo,
                        })
                        excluidos_resumen[(modo, semana, "LOAD", motivo)] += load_v

                # YARD
                if yard_v > 0:
                    ok_pair = (fm_norm in BLOQUES_SET) and (to_norm in BLOQUES_SET)
                    if ok_pair:
                        if COUNT_YARD_ONLY_IF_HAS_LOAD_DLVR and not tiene_load_o_dlvr:
                            if exportar_excluidos:
                                motivo = "YARD_excl_no_load_dlvr_seg"
                                excluidos_rows.append({
                                    "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "YARD",
                                    "__rowid__": rowid, "carrier": carrier, "shift": shift,
                                    "ime_fm": fm, "ime_to": to, "ime_fm_norm": fm_norm, "ime_to_norm": to_norm,
                                    "Movimientos": yard_v, "Distancia_unit_m": d, "Motivo": motivo,
                                })
                                excluidos_resumen[(modo, semana, "YARD", motivo)] += yard_v
                            continue

                        mov_seg["YARD"] += yard_v
                        dist_seg["YARD"] += yard_v * d
                        detalle_mov.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "YARD",
                            "Origen": fm_norm, "Destino": to_norm,
                            "Movimientos": yard_v, "Distancia_unit_m": d, "Distancia_total_m": yard_v * d,
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                        })
                    elif exportar_excluidos:
                        motivo = "YARD_excl_non_bloque_pair"
                        excluidos_rows.append({
                            "Modo": modo, "Semana": semana, "Segregacion": seg, "Tipo": "YARD",
                            "__rowid__": rowid, "carrier": carrier, "shift": shift,
                            "ime_fm": fm, "ime_to": to, "ime_fm_norm": fm_norm, "ime_to_norm": to_norm,
                            "Movimientos": yard_v, "Distancia_unit_m": d, "Motivo": motivo,
                        })
                        excluidos_resumen[(modo, semana, "YARD", motivo)] += yard_v

                if MAX_EXCLUIDOS is not None and len(excluidos_rows) >= MAX_EXCLUIDOS:
                    exportar_excluidos = False
                    print(f"[WARN] MAX_EXCLUIDOS alcanzado ({MAX_EXCLUIDOS}). Se dejarán de registrar excluidos.")

            resultados.append({
                "Modo": modo,
                "Semana": semana,
                "Segregacion": seg,
                "Distancia_Total": dist_seg["DLVR"] + dist_seg["LOAD"] + dist_seg["YARD"],
                "Distancia_DLVR": dist_seg["DLVR"],
                "Distancia_LOAD": dist_seg["LOAD"],
                "Distancia_YARD": dist_seg["YARD"],
                "Movimientos_DLVR": mov_seg["DLVR"],
                "Movimientos_LOAD": mov_seg["LOAD"],
                "Movimientos_YARD": mov_seg["YARD"],
            })

            for k in dist_sem:
                dist_sem[k] += dist_seg[k]
            for k in mov_sem:
                mov_sem[k] += mov_seg[k]

        resumen_semanal.append({
            "Modo": modo,
            "Semana": semana,
            "N_Segregaciones": n_seg_total,
            "N_Seg_con_mov": n_seg_con_mov,
            "Distancia_Total": dist_sem["DLVR"] + dist_sem["LOAD"] + dist_sem["YARD"],
            "Distancia_DLVR": dist_sem["DLVR"],
            "Distancia_LOAD": dist_sem["LOAD"],
            "Distancia_YARD": dist_sem["YARD"],
            "Movimientos_DLVR": mov_sem["DLVR"],
            "Movimientos_LOAD": mov_sem["LOAD"],
            "Movimientos_YARD": mov_sem["YARD"],
            "RAW_TOTAL_LOAD": raw_totales["LOAD"],
            "RAW_MODO_LOAD": raw_mode_load,
            "RAW_TOTAL_DLVR": raw_totales["DLVR"],
            "RAW_MODO_DLVR": raw_mode_dlvr,
            "RAW_TOTAL_YARD": raw_totales["YARD"],
            "RAW_MODO_YARD": raw_mode_yard,
        })

        excl_dlvr = _sum_excluidos_semana(modo, semana, "DLVR")
        excl_load = _sum_excluidos_semana(modo, semana, "LOAD")
        excl_yard = _sum_excluidos_semana(modo, semana, "YARD")

        reconciliacion_rows.append({
            "Modo": modo,
            "Semana": semana,
            "N_Segregaciones": n_seg_total,
            "N_Seg_con_mov": n_seg_con_mov,
            "RAW_MODO_DLVR": raw_mode_dlvr, "INC_DLVR": mov_sem["DLVR"], "EXC_DLVR": excl_dlvr,
            "NO_PROC_DLVR": raw_mode_dlvr - mov_sem["DLVR"] - excl_dlvr,

            "RAW_MODO_LOAD": raw_mode_load, "INC_LOAD": mov_sem["LOAD"], "EXC_LOAD": excl_load,
            "NO_PROC_LOAD": raw_mode_load - mov_sem["LOAD"] - excl_load,

            "RAW_MODO_YARD": raw_mode_yard, "INC_YARD": mov_sem["YARD"], "EXC_YARD": excl_yard,
            "NO_PROC_YARD": raw_mode_yard - mov_sem["YARD"] - excl_yard,

            "OMITIDOS_POR_FILTRO_LOAD": raw_totales["LOAD"] - raw_mode_load,
            "OMITIDOS_POR_FILTRO_DLVR": raw_totales["DLVR"] - raw_mode_dlvr,
            "OMITIDOS_POR_FILTRO_YARD": raw_totales["YARD"] - raw_mode_yard,
        })

    for semana in semanas:
        carpeta_semana = root_semanas / semana

        # Modelo 68 (hoja extra)
        carpeta_resultados_semana = Path(ROOT_RESULTADOS_MAGDALENA) / semana
        if carpeta_resultados_semana.is_dir():
            dist_modelo_path = carpeta_resultados_semana / DIST_MODELO_FILENAME_FMT.format(semana=semana)
            if not dist_modelo_path.exists():
                fallback = find_best_file(str(carpeta_resultados_semana), DIST_MODELO_GLOB)
                dist_modelo_path = Path(fallback) if fallback else None
            if dist_modelo_path:
                try:
                    df_m68 = read_distancias_modelo_68(str(dist_modelo_path), semana)
                    if not df_m68.empty:
                        resumen_modelo_68.extend(df_m68.to_dict(orient="records"))
                except Exception as e:
                    print(f"[WARN] Semana {semana}: error leyendo Distancias_Modelo_68 -> {e}")

        # Flujos (Excel analisis_flujos...)
        flujos_path = find_best_file(str(carpeta_semana), FLOW_GLOB)
        if not flujos_path:
            print(f"[SKIP] Semana {semana}: no encontré flujos con patrón {FLOW_GLOB}")
            continue

        flujos = read_flujos(flujos_path)

        cols_lower = {c.lower(): c for c in flujos.columns}
        def col(name: str) -> str:
            key = name.lower()
            if key not in cols_lower:
                raise ValueError(f"En {flujos_path} falta columna '{name}'. Disponibles: {list(flujos.columns)}")
            return cols_lower[key]

        c_criterio = col("criterio")
        c_fm = col("ime_fm")
        c_to = col("ime_to")
        c_dlvr = col("DLVR")
        c_load = col("LOAD")
        c_yard = col("YARD")

        c_carrier = cols_lower.get("carrier", None)
        c_shift   = cols_lower.get("shift", None)

        # Normaliza
        flujos[c_criterio] = flujos[c_criterio].astype(str).str.strip()
        flujos[c_fm] = _upper_series(flujos[c_fm])
        flujos[c_to] = _upper_series(flujos[c_to])
        for c in (c_dlvr, c_load, c_yard):
            flujos[c] = pd.to_numeric(flujos[c], errors="coerce").fillna(0).astype(int)

        seg_flujos = set(flujos[c_criterio].unique())

        # Instancia
        seg_inst = set()
        instancia_path = find_best_file(str(carpeta_semana), INSTANCIA_GLOB)
        if instancia_path and USAR_INSTANCIA:
            inst = pd.read_excel(instancia_path, sheet_name=INSTANCIA_SHEET_S, engine="openpyxl")
            inst = _norm_cols(inst)
            if INSTANCIA_COL_SEG not in inst.columns:
                raise ValueError(f"En {instancia_path} hoja {INSTANCIA_SHEET_S} falta columna {INSTANCIA_COL_SEG}")
            seg_inst = set(inst[INSTANCIA_COL_SEG].astype(str).str.strip().unique())

        # Auditoría seg fuera de instancia
        if seg_inst:
            fuera = sorted(seg_flujos - seg_inst)
            if fuera:
                df_fuera = (flujos[flujos[c_criterio].isin(fuera)]
                            .groupby(c_criterio)[[c_dlvr, c_load, c_yard]]
                            .sum()
                            .reset_index()
                            .rename(columns={c_criterio: "Segregacion", c_dlvr: "DLVR", c_load: "LOAD", c_yard: "YARD"}))
                df_fuera.insert(0, "Semana", semana)
                seg_fuera_instancia_rows.extend(df_fuera.to_dict(orient="records"))

        raw_totales = {
            "DLVR": int(flujos[c_dlvr].sum()),
            "LOAD": int(flujos[c_load].sum()),
            "YARD": int(flujos[c_yard].sum()),
        }

        # -------- Modo 1: SIN FILTRO
        procesar_modo(
            modo="SIN_FILTRO",
            semana=semana,
            flujos=flujos,
            segregaciones=sorted(seg_flujos),
            c_criterio=c_criterio, c_fm=c_fm, c_to=c_to,
            c_dlvr=c_dlvr, c_load=c_load, c_yard=c_yard,
            c_carrier=c_carrier, c_shift=c_shift,
            raw_totales=raw_totales
        )

        # -------- Modo 2: CON INSTANCIA
        if seg_inst:
            seg_filtradas = sorted(seg_flujos & seg_inst)
            procesar_modo(
                modo="CON_INSTANCIA",
                semana=semana,
                flujos=flujos,
                segregaciones=seg_filtradas,
                c_criterio=c_criterio, c_fm=c_fm, c_to=c_to,
                c_dlvr=c_dlvr, c_load=c_load, c_yard=c_yard,
                c_carrier=c_carrier, c_shift=c_shift,
                raw_totales=raw_totales
            )
        else:
            print(f"[WARN] Semana {semana}: no hay instancia. No se calcula CON_INSTANCIA.")

        # -------- Modo 3: CON FILTRO PARTICIPACIÓN (tipo generador)
        if MODO_FILTRO_PART:
            try:
                df_sb = pd.read_excel(flujos_path, sheet_name=FLOW_SHEET_SB, engine="openpyxl")
                seg_filtro = build_segs_por_participacion(
                    df_sb,
                    participacion_pct=PARTICIPACION_C,
                    min_load=MIN_LOAD, min_recv=MIN_RECV, min_dlvr=MIN_DLVR, min_dsch=MIN_DSCH
                )
                seg_filtro = sorted(set(seg_filtro) & seg_flujos)

                # Auditoría seg fuera del filtro
                fuera_filtro = sorted(seg_flujos - set(seg_filtro))
                if fuera_filtro:
                    df_fuera_f = (flujos[flujos[c_criterio].isin(fuera_filtro)]
                                  .groupby(c_criterio)[[c_dlvr, c_load, c_yard]]
                                  .sum()
                                  .reset_index()
                                  .rename(columns={c_criterio: "Segregacion", c_dlvr: "DLVR", c_load: "LOAD", c_yard: "YARD"}))
                    df_fuera_f.insert(0, "Semana", semana)
                    df_fuera_f.insert(1, "Participacion", PARTICIPACION_C)
                    seg_fuera_filtro_rows.extend(df_fuera_f.to_dict(orient="records"))

                procesar_modo(
                    modo=f"CON_FILTRO_P{PARTICIPACION_C}",
                    semana=semana,
                    flujos=flujos,
                    segregaciones=seg_filtro,
                    c_criterio=c_criterio, c_fm=c_fm, c_to=c_to,
                    c_dlvr=c_dlvr, c_load=c_load, c_yard=c_yard,
                    c_carrier=c_carrier, c_shift=c_shift,
                    raw_totales=raw_totales
                )
            except Exception as e:
                print(f"[WARN] Semana {semana}: no pude calcular CON_FILTRO_P{PARTICIPACION_C} -> {e}")

        print(f"[OK] Semana {semana} procesada | Flujos usado: {flujos_path}")

    # DataFrames finales
    df_resultados = pd.DataFrame(resultados)
    df_resumen = pd.DataFrame(resumen_semanal)
    df_detalle = pd.DataFrame(detalle_mov)
    df_missing = pd.DataFrame(missing_dist_rows).drop_duplicates() if missing_dist_rows else pd.DataFrame()

    df_resumen_modelo_68 = pd.DataFrame(resumen_modelo_68)
    if not df_resumen_modelo_68.empty:
        cols_pref = DIST_MODELO_COLS + ["Archivo_origen"]
        cols_final = [c for c in cols_pref if c in df_resumen_modelo_68.columns] + \
                     [c for c in df_resumen_modelo_68.columns if c not in cols_pref]
        df_resumen_modelo_68 = df_resumen_modelo_68[cols_final].sort_values(by=["Semana"], kind="stable")

    df_excluidos = pd.DataFrame(excluidos_rows) if excluidos_rows else pd.DataFrame()
    if not df_excluidos.empty:
        order = ["Modo","Semana","Segregacion","Tipo","Motivo","Movimientos","ime_fm_norm","ime_to_norm","ime_fm","ime_to","__rowid__","carrier","shift","Distancia_unit_m"]
        cols = [c for c in order if c in df_excluidos.columns] + [c for c in df_excluidos.columns if c not in order]
        df_excluidos = df_excluidos[cols]

    resumen_excl = []
    for (modo, sem, tipo, motivo), mv in excluidos_resumen.items():
        resumen_excl.append({"Modo": modo, "Semana": sem, "Tipo": tipo, "Motivo": motivo, "Movimientos_excluidos": int(mv)})
    df_resumen_excl = pd.DataFrame(resumen_excl).sort_values(["Modo","Semana","Tipo","Motivo"]) if resumen_excl else pd.DataFrame()

    df_seg_fuera_inst = pd.DataFrame(seg_fuera_instancia_rows) if seg_fuera_instancia_rows else pd.DataFrame()
    df_seg_fuera_filtro = pd.DataFrame(seg_fuera_filtro_rows) if seg_fuera_filtro_rows else pd.DataFrame()
    df_recon = pd.DataFrame(reconciliacion_rows) if reconciliacion_rows else pd.DataFrame()

    # Guardar Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        if not df_resumen.empty:
            _write_sheet_no_grouping(writer, df_resumen, "Resumen Semanal")
        if not df_resumen_modelo_68.empty:
            _write_sheet_no_grouping(writer, df_resumen_modelo_68, "Resumen Modelo 68")
        # if not df_resultados.empty:
        #     _write_sheet_no_grouping(writer, df_resultados, "Resultados por Segregación")
        # if not df_detalle.empty:
        #     _write_sheet_no_grouping(writer, df_detalle, "Detalle de Movimientos")
        # if not df_missing.empty:
        #     _write_sheet_no_grouping(writer, df_missing, "Pares sin distancia")
        # if not df_excluidos.empty:
        #     _write_sheet_no_grouping(writer, df_excluidos, "Excluidos Distancias")
        # if not df_resumen_excl.empty:
        #     _write_sheet_no_grouping(writer, df_resumen_excl, "Resumen Excluidos")
        # if not df_seg_fuera_inst.empty:
        #     _write_sheet_no_grouping(writer, df_seg_fuera_inst, "Seg_Fuera_Instancia")
        # if not df_seg_fuera_filtro.empty:
        #     _write_sheet_no_grouping(writer, df_seg_fuera_filtro, "Seg_Fuera_Filtro")
        # if not df_recon.empty:
        #     _write_sheet_no_grouping(writer, df_recon, "Reconciliacion")

    print(f"\nListo. Guardado en: {OUT_XLSX}")


if __name__ == "__main__":
    main()