import re
from pathlib import Path

import pandas as pd

# =========================================================
# PARAMETROS DE ENTRADA
# =========================================================
EXPERIMENTOS = [
    "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20",
]

# Topologia patio
BLOQUES = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "H1", "H2", "H3", "H4", "H5",
    "T1", "T2", "T3", "T4",
    "I1", "I2",
]
SITIOS_MUELLE = ["Y-SAI-1", "Y-SAI-2"]
GATE = "GATE"

DEFAULT_DIST_IF_MISSING = 0.0

# =========================================================
# RUTAS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_REPO = SCRIPT_DIR.parent
ROOT_RESULTADOS = ROOT_REPO / "resultados"
DISTANCIAS_XLSX = ROOT_REPO / "datos" / "Distancias_GranPatio.xlsx"
OUT_XLSX = SCRIPT_DIR / "analisis_resultados_dist.xlsx"

FLOW_SHEET = "FlujosAll_sbt"
DIST_SHEET = "Distancias"
DIST_COL_FROM = "ime_fm"
DIST_COL_TO = "ime_to"
DIST_COL_VAL = "Distancia[m]"

ANALISIS_FLUJOS_FMT = "analisis_flujos_w{semana}_0.xlsx"
DIST_MODELO_FMT = "Distancias_Modelo_{semana}.xlsx"
RESUMEN_SEMANAL_SHEET = "Resumen Semanal"
MODEL_COLS = [
    "Semana",
    "Distancia Total",
    "Distancia LOAD",
    "Distancia DLVR",
    "Movimientos_DLVR",
    "Movimientos_LOAD",
]

PATRON_SEMANA = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# =========================================================
# HELPERS
# =========================================================
def norm_node(x) -> str:
    return str(x).strip().upper()


def load_dist_map(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name=DIST_SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df[DIST_COL_FROM] = df[DIST_COL_FROM].astype(str).str.strip().str.upper()
    df[DIST_COL_TO] = df[DIST_COL_TO].astype(str).str.strip().str.upper()
    df[DIST_COL_VAL] = pd.to_numeric(df[DIST_COL_VAL], errors="coerce").fillna(0.0)
    return {(r[DIST_COL_FROM], r[DIST_COL_TO]): float(r[DIST_COL_VAL]) for _, r in df.iterrows()}


def get_dist(dist_map: dict, fm: str, to: str) -> float:
    fm = norm_node(fm)
    to = norm_node(to)
    if (fm, to) in dist_map:
        return dist_map[(fm, to)]
    if (to, fm) in dist_map:
        return dist_map[(to, fm)]
    return DEFAULT_DIST_IF_MISSING


def listar_semanas(carpeta: Path) -> list[str]:
    if not carpeta.exists():
        return []
    return sorted(p.name for p in carpeta.iterdir() if p.is_dir() and PATRON_SEMANA.match(p.name))


def calcular_distancias_reales(flujos_path: Path, dist_map: dict) -> dict:
    """
    Calcula movimientos y distancia recorrida por LOAD, DLVR y YARD usando FlujosAll_sbt.
    Filtros geograficos identicos a los del modelo:
      - LOAD: bloque -> muelle
      - DLVR: bloque -> GATE
      - YARD: bloque -> bloque
    """
    df = pd.read_excel(flujos_path, sheet_name=FLOW_SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    df["ime_fm"] = df["ime_fm"].astype(str).str.strip().str.upper()
    df["ime_to"] = df["ime_to"].astype(str).str.strip().str.upper()
    for c in ("LOAD", "DLVR", "YARD"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    bloques_set = {b.upper() for b in BLOQUES}
    muelle_set = {m.upper() for m in SITIOS_MUELLE}
    gate_up = GATE.upper()

    fm = df["ime_fm"]
    to = df["ime_to"]

    mask_load = fm.isin(bloques_set) & to.isin(muelle_set)
    mask_dlvr = fm.isin(bloques_set) & (to == gate_up)
    mask_yard = fm.isin(bloques_set) & to.isin(bloques_set)

    distancias = df.apply(lambda r: get_dist(dist_map, r["ime_fm"], r["ime_to"]), axis=1)

    mov_load = int(df.loc[mask_load, "LOAD"].sum())
    mov_dlvr = int(df.loc[mask_dlvr, "DLVR"].sum())
    mov_yard = int(df.loc[mask_yard, "YARD"].sum())

    dist_load = float((df.loc[mask_load, "LOAD"] * distancias.loc[mask_load]).sum())
    dist_dlvr = float((df.loc[mask_dlvr, "DLVR"] * distancias.loc[mask_dlvr]).sum())
    dist_yard = float((df.loc[mask_yard, "YARD"] * distancias.loc[mask_yard]).sum())

    return {
        "Movimientos_LOAD": mov_load,
        "Movimientos_DLVR": mov_dlvr,
        "Movimientos_YARD": mov_yard,
        "Distancia_LOAD": dist_load,
        "Distancia_DLVR": dist_dlvr,
        "Distancia_YARD": dist_yard,
        "Distancia_Total": dist_load + dist_dlvr + dist_yard,
    }


def leer_distancias_modelo(path: Path, semana: str) -> dict | None:
    df = pd.read_excel(path, sheet_name=RESUMEN_SEMANAL_SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        return None

    fila = df.iloc[0].to_dict()
    out = {"Semana": semana}
    for c in MODEL_COLS[1:]:
        v = fila.get(c)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
            try:
                v = float(v)
            except ValueError:
                v = None
        if pd.isna(v) if v is not None else True:
            out[c] = None
        else:
            out[c] = int(round(float(v)))
    return out


# =========================================================
# MAIN
# =========================================================
def main():
    if not DISTANCIAS_XLSX.exists():
        raise FileNotFoundError(f"No existe {DISTANCIAS_XLSX}")

    dist_map = load_dist_map(DISTANCIAS_XLSX)

    filas_reales = []
    filas_modelo = []

    for exp in EXPERIMENTOS:
        exp_dir = ROOT_RESULTADOS / exp
        instancias_dir = exp_dir / "instancias_coloracion"
        resultados_dir = exp_dir / "resultados_coloracion"

        if not instancias_dir.exists():
            print(f"[SKIP] No existe {instancias_dir}")
            continue

        semanas = listar_semanas(instancias_dir)
        if not semanas:
            print(f"[SKIP] {exp}: sin semanas")
            continue

        print(f"\nProcesando {exp} ({len(semanas)} semanas)")

        for semana in semanas:
            flujos_path = instancias_dir / semana / ANALISIS_FLUJOS_FMT.format(semana=semana)
            modelo_path = resultados_dir / semana / DIST_MODELO_FMT.format(semana=semana)

            if flujos_path.exists():
                try:
                    real = calcular_distancias_reales(flujos_path, dist_map)
                    filas_reales.append({"Experimento": exp, "Semana": semana, **real})
                except Exception as e:
                    print(f"  [ERROR] reales {exp}/{semana}: {e}")
            else:
                print(f"  [AVISO] no existe {flujos_path.name} en {semana}")

            if modelo_path.exists():
                try:
                    mod = leer_distancias_modelo(modelo_path, semana)
                    if mod:
                        filas_modelo.append({"Experimento": exp, **mod})
                except Exception as e:
                    print(f"  [ERROR] modelo {exp}/{semana}: {e}")
            else:
                print(f"  [AVISO] no existe {modelo_path.name} en {semana}")

            print(f"  [OK] {semana}")

    df_reales = pd.DataFrame(filas_reales)
    df_modelo = pd.DataFrame(filas_modelo)

    if not df_reales.empty:
        cols = ["Experimento", "Semana",
                "Distancia_Total", "Distancia_LOAD", "Distancia_DLVR", "Distancia_YARD",
                "Movimientos_LOAD", "Movimientos_DLVR", "Movimientos_YARD"]
        df_reales = df_reales[cols].sort_values(["Experimento", "Semana"])
        for c in cols[2:]:
            df_reales[c] = pd.to_numeric(df_reales[c], errors="coerce").round(0).astype("Int64")

    if not df_modelo.empty:
        cols = ["Experimento"] + MODEL_COLS
        df_modelo = df_modelo[cols].sort_values(["Experimento", "Semana"])
        for c in MODEL_COLS[1:]:
            df_modelo[c] = pd.to_numeric(df_modelo[c], errors="coerce").round(0).astype("Int64")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        if not df_reales.empty:
            df_reales.to_excel(writer, sheet_name="Distancias reales", index=False)
        if not df_modelo.empty:
            df_modelo.to_excel(writer, sheet_name="Distancias modelo", index=False)

        for sname in writer.sheets:
            ws = writer.sheets[sname]
            for row in ws.iter_rows(min_row=2, min_col=3):
                for cell in row:
                    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        cell.number_format = "0"

    print(f"\nGuardado en: {OUT_XLSX}")


if __name__ == "__main__":
    main()
