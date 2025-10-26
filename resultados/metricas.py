# metricas.py
# Recolector de métricas Camila/Magdalena por carpeta principal (vive en .../resultados).
# - Produce un EXCEL ordenado con hojas separadas: Camila_Resumen y Magdalena_Promedios.
# - Produce un CSV aparte SOLO con las semanas que tienen .iis.ilp (plano por fila).
# - Corrige parseo numérico (respeta notación científica, miles, etc.).

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import argparse
import sys
import os
import re
import csv
import pandas as pd

IGNORAR_NOMBRES = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode"}

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def es_carpeta_valida(p: Path) -> bool:
    return p.is_dir() and (p.name not in IGNORAR_NOMBRES) and (not p.name.startswith("."))

_sci_pat = re.compile(r"^[\+\-]?\d+(?:\.\d+)?[eE][\+\-]?\d+$")
_thousands_group_pat = re.compile(r"^\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?$")

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Convierte strings con miles/decimales y notación científica a números.
    Reglas:
      - Si es notación científica -> to_numeric directo.
      - Si tiene grupos de miles (1.234.567 o 1,234,567) -> elimina miles y usa '.' como decimal.
      - Caso simple: quita NBSP/espacios; si hay una sola ',' y parece decimal -> la cambio a '.'
    """
    s = series.astype(str).str.strip().str.replace("\u00A0", "", regex=False)  # NBSP fuera

    def _fix_one(x: str) -> str:
        x = x.strip()
        if not x:
            return x
        # notación científica preservada
        if _sci_pat.match(x) or "e" in x.lower():
            return x
        # patrones de miles tipo 1.234.567,89 o 1,234,567.89
        if _thousands_group_pat.match(x):
            last_comma = x.rfind(",")
            last_dot = x.rfind(".")
            dec_idx = max(last_comma, last_dot)
            if dec_idx != -1 and (len(x) - dec_idx - 1) not in (0,3,6,9,12):  # no múltiplo de 3 ⇒ decimal
                int_part = x[:dec_idx].replace(".", "").replace(",", "")
                dec_part = x[dec_idx+1:].replace(".", "").replace(",", "")
                return int_part + "." + dec_part
            else:
                return x.replace(".", "").replace(",", "")
        # sólo comas
        if x.count(",") >= 1 and x.count(".") == 0:
            parts = x.split(",")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                return "".join(parts)  # miles
            if x.count(",") == 1 and len(parts[1]) in (1,2):
                return x.replace(",", ".")  # decimal
            return x.replace(",", "")
        # sólo puntos
        if x.count(".") >= 1 and x.count(",") == 0:
            parts = x.split(".")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                return "".join(parts)
            return x
        # mixto raro
        x2 = re.sub(r"[^0-9\.\+\-eE]", "", x)
        return x2

    cleaned = s.map(_fix_one)
    return pd.to_numeric(cleaned, errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# Descubrimiento de principales (este nivel)
# ─────────────────────────────────────────────────────────────────────────────

def encontrar_carpetas_principales_en_raiz(raiz: Path, debug: bool = False) -> List[Path]:
    principales: List[Path] = []
    for sub in sorted([p for p in raiz.iterdir() if es_carpeta_valida(p)]):
        if (sub / "resultados_camila").is_dir() and (sub / "resultados_magdalena").is_dir():
            principales.append(sub)
            if debug:
                print(f"[DEBUG] Principal: {sub.name}")
    return principales

# ─────────────────────────────────────────────────────────────────────────────
# CAMILA
# ─────────────────────────────────────────────────────────────────────────────

def contar_archivos_camila(dir_camila: Path) -> Tuple[int, int, int]:
    count_xlsx = 0
    count_iis_ilp = 0
    if not dir_camila.exists():
        return 0, 0, 0
    for root, _, files in os.walk(dir_camila):
        for f in files:
            fl = f.lower()
            if fl.endswith(".xlsx"):
                count_xlsx += 1
            elif fl.endswith(".iis.ilp"):
                count_iis_ilp += 1
    total = count_xlsx + count_iis_ilp
    return total, count_xlsx, count_iis_ilp

def semanas_con_iis_ilp(dir_camila: Path) -> List[str]:
    """
    Retorna nombres de subcarpetas (semanas) dentro de resultados_camila
    que contienen al menos un archivo .iis.ilp, pero SIN el prefijo 'resultados_turno_'.
    """
    if not dir_camila.exists():
        return []
    semanas = []
    for sub in sorted([d for d in dir_camila.iterdir() if d.is_dir()]):
        found = False
        for root, _, files in os.walk(sub):
            if any(f.lower().endswith(".iis.ilp") for f in files):
                found = True
                break
        if found:
            nombre = sub.name
            if nombre.startswith("resultados_turno_"):
                nombre = nombre[len("resultados_turno_"):]
            semanas.append(nombre)
    return semanas

# ─────────────────────────────────────────────────────────────────────────────
# MAGDALENA
# ─────────────────────────────────────────────────────────────────────────────

def _elegir_distancias_file(week_dir: Path) -> Optional[Path]:
    candidatos = sorted(week_dir.glob("Distancias_Modelo_*.xlsx"))
    if not candidatos:
        return None
    con_68 = [p for p in candidatos if "_68" in p.stem]
    return con_68[0] if con_68 else candidatos[0]

def _leer_resumen_semanal(xlsx_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Resumen Semanal", engine="openpyxl")
    except Exception:
        try:
            xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
            hoja = None
            for name in xls.sheet_names:
                ln = name.lower().strip()
                if "resumen" in ln and ("seman" in ln or "semana" in ln):
                    hoja = name
                    break
            if hoja is None:
                return None
            df = pd.read_excel(xlsx_path, sheet_name=hoja, engine="openpyxl")
        except Exception:
            return None

    df.columns = [str(c).strip() for c in df.columns]
    mapping = {}
    for col in df.columns:
        key = col.lower().replace("\n", " ").strip()
        if key.startswith("semana"):
            mapping[col] = "Semana"
        elif "distancia total" in key:
            mapping[col] = "Distancia Total"
        elif "distancia load" in key:
            mapping[col] = "Distancia LOAD"
        elif "distancia dlvr" in key or "distancia deliver" in key:
            mapping[col] = "Distancia DLVR"
        elif "movimientos_dlvr" in key or ("movimientos" in key and "dlvr" in key):
            mapping[col] = "Movimientos_DLVR"
        elif "movimientos_load" in key or ("movimientos" in key and "load" in key):
            mapping[col] = "Movimientos_LOAD"

    df = df.rename(columns=mapping)

    needed = ["Semana", "Distancia Total", "Distancia LOAD", "Distancia DLVR",
              "Movimientos_DLVR", "Movimientos_LOAD"]
    cols = [c for c in needed if c in df.columns]
    if "Semana" not in cols:
        return None

    tmp = df[cols].copy()
    tmp["Semana"] = pd.to_datetime(tmp["Semana"], errors="coerce", utc=False)
    tmp = tmp.dropna(subset=["Semana"])
    if tmp.empty:
        return None

    for mcol in ["Distancia Total", "Distancia LOAD", "Distancia DLVR",
                 "Movimientos_DLVR", "Movimientos_LOAD"]:
        if mcol in tmp.columns:
            tmp[mcol] = _coerce_numeric(tmp[mcol])

    return tmp

def extraer_magdalena_raw(dir_magdalena: Path, debug: bool = False) -> Optional[pd.DataFrame]:
    if not dir_magdalena.exists():
        return None
    registros: List[pd.DataFrame] = []
    semanas = sorted([d for d in dir_magdalena.iterdir() if d.is_dir()])
    if debug:
        print(f"[DEBUG][MAGDALENA] {dir_magdalena.name}: {len(semanas)} carpetas-semana")
    for semana_dir in semanas:
        xlsx_path = _elegir_distancias_file(semana_dir)
        if xlsx_path is None:
            if debug:
                print(f"[DEBUG][MAGDALENA] Omitida (sin Distancias): {semana_dir.name}")
            continue
        tmp = _leer_resumen_semanal(xlsx_path)
        if tmp is None or tmp.empty:
            if debug:
                print(f"[DEBUG][MAGDALENA] No legible/columnas faltantes: {xlsx_path.name}")
            continue
        registros.append(tmp)
    if not registros:
        return None
    data = pd.concat(registros, ignore_index=True)
    data["anio"] = data["Semana"].dt.year
    return data

def magdalena_promedios(dir_magdalena: Path, debug: bool = False) -> Optional[pd.DataFrame]:
    raw = extraer_magdalena_raw(dir_magdalena, debug=debug)
    if raw is None or raw.empty:
        return None
    metrics = [c for c in ["Distancia Total", "Distancia LOAD", "Distancia DLVR",
                           "Movimientos_DLVR", "Movimientos_LOAD"]
               if c in raw.columns and pd.api.types.is_numeric_dtype(raw[c])]
    prom = raw.groupby("anio", as_index=False)[metrics].mean()
    prom["anio"] = pd.to_numeric(prom["anio"], errors="coerce").astype("Int64")
    return prom

# ─────────────────────────────────────────────────────────────────────────────
# Helpers Excel
# ─────────────────────────────────────────────────────────────────────────────

def _autosize_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame):
    try:
        ws = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns, start=1):
            # largo máximo entre header y valores
            max_len = max(
                len(str(col)),
                *(len(str(v)) for v in df[col].astype(str).values)
            )
            # margen
            ws.column_dimensions[chr(64 + idx)].width = min(max_len + 2, 60)
        # congelar header
        ws.freeze_panes = "A2"
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Main / CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recolector de resultados Camila/Magdalena por carpeta principal (vive en 'resultados').")
    parser.add_argument("--root", type=str, default=None, help="Directorio raíz (por defecto: carpeta del script).")
    parser.add_argument("--out_excel", type=str, default="reporte_metricas.xlsx", help="Ruta de salida del Excel consolidado.")
    parser.add_argument("--out_weeks_csv", type=str, default="camila_semanas_con_iis_ilp.csv", help="Ruta del CSV con semanas que tienen .iis.ilp (formato plano).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    raiz = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    if not raiz.exists():
        print(f"ERROR: la ruta raíz no existe -> {raiz}")
        sys.exit(1)

    principales = encontrar_carpetas_principales_en_raiz(raiz, debug=args.debug)
    if args.debug:
        print(f"[DEBUG] Raíz: {raiz}")
        for p in principales:
            print(f"[DEBUG] Principal detectada: {p.name}")

    filas_camila: List[Dict] = []
    filas_mag_prom: List[Dict] = []
    filas_weeks_planas: List[Dict] = []  # ← para el CSV “solo listas”

    for principal in principales:
        nombre = principal.name
        dir_camila = principal / "resultados_camila"
        dir_mag = principal / "resultados_magdalena"

        total, c_xlsx, c_iis = contar_archivos_camila(dir_camila)
        prop_xlsx = (c_xlsx / total) if total > 0 else 0.0
        prop_iis = (c_iis / total) if total > 0 else 0.0

        # lista de semanas con al menos un .iis.ilp
        semanas_iis = semanas_con_iis_ilp(dir_camila)

        # (1) Resumen por principal
        filas_camila.append({
            "carpeta_principal": nombre,
            "camila_total_archivos": total,
            "camila_count_xlsx": c_xlsx,
            "camila_count_iis_ilp": c_iis,
            "camila_prop_xlsx": round(prop_xlsx, 6),
            "camila_prop_iis_ilp": round(prop_iis, 6),
        })

        # (2) Plano por semana (para CSV aparte)
        for sem in semanas_iis:
            filas_weeks_planas.append({
                "carpeta_principal": nombre,
                "semana": sem
            })

        if args.debug:
            print(f"[DEBUG][CAMILA] {nombre}: total={total} xlsx={c_xlsx} iis.ilp={c_iis} semanas_iis={len(semanas_iis)}")

        # MAGDALENA promedios
        prom = magdalena_promedios(dir_mag, debug=args.debug)
        if prom is not None and not prom.empty:
            for _, row in prom.iterrows():
                out = {"carpeta_principal": nombre, "anio": int(row["anio"])}
                for k in ["Distancia Total", "Distancia LOAD", "Distancia DLVR",
                          "Movimientos_DLVR", "Movimientos_LOAD"]:
                    if k in prom.columns and pd.notna(row.get(k)):
                        out[f"prom_{k}"] = float(row[k])
                filas_mag_prom.append(out)

    # ── DataFrames ordenados
    df_camila = pd.DataFrame(filas_camila)
    if not df_camila.empty:
        col_order_camila = [
            "carpeta_principal",
            "camila_total_archivos",
            "camila_count_xlsx",
            "camila_count_iis_ilp",
            "camila_prop_xlsx",
            "camila_prop_iis_ilp",
        ]
        df_camila = df_camila[col_order_camila].sort_values("carpeta_principal").reset_index(drop=True)

    df_mag = pd.DataFrame(filas_mag_prom)
    if not df_mag.empty:
        if "anio" in df_mag.columns:
            df_mag["anio"] = pd.to_numeric(df_mag["anio"], errors="coerce").astype("Int64")
        # ordenar columnas: carpeta, anio, luego métricas
        metric_cols = [c for c in df_mag.columns if c.startswith("prom_")]
        df_mag = df_mag[["carpeta_principal", "anio", *metric_cols]].sort_values(["carpeta_principal","anio"]).reset_index(drop=True)

    df_weeks = pd.DataFrame(filas_weeks_planas)
    if not df_weeks.empty:
        df_weeks = df_weeks.sort_values(["carpeta_principal","semana"]).reset_index(drop=True)

    # ── Escrituras
    # 1) Excel consolidado con hojas separadas
    with pd.ExcelWriter(args.out_excel, engine="openpyxl") as wr:
        if not df_camila.empty:
            df_camila.to_excel(wr, sheet_name="Camila_Resumen", index=False)
            _autosize_columns(wr, "Camila_Resumen", df_camila)
        else:
            pd.DataFrame(columns=[
                "carpeta_principal","camila_total_archivos","camila_count_xlsx",
                "camila_count_iis_ilp","camila_prop_xlsx","camila_prop_iis_ilp"
            ]).to_excel(wr, sheet_name="Camila_Resumen", index=False)

        if not df_mag.empty:
            df_mag.to_excel(wr, sheet_name="Magdalena_Promedios", index=False)
            _autosize_columns(wr, "Magdalena_Promedios", df_mag)
        else:
            pd.DataFrame(columns=["carpeta_principal","anio"]).to_excel(wr, sheet_name="Magdalena_Promedios", index=False)

    # 2) CSV plano SOLO con semanas con .iis.ilp
    if df_weeks.empty:
        # crear CSV con headers igual
        with open(args.out_weeks_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["carpeta_principal","semana"])
    else:
        df_weeks.to_csv(args.out_weeks_csv, index=False, encoding="utf-8-sig")

    # ── Consola
    print("\n=== CAMILA (Resumen por carpeta principal) ===")
    print(df_camila.to_string(index=False) if not df_camila.empty else "(sin filas)")

    print("\n=== MAGDALENA (promedios por año) ===")
    print(df_mag.to_string(index=False) if not df_mag.empty else "(sin filas)")

    print("\n=== CAMILA – semanas con .iis.ilp (plano) ===")
    print(df_weeks.to_string(index=False) if not df_weeks.empty else "(sin filas)")

    print(f"\nArchivos generados:\n - {args.out_excel}\n - {args.out_weeks_csv}")

if __name__ == "__main__":
    main()
