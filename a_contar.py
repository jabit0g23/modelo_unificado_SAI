from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
ROOT = Path(".")  # cambia si tu carpeta base es otra
PATRON_BASE = "resultados_generados_*"

# Camila
SUB_CAMILA = "resultados_camila"
TURNOS_CAMILA = [
    "resultados_turno_2022-02-07",  # Instancia 1
    "resultados_turno_2022-02-21",  # Instancia 2
    "resultados_turno_2022-02-28",  # Instancia 3
    "resultados_turno_2022-03-14",  # Instancia 4
]
TOTAL_OBJETIVO = 21  # 21 - #xlsx

# Magdalena (distancias)
SUB_MAGDA = "resultados_magdalena"
SEMANAS = ["2022-02-07", "2022-02-21", "2022-02-28", "2022-03-14"]
DIST_FILE_TMPL = "Distancias_Modelo_{semana}_68.xlsx"
DIST_SHEET = "Resumen Semanal"
DIST_COLNAME = "Distancia Total"

# Output
SALIDA_XLSX = "resumen_conteo_instancias_camila_con_distancia.xlsx"


# =========================
# HELPERS
# =========================
def contar_xlsx_en_carpeta(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return len([f for f in folder.glob("*.xlsx") if f.is_file() and not f.name.startswith("~$")])


def leer_distancia_total(dist_xlsx: Path) -> float | None:
    if not dist_xlsx.exists() or not dist_xlsx.is_file():
        return None

    try:
        df = pd.read_excel(dist_xlsx, sheet_name=DIST_SHEET, engine="openpyxl")
    except Exception:
        return None

    # Normaliza nombres de columnas por si vienen con espacios raros
    cols_norm = {c: str(c).strip() for c in df.columns}
    df.rename(columns=cols_norm, inplace=True)

    if DIST_COLNAME not in df.columns:
        return None

    # Tomamos el primer valor numérico no nulo de esa columna
    s = pd.to_numeric(df[DIST_COLNAME], errors="coerce").dropna()
    if len(s) == 0:
        return None

    return float(s.iloc[0])


# =========================
# MAIN
# =========================
filas = []
carpetas_base = sorted([p for p in ROOT.glob(PATRON_BASE) if p.is_dir()])

for base in carpetas_base:
    fila = {"Carpeta_nombre _ordenadas": base.name}

    # ---- Camila: Instancias = 21 - #xlsx en cada turno ----
    camila_dir = base / SUB_CAMILA
    for idx, turno in enumerate(TURNOS_CAMILA, start=1):
        n = contar_xlsx_en_carpeta(camila_dir / turno)
        fila[f"Instancia {idx}"] = TOTAL_OBJETIVO - n

    # ---- Magdalena: Distancia promedio de 4 semanas ----
    magda_dir = base / SUB_MAGDA
    dist_vals = []

    for semana in SEMANAS:
        semana_dir = magda_dir / semana  # acá el turno es solo la semana
        dist_xlsx = semana_dir / DIST_FILE_TMPL.format(semana=semana)

        val = leer_distancia_total(dist_xlsx)
        if val is not None:
            dist_vals.append(val)

    if dist_vals:
        promedio = sum(dist_vals) / len(dist_vals)
        fila["Distancia"] = int(round(promedio))  # número limpio, sin comas ni puntos
    else:
        fila["Distancia"] = None


    filas.append(fila)

df = pd.DataFrame(
    filas,
    columns=[
        "Carpeta_nombre _ordenadas",
        "Instancia 1", "Instancia 2", "Instancia 3", "Instancia 4",
        "Distancia",
    ],
)

df.to_excel(SALIDA_XLSX, index=False)
print("Guardado:", SALIDA_XLSX)
print(df)
