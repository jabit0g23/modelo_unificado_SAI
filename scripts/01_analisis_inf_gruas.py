import os
import glob
from pathlib import Path

import pandas as pd

# =========================================================
# PARÁMETROS DE ENTRADA — modificar según necesidad
# =========================================================
EXPERIMENTOS = [
    "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20",
]

SEMANAS = [
    "2022-01-03",
    "2022-01-10",
    "2022-01-17",
    "2022-01-24",
    "2022-01-31",
    "2022-02-07",
    "2022-02-14",
    "2022-02-21",
    "2022-02-28",
    "2022-03-07",
    "2022-03-14",
    "2022-03-21",
    "2022-04-04",
    "2022-04-11",
    "2022-04-18",
    "2022-04-25",
    "2022-05-02",
    "2022-05-09",
    "2022-05-16",
    "2022-05-23",
    "2022-05-30",
    "2022-06-06",
    "2022-06-13",
    "2022-06-20",
    "2022-06-27",
    "2022-07-04",
]

# =========================================================
# RUTAS
# =========================================================
SCRIPT_DIR = Path(__file__).parent
ROOT_RESULTADOS = SCRIPT_DIR.parent / "resultados"
OUT_XLSX = SCRIPT_DIR / "analisis_inf_gruas.xlsx"


# =========================================================
# LÓGICA PRINCIPAL
# =========================================================
def contar_xlsx(experimento: str, semana: str) -> int:
    carpeta = ROOT_RESULTADOS / experimento / "resultados_gruas" / f"resultados_turno_{semana}"
    if not carpeta.exists():
        return 0
    return len(glob.glob(str(carpeta / "*.xlsx")))


def main():
    data = {}
    for exp in EXPERIMENTOS:
        data[exp] = {}
        for semana in SEMANAS:
            data[exp][semana] = contar_xlsx(exp, semana)

    df = pd.DataFrame(data, index=SEMANAS).T
    df.index.name = "Experimento"

    df["Total"] = df[SEMANAS].sum(axis=1)

    totals = df.sum(axis=0).rename("TOTAL")
    df = pd.concat([df, totals.to_frame().T])

    df.to_excel(OUT_XLSX)
    print(f"Guardado en: {OUT_XLSX}")
    print(df.to_string())


if __name__ == "__main__":
    main()
