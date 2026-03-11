from pathlib import Path
import pandas as pd

BASE_DIR = Path("./analisis_c2_c3")
FILE_C2 = BASE_DIR / "analisis_flujos_w2022-01-03_0_c2.xlsx"
FILE_C3 = BASE_DIR / "analisis_flujos_w2022-01-03_0_c3.xlsx"

SHEET = "FlujosAll_sbt"
MOVE_COLS = ["DLVR", "DSCH", "LOAD", "OTHR", "RECV", "SHFT", "YARD"]


def read_and_sum(path: Path, sheet: str, cols: list[str]) -> pd.Series:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    tmp = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return tmp.sum()

def main():
    sum_c2 = read_and_sum(FILE_C2, SHEET, MOVE_COLS)
    sum_c3 = read_and_sum(FILE_C3, SHEET, MOVE_COLS)

    # Tabla bonita
    out = pd.DataFrame({
        "c2": sum_c2,
        "c3": sum_c3,
        "diff": (sum_c3 - sum_c2),
    })

    print("\n=== SUMAS FlujosAll_sbt ===")
    print(out)

if __name__ == "__main__":
    main()