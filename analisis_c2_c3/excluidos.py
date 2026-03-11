from pathlib import Path
import pandas as pd

# ===== CONFIG =====
BASE_DIR = Path("./analisis_c2_c3")
FILE_C2 = BASE_DIR / "analisis_flujos_w2022-01-03_0_c2.xlsx"
FILE_C3 = BASE_DIR / "analisis_flujos_w2022-01-03_0_c3.xlsx"
SHEET = "FlujosAll_sbt"

BLOQUES = {"C1","C2","C3","C4","C5","C6","C7","C8","C9","H1","H2","H3","H4","H5","T1","T2","T3","T4","I1","I2"}
SITIOS_MUELLE = {"Y-SAI-1", "Y-SAI-2"}
GATE = "GATE"

COUNT_YARD_ONLY_IF_HAS_LOAD_DLVR = True

MOVE_COLS = ["DLVR","LOAD","YARD"]  # las que usa tu script de distancias


def norm_node(x) -> str:
    return str(x).strip().upper()


def load_flujos(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    # Normaliza nodos
    df["ime_fm"] = df["ime_fm"].apply(norm_node)
    df["ime_to"] = df["ime_to"].apply(norm_node)
    df["criterio"] = df["criterio"].astype(str).str.strip()
    # Movimientos a int
    for c in MOVE_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def compute_counts(df: pd.DataFrame) -> dict:
    out = {}

    # Totales raw (tal como vienen en la hoja)
    out["raw_DLVR"] = int(df["DLVR"].sum())
    out["raw_LOAD"] = int(df["LOAD"].sum())
    out["raw_YARD"] = int(df["YARD"].sum())

    # Regla: “tiene load o dlvr” por segregación
    has_ld = df.groupby("criterio")[["LOAD","DLVR"]].sum()
    has_ld = ((has_ld["LOAD"] > 0) | (has_ld["DLVR"] > 0)).to_dict()

    # Contadores de lo que efectivamente cuenta tu script
    dlvr_ok = (df["ime_fm"].isin(BLOQUES)) & (df["ime_to"] == GATE) & (df["DLVR"] > 0)
    load_ok = (df["ime_fm"].isin(BLOQUES)) & (df["ime_to"].isin(SITIOS_MUELLE)) & (df["LOAD"] > 0)

    if COUNT_YARD_ONLY_IF_HAS_LOAD_DLVR:
        yard_ok = (df["ime_fm"].isin(BLOQUES)) & (df["ime_to"].isin(BLOQUES)) & (df["YARD"] > 0) & (df["criterio"].map(has_ld).fillna(False))
    else:
        yard_ok = (df["ime_fm"].isin(BLOQUES)) & (df["ime_to"].isin(BLOQUES)) & (df["YARD"] > 0)

    out["counted_DLVR"] = int(df.loc[dlvr_ok, "DLVR"].sum())
    out["counted_LOAD"] = int(df.loc[load_ok, "LOAD"].sum())
    out["counted_YARD"] = int(df.loc[yard_ok, "YARD"].sum())

    # Razones de exclusión (para entender el gap)
    # DLVR excluido
    out["DLVR_excl_fm_not_bloque"] = int(df.loc[(~df["ime_fm"].isin(BLOQUES)) & (df["DLVR"] > 0), "DLVR"].sum())
    out["DLVR_excl_to_not_gate"]   = int(df.loc[(df["ime_fm"].isin(BLOQUES)) & (df["ime_to"] != GATE) & (df["DLVR"] > 0), "DLVR"].sum())

    # LOAD excluido
    out["LOAD_excl_fm_not_bloque"]    = int(df.loc[(~df["ime_fm"].isin(BLOQUES)) & (df["LOAD"] > 0), "LOAD"].sum())
    out["LOAD_excl_to_not_muelle"]    = int(df.loc[(df["ime_fm"].isin(BLOQUES)) & (~df["ime_to"].isin(SITIOS_MUELLE)) & (df["LOAD"] > 0), "LOAD"].sum())

    # YARD excluido
    out["YARD_excl_non_bloque_pair"]  = int(df.loc[(~df["ime_fm"].isin(BLOQUES) | ~df["ime_to"].isin(BLOQUES)) & (df["YARD"] > 0), "YARD"].sum())
    if COUNT_YARD_ONLY_IF_HAS_LOAD_DLVR:
        out["YARD_excl_no_load_dlvr_seg"] = int(df.loc[(df["ime_fm"].isin(BLOQUES)) & (df["ime_to"].isin(BLOQUES)) & (df["YARD"] > 0) & (~df["criterio"].map(has_ld).fillna(False)), "YARD"].sum())
    else:
        out["YARD_excl_no_load_dlvr_seg"] = 0

    return out


def top_pair_diff(df2: pd.DataFrame, df3: pd.DataFrame, col: str, topn=15) -> pd.DataFrame:
    g2 = df2.groupby(["ime_fm","ime_to"], as_index=False)[col].sum().rename(columns={col:"c2"})
    g3 = df3.groupby(["ime_fm","ime_to"], as_index=False)[col].sum().rename(columns={col:"c3"})
    m = g2.merge(g3, on=["ime_fm","ime_to"], how="outer").fillna(0)
    m["diff"] = m["c3"] - m["c2"]
    m = m[m["diff"] != 0].copy()
    return m.sort_values("diff", key=lambda s: s.abs(), ascending=False).head(topn)


def main():
    df2 = load_flujos(FILE_C2)
    df3 = load_flujos(FILE_C3)

    c2 = compute_counts(df2)
    c3 = compute_counts(df3)

    print("\n=== RESUMEN (raw vs counted) ===")
    out = pd.DataFrame([c2, c3], index=["C2","C3"]).T
    print(out)

    print("\n=== TOP diferencias por par (ime_fm, ime_to) en DLVR ===")
    print(top_pair_diff(df2, df3, "DLVR").to_string(index=False))

    print("\n=== TOP diferencias por par (ime_fm, ime_to) en LOAD ===")
    print(top_pair_diff(df2, df3, "LOAD").to_string(index=False))

    print("\n=== TOP diferencias por par (ime_fm, ime_to) en YARD ===")
    print(top_pair_diff(df2, df3, "YARD").to_string(index=False))


if __name__ == "__main__":
    main()