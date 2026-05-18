"""
Helpers de lectura y normalización de las hojas Excel de Magdalena.

El trabajo sucio: leer las hojas heterogéneas (S, Cargar, Entregar, Volumen…),
normalizar los códigos de segregación/bloque (mapear C#→b#, minúsculas, mapear
nombres largos de 'Segregación' al código 'S*') y poder mezclar hojas distintas
con las mismas claves.
"""

import pandas as pd


def build_compat_df(blocks, colname):
    """Matriz completa de compatibilidad 1 por defecto en formato largo (b1, b2, <colname>)."""
    rows = [{"b1": b1, "b2": b2, colname: 1} for b1 in blocks for b2 in blocks]
    return pd.DataFrame(rows)


def get_size_from_segregation(seg_string):
    """Extrae 20/40 del texto de Segregacion; None si no se puede parsear."""
    try:
        parts = str(seg_string).split('-')
        size = int(parts[2]) if len(parts) > 2 else None
        return size if size in (20, 40) else None
    except Exception:
        return None


def coerce_numeric(series):
    """Convierte una serie a numérico tolerando comas y strings; los no convertibles quedan NaN."""
    s = series
    if s.dtype == "object":
        s = s.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def build_normalizers(df_S):
    """
    Prepara estructuras para mapear entre códigos S* y texto de segregación.

    Retorna:
        s_low_set            : conjunto de códigos 's1, s2, …' válidos
        map_segtext_to_code  : texto 'Segregacion' (lower) → 's*'
        teu_factor_by_S_low  : 's*' → 1 (20') o 2 (40')
    """
    df_S = df_S.copy()
    df_S["S_low"] = df_S["S"].astype(str).str.lower()
    if "Size" not in df_S.columns or df_S["Size"].isnull().any():
        df_S["Size"] = df_S["Segregacion"].apply(get_size_from_segregation)

    s_low_set = set(df_S["S_low"])
    map_segtext_to_code = (
        df_S.assign(seg_low=df_S["Segregacion"].astype(str).str.lower())
            .dropna(subset=["seg_low"])
            .set_index("seg_low")["S_low"]
            .to_dict()
    )
    teu_factor_by_S_low = df_S.set_index("S_low")["Size"].map(lambda z: 2 if z == 40 else 1).to_dict()

    return s_low_set, map_segtext_to_code, teu_factor_by_S_low


def try_read_teu_sheet(file_instancia, teu_factor_by_S_low):
    """
    Si la instancia trae una hoja opcional con TEU por segregación (nombres
    'TEU', 'TEUs', 'TEUs_instancia'), la usa para sobrescribir los factores TEU.
    """
    xl = pd.ExcelFile(file_instancia)
    candidates = [s for s in xl.sheet_names if s.strip().lower() in {"teu", "teus", "teus_instancia"}]
    if not candidates:
        return teu_factor_by_S_low

    try:
        df_teu = pd.read_excel(file_instancia, sheet_name=candidates[0])
        cols = [c.lower() for c in df_teu.columns]
        if "teu" not in cols:
            return teu_factor_by_S_low
        if "s" in cols:
            s_col = df_teu.columns[cols.index("s")]
            df_teu["S_low"] = df_teu[s_col].astype(str).str.lower()
        elif "segregacion" in cols:
            sg_col = df_teu.columns[cols.index("segregacion")]
            df_teu["S_low"] = df_teu[sg_col].astype(str).str.lower()
        else:
            return teu_factor_by_S_low

        df_teu["TEU"] = coerce_numeric(df_teu["TEU"])
        df_teu = df_teu.dropna(subset=["S_low", "TEU"])
        for _, r in df_teu.iterrows():
            v = int(r["TEU"])
            if v >= 1:
                teu_factor_by_S_low[r["S_low"]] = v
        print("   ✓ Hoja TEU detectada y aplicada.")
    except Exception as e:
        print(f"   ! Advertencia: no se pudo aplicar hoja TEU opcional: {e}")
    return teu_factor_by_S_low


def try_read_os(file_instancia, default_val=1.0):
    """
    Lee la densidad OS de una hoja opcional 'OS'. Acepta columna 'OS' o el
    primer valor de la primera celda. Si falla, retorna default_val.
    """
    try:
        df_os = pd.read_excel(file_instancia, sheet_name="OS")
        cols_lower = [c.lower() for c in df_os.columns]
        val = df_os.iloc[0, cols_lower.index("os")] if "os" in cols_lower else df_os.iloc[0, 0]
        v = float(val)
        return v if v > 0 else default_val
    except Exception:
        return default_val


def normalize_S(series, s_low_set, map_segtext_to_code, label_for_log=""):
    """
    Convierte una serie de códigos ('S1') o nombres largos ('expo-dry-…') al
    código S_low ('s1'). Devuelve NaN para filas no mapeables y las loggea.
    """
    s_norm = series.astype(str).str.strip()
    s_low = s_norm.str.lower()

    mask_code = s_low.isin(s_low_set)
    mapped = s_low.mask(mask_code).map(map_segtext_to_code)
    out = s_low.where(mask_code, mapped)

    bad = out[out.isna()]
    if not bad.empty:
        muestras = s_norm.loc[bad.index].unique()[:5]
        print(f"   ! Advertencia: {bad.size} filas con 'Segregación' no mapeada en {label_for_log}. Ejemplos:", list(muestras))
    return out


def normalize_B(series):
    """Convierte C# o B# a 'b#' (lower); H/T/I quedan en minúsculas."""
    x = series.astype(str).str.strip()
    x = x.str.replace("^C", "b", regex=True)
    return x.str.lower()


def standardize_sheet(
    df, s_low_set, map_segtext_to_code,
    label_for_log="",
    seg_col_guess=("Segregación", "Segregacion", "S"),
    block_col_guess=("Bloque", "B"),
):
    """Añade columnas 'S_low', 'B' y 'Bloque' normalizadas según las columnas detectadas."""
    df = df.copy()

    seg_col = next((c for c in seg_col_guess if c in df.columns), None)
    if seg_col is not None:
        df["S_low"] = normalize_S(df[seg_col], s_low_set, map_segtext_to_code, label_for_log=label_for_log)
    elif "S" in df.columns:
        df["S_low"] = df["S"].astype(str).str.lower()

    blk_col = next((c for c in block_col_guess if c in df.columns), None)
    if blk_col is not None:
        b = normalize_B(df[blk_col])
        df["B"] = b
        df["Bloque"] = b
    elif "B" in df.columns:
        df["B"] = normalize_B(df["B"])
        df["Bloque"] = df["B"]

    return df
