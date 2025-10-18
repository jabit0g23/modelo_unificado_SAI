import numpy as np
import pandas as pd
from pathlib import Path
import sys


# -------------------------------
# Helpers
# -------------------------------

def build_compat_df(blocks, colname):
    """Matriz completa de compatibilidad (1 por defecto) en formato largo: b1, b2, <colname>."""
    rows = [{"b1": b1, "b2": b2, colname: 1} for b1 in blocks for b2 in blocks]
    return pd.DataFrame(rows)

def get_size_from_segregation(seg_string):
    """Extrae 20/40 desde el texto de Segregacion; devuelve None si no se puede."""
    try:
        parts = str(seg_string).split('-')
        size = int(parts[2]) if len(parts) > 2 else None
        return size if size in (20, 40) else None
    except Exception:
        return None

def coerce_numeric(series):
    """Convierte una serie a numérico tolerando comas/strings; ignora nulos no convertibles."""
    s = series
    if s.dtype == "object":
        s = (s.astype(str).str.replace(",", "", regex=False))
    return pd.to_numeric(s, errors="coerce")

def build_normalizers(df_S):
    """
    Construye:
      - s_low_set: conjunto de códigos s* válidos
      - map_segtext_to_code: texto 'Segregacion' (lower) -> 's*'
      - teu_factor_by_S_low: S_low -> factor TEU (por defecto 1 para 20', 2 para 40')
    """
    df_S = df_S.copy()
    df_S["S_low"] = df_S["S"].astype(str).str.lower()
    # Tamaño desde texto para fallback
    if "Size" not in df_S.columns or df_S["Size"].isnull().any():
        df_S["Size"] = df_S["Segregacion"].apply(get_size_from_segregation)

    s_low_set = set(df_S["S_low"])
    map_segtext_to_code = (
        df_S.assign(seg_low=df_S["Segregacion"].astype(str).str.lower())
            .dropna(subset=["seg_low"])
            .set_index("seg_low")["S_low"]
            .to_dict()
    )
    # TEU factor por default desde Size (20->1, 40->2)
    teu_factor_by_S_low = df_S.set_index("S_low")["Size"].map(lambda z: 2 if z == 40 else 1).to_dict()

    return s_low_set, map_segtext_to_code, teu_factor_by_S_low

def try_read_teu_sheet(file_instancia, teu_factor_by_S_low):
    """
    Si existe una hoja con TEU por S (nombres comunes), la usa para actualizar el factor TEU.
    Columnas esperadas: 'S' o 'Segregacion' + 'TEU'
    """
    xl = pd.ExcelFile(file_instancia)
    candidates = [s for s in xl.sheet_names if s.strip().lower() in {"teu", "teus", "teus_instancia"}]
    if not candidates:
        return teu_factor_by_S_low  # no hay hoja opcional

    try:
        df_teu = pd.read_excel(file_instancia, sheet_name=candidates[0])
        cols = [c.lower() for c in df_teu.columns]
        if "teu" not in cols:
            return teu_factor_by_S_low
        # Normaliza S_low
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
            s = r["S_low"]
            v = int(r["TEU"])
            if v >= 1:
                teu_factor_by_S_low[s] = v
        print("   ✓ Hoja TEU detectada y aplicada.")
    except Exception as e:
        print(f"   ! Advertencia: no se pudo aplicar hoja TEU opcional: {e}")
    return teu_factor_by_S_low

def try_read_os(file_instancia, default_val=1.0):
    """
    Intenta leer OS (densidad) desde una hoja opcional 'OS'.
    Acepta:
      - una columna llamada 'OS', o
      - primer valor de la hoja.
    Si falla, retorna default_val (1.0).
    """
    try:
        df_os = pd.read_excel(file_instancia, sheet_name="OS")
        cols_lower = [c.lower() for c in df_os.columns]
        if "os" in cols_lower:
            val = df_os.iloc[0, cols_lower.index("os")]
        else:
            val = df_os.iloc[0, 0]
        v = float(val)
        return v if v > 0 else default_val
    except Exception:
        return default_val

def normalize_S(series, s_low_set, map_segtext_to_code, label_for_log=""):
    """
    Devuelve una serie de S_low (códigos s*) a partir de posibles valores:
      - ya codificados (S1 -> s1)
      - texto de segregación (expo-dry-...)
    """
    s_norm = series.astype(str).str.strip()
    s_low = s_norm.str.lower()

    # si ya está en set de códigos, ok
    mask_code = s_low.isin(s_low_set)
    # si no, intenta mapear texto -> código
    mapped = s_low.mask(mask_code).map(map_segtext_to_code)
    out = s_low.where(mask_code, mapped)

    # log de no mapeados
    bad = out[out.isna()]
    if not bad.empty:
        muestras = s_norm.loc[bad.index].unique()[:5]
        print(f"   ! Advertencia: {bad.size} filas con 'Segregación' no mapeada en {label_for_log}. Ejemplos:", list(muestras))
    return out

def normalize_B(series):
    """Convierte C# o B# a 'b#' (lower)."""
    x = series.astype(str).str.strip()
    # mapea C#->b#
    x = x.str.replace("^C", "b", regex=True)
    # normaliza a lowercase (H/T/I quedan como h/t/i)
    x = x.str.lower()
    return x

def standardize_sheet(df, s_low_set, map_segtext_to_code, label_for_log="", seg_col_guess=("Segregación", "Segregacion", "S"), block_col_guess=("Bloque", "B")):
    """Estándar: añade columnas 'S_low', 'B' y 'Bloque' normalizadas si existen."""
    df = df.copy()

    # S_low
    seg_col = next((c for c in seg_col_guess if c in df.columns), None)
    if seg_col is not None:
        df["S_low"] = normalize_S(df[seg_col], s_low_set, map_segtext_to_code, label_for_log=label_for_log)
    elif "S" in df.columns:
        df["S_low"] = df["S"].astype(str).str.lower()

    # B / Bloque
    blk_col = next((c for c in block_col_guess if c in df.columns), None)
    if blk_col is not None:
        b = normalize_B(df[blk_col])
        df["B"] = b
        df["Bloque"] = b
    elif "B" in df.columns:
        df["B"] = normalize_B(df["B"])
        df["Bloque"] = df["B"]

    return df


# --- Exclusividad a horizonte para RTG (pares legacy en Costanera) ---
ADYAC_RTG_B = {
    ('b1','b1'), ('b1','b3'),
    ('b2','b2'), ('b2','b4'),
    ('b3','b3'), ('b6','b3'),
    ('b4','b4'), ('b7','b4'),
    ('b5','b5'), ('b5','b8'),
    ('b6','b6'),
    ('b7','b7'),
    ('b8','b8'),
    ('b9','b9'),
}

def build_ex_rtg(B_C, BLOQUES, allowed_pairs):
    """
    EX_RTG[b1,b2] en {1,2}:
      - 2 si AMBOS son de Costanera y (b1,b2) es par permitido legacy (o su simétrico).
      - 2 también en la diagonal (b,b) si b es de Costanera.
      - 1 en cualquier otro caso.
    """
    rows = []
    allowed_sym = allowed_pairs | {(b2, b1) for (b1, b2) in allowed_pairs}
    for b1 in BLOQUES:
        for b2 in BLOQUES:
            if (b1 in B_C) and (b2 in B_C) and ((b1, b2) in allowed_sym):
                ex = 2
            elif (b1 == b2) and (b1 in B_C):
                ex = 2
            else:
                ex = 1
            rows.append({"b1": b1, "b2": b2, "EX": ex})
    return pd.DataFrame(rows)


# -------------------------------
# Main
# -------------------------------
def generar_instancias_gruas(semanas, participacion, resultados_dir):

    # Cantidad de grúas por tipo
    N_RTG = 12
    N_RS  = 11

    # Productividades por tipo (mov/hr)
    MU_RTG = 30
    MU_RS  = 20

    resultados_dir = Path(resultados_dir)
    inst_magdalena_root = resultados_dir / "instancias_magdalena"
    res_magdalena_root  = resultados_dir / "resultados_magdalena"
    inst_camila_root    = resultados_dir / "instancias_camila"

    for semana in semanas:

        carpeta_inst = inst_magdalena_root / semana
        carpeta_res  = res_magdalena_root  / semana
        out_dir      = inst_camila_root    / f"instancias_turno_{semana}"
        out_dir.mkdir(parents=True, exist_ok=True)

        file_instancia = carpeta_inst / f"Instancia_{semana}_{participacion}_K.xlsx"
        file_resultado = carpeta_res  / f"resultado_{semana}_{participacion}_K.xlsx"

        if not file_instancia.exists():
            print(f"✗ No encontré {file_instancia}. Saltando semana.")
            continue
        if not file_resultado.exists():
            print(f"✗ No encontré {file_resultado}. Saltando semana.")
            continue

        print(f"\n➡️ Semana {semana}")
        print("  • Instancia:", file_instancia.name)
        print("  • Resultado:", file_resultado.name)
        print("  • Salida   :", out_dir)

        # ==========================
        # 1) Datos base (instancia)
        # ==========================
        try:
            print("Leyendo datos base de instancia...")
            df_S = pd.read_excel(file_instancia, sheet_name="S")
            df_S["S_low"] = df_S["S"].astype(str).str.lower()

            # Normalizadores y TEU factor
            s_low_set, map_segtext_to_code, teu_factor_by_S_low = build_normalizers(df_S)
            teu_factor_by_S_low = try_read_teu_sheet(file_instancia, teu_factor_by_S_low)

            # Export / Import (a partir del texto de Segregacion)
            df_S_E = df_S[df_S["Segregacion"].str.contains("expo", case=False, na=False)].copy()
            df_S_I = df_S[df_S["Segregacion"].str.contains("impo", case=False, na=False)].copy()

            # I0_sb
            df_i0 = pd.read_excel(file_instancia, sheet_name="I0_sb")
            df_i0 = standardize_sheet(df_i0, s_low_set, map_segtext_to_code, label_for_log="I0_sb")
            if "I0" not in df_i0.columns:
                raise ValueError("Hoja I0_sb no contiene columna 'I0'.")

            # ► Capacidad por unidad (C_b) y densidad (OS)
            df_cb = pd.read_excel(file_instancia, sheet_name="C_b")
            df_cb["B_norm"] = normalize_B(df_cb["B"])
            cap_unit_map = df_cb.set_index("B_norm")["C"].to_dict()

            OS = try_read_os(file_instancia, default_val=1.0)
            print(f"   ✓ C_b leído ({len(cap_unit_map)} bloques). OS={OS:g}")

            # D_params_168h
            dpar = pd.read_excel(file_instancia, sheet_name="D_params_168h")
            dpar["S_low"] = dpar["S"].astype(str).str.lower()
            for col in ("T", "DR", "DC", "DD", "DE"):
                if col in dpar.columns:
                    dpar[col] = coerce_numeric(dpar[col]).fillna(0).astype(int)

            print("   ✓ Hojas base 'S', 'I0_sb' y 'D_params_168h' leídas y normalizadas.")

        except Exception as e:
            print(f"Error al leer base de instancia: {e}. Abortando.")
            sys.exit(1)

        # ==========================
        # 2) Datos de resultados
        # ==========================
        try:
            print("Leyendo datos de resultados...")
            df_cargar   = pd.read_excel(file_resultado, sheet_name="Cargar")
            df_entregar = pd.read_excel(file_resultado, sheet_name="Entregar")
            df_recibir  = pd.read_excel(file_resultado, sheet_name="Recibir")

            # normalización común
            df_cargar   = standardize_sheet(df_cargar,   s_low_set, map_segtext_to_code, label_for_log="Cargar")
            df_entregar = standardize_sheet(df_entregar, s_low_set, map_segtext_to_code, label_for_log="Entregar")
            df_recibir  = standardize_sheet(df_recibir,  s_low_set, map_segtext_to_code, label_for_log="Recibir")

            for dfx, name in [(df_cargar, "Cargar"), (df_entregar, "Entregar"), (df_recibir, "Recibir")]:
                if "Periodo" in dfx.columns:
                    dfx["Periodo"] = coerce_numeric(dfx["Periodo"]).astype("Int64")

            # asegurar columnas numéricas
            if "Recibir" not in df_recibir.columns:
                raise ValueError("Hoja 'Recibir' (resultados) no tiene columna 'Recibir'.")
            df_recibir["Recibir"] = coerce_numeric(df_recibir["Recibir"]).fillna(0).astype(int)

            # Volumen bloques (TEUs)
            SHEET_VOLUMEN = "Volumen bloques (TEUs)"
            df_volumen = pd.read_excel(file_resultado, sheet_name=SHEET_VOLUMEN)
            df_volumen = standardize_sheet(df_volumen, s_low_set, map_segtext_to_code, label_for_log=SHEET_VOLUMEN)
            req = {"Segregación", "Bloque", "Periodo", "Volumen"}
            if not req.issubset(set(df_volumen.columns)):
                missing = list(req - set(df_volumen.columns))
                raise ValueError(f"Faltan columnas en '{SHEET_VOLUMEN}': {missing}")
            df_volumen["Volumen"] = coerce_numeric(df_volumen["Volumen"])
            df_volumen["Periodo"] = coerce_numeric(df_volumen["Periodo"]).astype("Int64")
            df_volumen = df_volumen.dropna(subset=["Volumen", "Periodo", "S_low", "B"])

            # Bahías por bloques (v ya viene como v*TEU[s] desde el modelo de Magdalena)
            df_bahias = pd.read_excel(file_resultado, sheet_name="Bahías por bloques")
            df_bahias = standardize_sheet(df_bahias, s_low_set, map_segtext_to_code, label_for_log="Bahías por bloques")
            if "Bahías ocupadas" not in df_bahias.columns:
                raise ValueError("Hoja 'Bahías por bloques' no tiene 'Bahías ocupadas'.")
            df_bahias["Periodo"] = coerce_numeric(df_bahias["Periodo"]).astype("Int64")
            df_bahias["Bahías ocupadas"] = coerce_numeric(df_bahias["Bahías ocupadas"]).fillna(0).astype(int)

            print("   ✓ Hojas 'Cargar', 'Entregar', 'Recibir', 'Volumen bloques (TEUs)', 'Bahías por bloques' listas.")

        except Exception as e:
            print(f"Error al leer resultados: {e}. Abortando.")
            sys.exit(1)

        # ==========================
        # 3) Sets y estáticos
        # ==========================
        B_C = [f"b{i}" for i in range(1, 10)]  # Costanera: b1..b9
        B_H = [f"h{i}" for i in range(1, 6)]   # O'Higgins: h1..h5
        B_T = [f"t{i}" for i in range(1, 5)]   # Tebas: t1..t4
        B_I = [f"i{i}" for i in range(1, 3)]   # Imo: i1..i2

        BLOQUES   = sorted(B_C + B_H + B_T + B_I)
        ALL_S_LOW = sorted(df_S["S_low"].unique())

        # Grúas por tipo
        G_RTG = [f"rtg{i}" for i in range(1, N_RTG+1)]
        G_RS  = [f"rs{i}"  for i in range(1, N_RS+1)]
        G_ALL = G_RTG + G_RS

        df_static_G    = pd.DataFrame({"G": G_ALL})
        df_static_GRT  = pd.DataFrame({"GRT": G_RTG})
        df_static_GRS  = pd.DataFrame({"GRS": G_RS})

        df_static_B    = pd.DataFrame({"B": BLOQUES})
        df_static_B_E  = pd.DataFrame({"B_E": BLOQUES})
        df_static_B_I  = pd.DataFrame({"B_I": BLOQUES}) 
        df_BC = pd.DataFrame({"BC": B_C})
        df_BT = pd.DataFrame({"BT": B_T})
        df_BH = pd.DataFrame({"BH": B_H})
        df_BI = pd.DataFrame({"BI": B_I})

        df_static_T    = pd.DataFrame({"T": list(range(1, 9))})

        # Parámetros escalares “históricos” (se mantienen por compatibilidad)
        df_static_mu   = pd.DataFrame({"mu": [300]})
        df_static_W    = pd.DataFrame({"W": [3]})
        df_static_K    = pd.DataFrame({"K": [2]})  # ahora el modelo usará K_g; esto queda como referencia
        
        df_Wb = pd.DataFrame({"B": BLOQUES, "W_b": [3]*len(BLOQUES)})

        # Disponibilidades por tipo (opcional informativo)
        df_static_Rmax_rtg = pd.DataFrame({"Rmax": [len(G_RTG)]})
        df_static_Rmax_rs  = pd.DataFrame({"Rmax": [len(G_RS)]})

        # Compatibilidades por tipo para SIMULTANEIDAD (1=permitido para todos)
        df_CBR = build_compat_df(BLOQUES, "CBR")  # RTG
        df_CBS = build_compat_df(BLOQUES, "CBS")  # RS

        # Exclusividad a horizonte para RTG (legacy en Costanera)
        df_EX_RTG = build_ex_rtg(B_C, BLOQUES, ADYAC_RTG_B)

        # K por grúa: RTG=2, RS=1
        df_Kg = pd.DataFrame({
            "G": G_ALL,
            "K": ([2]*len(G_RTG)) + ([1]*len(G_RS))
        })

        print("✓ Definiciones de hojas estáticas creadas.")
        print("-" * 40)

        # grilla B×S (códigos)
        grid_bs = pd.MultiIndex.from_product([BLOQUES, ALL_S_LOW],
                                             names=["Bloque", "S_low"]).to_frame(index=False)

        # ==========================
        # 4) Generación por turno
        # ==========================
        print("Iniciando generación de archivos por turno...")
        for turno in range(1, 22):
            h_ini, h_fin = (turno - 1) * 8 + 1, turno * 8
            print(f"   Procesando Turno {turno:02d} (Horas {h_ini}-{h_fin})...")

            # ---- Gs: recepciones de export (RECV) desde 'Recibir' del archivo de resultados
            rec_turno = df_recibir.query("Periodo == @turno and S_low in @df_S_E.S_low").copy()
            Gs_calc = (
                rec_turno.groupby("S_low", as_index=False)["Recibir"]
                         .sum()
                         .rename(columns={"S_low": "S_E", "Recibir": "Gs"})
            )
            Gs = (
                pd.DataFrame({"S_E": df_S_E["S_low"].unique()})
                  .merge(Gs_calc, on="S_E", how="left")
                  .fillna({"Gs": 0})
                  .astype({"Gs": int})
            )

            # ---- Inventario previo: AEbs/AIbs
            if turno == 1:
                # usar I0_sb (en contenedores)
                i0 = (df_i0.groupby(["Bloque", "S_low"], as_index=False)["I0"].sum())
                inv_prev_full = (grid_bs.merge(i0, on=["Bloque", "S_low"], how="left")
                                      .fillna({"I0": 0})
                                      .rename(columns={"I0": "cont_prev"}))
            else:
                # usar Volumen (TEUs) del periodo previo y convertir a contenedores según TEU factor de S
                prev_turno = turno - 1
                vol_prev = df_volumen[df_volumen["Periodo"] == prev_turno].copy()
                if vol_prev.empty:
                    vol_prev = df_volumen[df_volumen["Periodo"] == turno].copy()

                # TEUs -> contenedores
                vol_prev["teu_factor"] = vol_prev["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
                vol_prev["cont_prev"] = (vol_prev["Volumen"] / vol_prev["teu_factor"]).round().fillna(0).astype(int)

                inv_prev = (vol_prev.groupby(["B", "S_low"], as_index=False)["cont_prev"].sum()
                                   .rename(columns={"B": "Bloque"}))

                inv_prev_full = (grid_bs.merge(inv_prev, on=["Bloque", "S_low"], how="left")
                                      .fillna({"cont_prev": 0}))

            # AEbs: sólo export
            AEbs = (inv_prev_full[inv_prev_full["S_low"].isin(df_S_E["S_low"].unique())]
                    .rename(columns={"Bloque": "B_E", "S_low": "S_E", "cont_prev": "AEbs"})
                    .sort_values(["B_E", "S_E"]))

            # AIbs: sólo import
            AIbs = (inv_prev_full[inv_prev_full["S_low"].isin(df_S_I["S_low"].unique())]
                    .rename(columns={"Bloque": "B_I", "S_low": "S_I", "cont_prev": "AIbs"})
                    .sort_values(["B_I", "S_I"]))

            # ---- EIbs: plan de ENTREGAR del turno (import)
            if "Entregar" in df_entregar.columns:
                eibs = (df_entregar.query("Periodo == @turno")
                        .groupby(["B", "S_low"], as_index=False)["Entregar"].sum())
            else:
                eibs = pd.DataFrame(columns=["B", "S_low", "Entregar"])

            EIbs = (grid_bs.merge(eibs.rename(columns={"B": "Bloque"}), on=["Bloque", "S_low"], how="left")
                           .fillna({"Entregar": 0})
                           .astype({"Entregar": int})
                           .rename(columns={"Bloque": "B_I", "S_low": "S_I", "Entregar": "EIbs"})
                           .sort_values(["B_I", "S_I"]))

            # ---- DMEst (demanda de cargar export dentro del turno)
            exp = dpar.query("@h_ini <= T <= @h_fin and S_low in @df_S_E.S_low").copy()
            if not exp.empty:
                exp["T_rel"] = exp["T"] - h_ini + 1
                DMEst_calc = exp[["S_low", "T_rel", "DC"]].rename(
                    columns={"S_low": "S_E", "T_rel": "T", "DC": "DMEst"})
            else:
                DMEst_calc = pd.DataFrame(columns=["S_E", "T", "DMEst"])
            DMEst = (pd.MultiIndex.from_product([df_S_E.S_low.unique(), range(1, 9)],
                                                names=["S_E", "T"]).to_frame(index=False)
                     .merge(DMEst_calc, on=["S_E", "T"], how="left")
                     .fillna({"DMEst": 0})
                     .astype({"DMEst": int}))

            # ---- DMIst (demanda de descargar import dentro del turno)
            imp = dpar.query("@h_ini <= T <= @h_fin and S_low in @df_S_I.S_low").copy()
            if not imp.empty:
                imp["T_rel"] = imp["T"] - h_ini + 1
                DMIst_calc = imp[["S_low", "T_rel", "DD"]].rename(
                    columns={"S_low": "S_I", "T_rel": "T", "DD": "DMIst"})
            else:
                DMIst_calc = pd.DataFrame(columns=["S_I", "T", "DMIst"])
            DMIst = (pd.MultiIndex.from_product([df_S_I.S_low.unique(), range(1, 9)],
                                                names=["S_I", "T"]).to_frame(index=False)
                     .merge(DMIst_calc, on=["S_I", "T"], how="left")
                     .fillna({"DMIst": 0})
                     .astype({"DMIst": int}))

            # ---- Cbs (capacidad de contenedores por unidad asignada)
            bayas_actual = df_bahias[df_bahias["Periodo"] == turno].copy()
            bayas_prev   = df_bahias[df_bahias["Periodo"] == (turno - 1)].copy()
            if bayas_prev.empty:
                bayas_prev = bayas_actual.copy()

            bayas_actual = bayas_actual.rename(columns={"Bahías ocupadas": "v"})      # v*TEU[s]
            bayas_prev   = bayas_prev.rename(columns={"Bahías ocupadas": "v_prev"})   # v*TEU[s] (t-1)

            bayas = (pd.merge(bayas_actual[["S_low", "B", "v"]],
                              bayas_prev[["S_low", "B", "v_prev"]],
                              on=["S_low", "B"], how="outer").fillna(0))

            # Capacidad en TEUs = max(v*TEU, v_prev*TEU) * C[b] * OS
            bayas["max_v"]    = bayas[["v", "v_prev"]].max(axis=1)
            bayas["cap_unit"] = bayas["B"].map(cap_unit_map).fillna(0).astype(int)
            bayas["cap_teu"]  = bayas["max_v"] * bayas["cap_unit"] * OS

            # Contenedores equivalentes (por segregación) = ceil( cap_teu / TEU_factor(S) )
            bayas["teu_factor"]    = bayas["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
            bayas["Cbs_calculado"] = np.ceil(bayas["cap_teu"] / bayas["teu_factor"]).astype(int)

            calculated_cbs_data = bayas.groupby(["B", "S_low"])["Cbs_calculado"].sum().reset_index()

            full_grid_cbs = pd.MultiIndex.from_product([BLOQUES, ALL_S_LOW],
                                                       names=["B", "S_low"]).to_frame(index=False)
            Cbs_tmp = (full_grid_cbs.merge(calculated_cbs_data, on=["B", "S_low"], how="left")
                                   .fillna({"Cbs_calculado": 0}))

            if turno == 1:
                # Considerar I0 para forzar capacidad mínima compatible con inventario inicial
                i0_bs = (df_i0.groupby(["B", "S_low"], as_index=False)["I0"].sum()
                               .rename(columns={"I0": "I0_cont"}))
                i0_bs["teu_factor"] = i0_bs["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
                i0_bs["cap_unit"]   = i0_bs["B"].map(cap_unit_map).fillna(0).astype(int)

                # Unidades mínimas requeridas (en “unidades equivalentes 20’”)
                # b0 = ceil( (I0_cont * TEU_factor) / (C[b] * OS) )
                i0_bs["b0"]   = np.ceil((i0_bs["I0_cont"] * i0_bs["teu_factor"]) /
                                        (i0_bs["cap_unit"] * OS + 1e-9)).astype(int)

                # Capacidad en contenedores si asignas esas unidades
                # Cbs0 = ceil( (b0 * C[b] * OS) / TEU_factor )
                i0_bs["Cbs0"] = np.ceil((i0_bs["b0"] * i0_bs["cap_unit"] * OS) /
                                        (i0_bs["teu_factor"] + 1e-9)).astype(int)

                Cbs_tmp = (Cbs_tmp.merge(i0_bs[["B", "S_low", "Cbs0"]], on=["B", "S_low"], how="left")
                                   .fillna({"Cbs0": 0}))
                Cbs_tmp["Cbs_final"] = Cbs_tmp[["Cbs_calculado", "Cbs0"]].max(axis=1)
            else:
                Cbs_tmp["Cbs_final"] = Cbs_tmp["Cbs_calculado"]

            Cbs = (Cbs_tmp[["B", "S_low", "Cbs_final"]]
                   .rename(columns={"S_low": "S", "Cbs_final": "Cbs"})
                   .astype({"Cbs": int})
                   .sort_values(["B", "S"]))

            # ==========================
            # 5) Escribir Excel
            # ==========================
            out_file = out_dir / f"Instancia_{semana}_{participacion}_T{turno:02d}.xlsx"
            try:
                with pd.ExcelWriter(out_file, engine="openpyxl") as wr:
                    # Sets S / S_E / S_I (siempre con códigos en 'S*')
                    df_S[["S_low", "Segregacion"]].rename(columns={"S_low": "S"}).to_excel(wr, sheet_name="S", index=False)
                    df_S_E[["S_low", "Segregacion"]].rename(columns={"S_low": "S_E"}).to_excel(wr, sheet_name="S_E", index=False)
                    df_S_I[["S_low", "Segregacion"]].rename(columns={"S_low": "S_I"}).to_excel(wr, sheet_name="S_I", index=False)

                    AEbs.to_excel(wr, sheet_name="AEbs", index=False)
                    AIbs.to_excel(wr, sheet_name="AIbs", index=False)
                    DMEst.to_excel(wr, sheet_name="DMEst", index=False)
                    DMIst.to_excel(wr, sheet_name="DMIst", index=False)
                    Cbs.to_excel(wr, sheet_name="Cbs", index=False)
                    Gs.to_excel(wr, sheet_name="Gs", index=False)
                    EIbs.to_excel(wr, sheet_name="EIbs", index=False)

                    df_static_G.to_excel(wr, sheet_name="G", index=False)
                    df_static_B.to_excel(wr, sheet_name="B", index=False)
                    df_static_B_I.to_excel(wr, sheet_name="B_I", index=False)
                    df_static_B_E.to_excel(wr, sheet_name="B_E", index=False)
                    df_static_T.to_excel(wr, sheet_name="T", index=False)
                    df_static_mu.to_excel(wr, sheet_name="mu", index=False)
                    df_static_W.to_excel(wr, sheet_name="W", index=False)
                    df_static_K.to_excel(wr, sheet_name="K", index=False)

                    df_static_Rmax_rtg.to_excel(wr, sheet_name="Rmax_rtg", index=False)
                    df_static_Rmax_rs.to_excel(wr, sheet_name="Rmax_rs", index=False)

                    # Compatibilidades por tipo (simultaneidad)
                    df_CBR.to_excel(wr, sheet_name="CBR", index=False)  # columnas: b1, b2, CBR
                    df_CBS.to_excel(wr, sheet_name="CBS", index=False)  # columnas: b1, b2, CBS

                    # Exclusividad a horizonte para RTG y K por grúa
                    df_EX_RTG.to_excel(wr, sheet_name="EX_RTG", index=False)  # columnas: b1, b2, EX
                    df_Kg.to_excel(wr, sheet_name="K_g", index=False)         # columnas: G, K

                    # Zonas y tipos de grúa
                    df_BC.to_excel(wr, sheet_name="BC", index=False)
                    df_BT.to_excel(wr, sheet_name="BT", index=False)
                    df_BH.to_excel(wr, sheet_name="BH", index=False)
                    df_BI.to_excel(wr, sheet_name="BI", index=False)
                    df_static_GRT.to_excel(wr, sheet_name="GRT", index=False)
                    df_static_GRS.to_excel(wr, sheet_name="GRS", index=False)
                    
                    df_Wb.to_excel(wr, sheet_name="W_b", index=False) 

                    # Productividades por tipo
                    pd.DataFrame(
                        [{"Tipo": "RTG", "Prod": MU_RTG},
                         {"Tipo": "RS",  "Prod": MU_RS}]
                    ).to_excel(wr, sheet_name="PROD", index=False)

                print(f"      ✓ Instancia turno {turno:02d} guardada: {out_file.name}")
            except Exception as e:
                print(f"      ✗ Error al escribir turno {turno:02d}: {e}")

        # --------------------------
        # Fin semana
        # --------------------------
