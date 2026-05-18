"""
Orquestador de generación de instancias del modelo de grúas.

Lee, por cada semana:
  - Instancia de coloración (Instancia_{semana}[_K].xlsx)
  - Resultado de coloración (resultado_{semana}[_K].xlsx)

Y escribe 21 Excels (uno por turno) con los parámetros que consume el modelo
de grúas. La lógica por turno vive en `transformaciones.py`; las constantes de
flota y zonas en `estaticos.py`; los lectores/normalizadores en `lectores.py`.
"""

import sys
from pathlib import Path

import pandas as pd

from .lectores import (
    build_normalizers,
    coerce_numeric,
    normalize_B,
    standardize_sheet,
    try_read_os,
    try_read_teu_sheet,
)
from .estaticos import build_static_sheets
from .transformaciones import (
    compute_AEbs_AIbs,
    compute_Cbs,
    compute_DMEst,
    compute_DMIst,
    compute_EIbs,
    compute_Gs,
)


N_TURNOS = 21
HORAS_POR_TURNO = 8


def _leer_instancia_base(file_instancia):
    """Lee hojas 'S', 'I0_sb', 'C_b', 'D_params_168h' y extras opcionales (TEU, OS)."""
    df_S = pd.read_excel(file_instancia, sheet_name="S")
    df_S["S_low"] = df_S["S"].astype(str).str.lower()

    s_low_set, map_segtext_to_code, teu_factor_by_S_low = build_normalizers(df_S)
    teu_factor_by_S_low = try_read_teu_sheet(file_instancia, teu_factor_by_S_low)

    df_S_E = df_S[df_S["Segregacion"].str.contains("expo", case=False, na=False)].copy()
    df_S_I = df_S[df_S["Segregacion"].str.contains("impo", case=False, na=False)].copy()

    df_i0 = pd.read_excel(file_instancia, sheet_name="I0_sb")
    df_i0 = standardize_sheet(df_i0, s_low_set, map_segtext_to_code, label_for_log="I0_sb")
    if "I0" not in df_i0.columns:
        raise ValueError("Hoja I0_sb no contiene columna 'I0'.")

    df_cb = pd.read_excel(file_instancia, sheet_name="C_b")
    df_cb["B_norm"] = normalize_B(df_cb["B"])
    cap_unit_map = df_cb.set_index("B_norm")["C"].to_dict()

    OS = try_read_os(file_instancia, default_val=1.0)

    dpar = pd.read_excel(file_instancia, sheet_name="D_params_168h")
    dpar["S_low"] = dpar["S"].astype(str).str.lower()
    for col in ("T", "DR", "DC", "DD", "DE"):
        if col in dpar.columns:
            dpar[col] = coerce_numeric(dpar[col]).fillna(0).astype(int)

    return {
        "df_S": df_S, "df_S_E": df_S_E, "df_S_I": df_S_I,
        "df_i0": df_i0, "cap_unit_map": cap_unit_map, "OS": OS,
        "dpar": dpar,
        "s_low_set": s_low_set,
        "map_segtext_to_code": map_segtext_to_code,
        "teu_factor_by_S_low": teu_factor_by_S_low,
    }


def _leer_resultado_coloracion(file_resultado, s_low_set, map_segtext_to_code):
    """Lee la hoja General del resultado de coloración y extrae los datos necesarios."""
    df_gen = pd.read_excel(file_resultado, sheet_name="General")
    df_gen = standardize_sheet(df_gen, s_low_set, map_segtext_to_code, label_for_log="General")
    df_gen["Periodo"] = coerce_numeric(df_gen["Periodo"]).astype("Int64")

    df_cargar = df_gen[["Segregación", "Bloque", "Periodo", "S_low", "B", "Carga"]].rename(columns={"Carga": "Cargar"})
    df_entregar = df_gen[["Segregación", "Bloque", "Periodo", "S_low", "B", "Entrega"]].rename(columns={"Entrega": "Entregar"})
    df_recibir = df_gen[["Segregación", "Bloque", "Periodo", "S_low", "B", "Recepción"]].rename(columns={"Recepción": "Recibir"})
    df_recibir["Recibir"] = coerce_numeric(df_recibir["Recibir"]).fillna(0).astype(int)

    df_volumen = df_gen[["Segregación", "Bloque", "Periodo", "S_low", "B", "Volumen (TEUs)"]].rename(columns={"Volumen (TEUs)": "Volumen"})
    df_volumen["Volumen"] = coerce_numeric(df_volumen["Volumen"])
    df_volumen = df_volumen.dropna(subset=["Volumen", "Periodo", "S_low", "B"])

    df_bahias = df_gen[["Segregación", "Bloque", "Periodo", "S_low", "B", "Bahías Ocupadas"]].rename(columns={"Bahías Ocupadas": "Bahías ocupadas"})
    df_bahias["Bahías ocupadas"] = coerce_numeric(df_bahias["Bahías ocupadas"]).fillna(0).astype(int)

    return {
        "df_cargar": df_cargar, "df_entregar": df_entregar, "df_recibir": df_recibir,
        "df_volumen": df_volumen, "df_bahias": df_bahias,
    }


def _escribir_turno(out_file, df_S, df_S_E, df_S_I, per_turno, static_sheets):
    """Escribe el Excel de un turno combinando hojas dinámicas y estáticas."""
    AEbs, AIbs, EIbs, DMEst, DMIst, Cbs, Gs = per_turno
    with pd.ExcelWriter(out_file, engine="openpyxl") as wr:
        # Sets S / S_E / S_I (siempre con códigos 's*')
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

        for sheet_name, df in static_sheets.items():
            df.to_excel(wr, sheet_name=sheet_name, index=False)


def generar_instancias_gruas(semanas, resultados_dir, n_turnos: int = N_TURNOS, usar_ki_flujo: bool = True):
    """
    Por cada semana factible, genera `n_turnos` Excels (uno por turno) para el
    modelo de grúas, leyendo la instancia y el resultado de coloración desde
    `{resultados_dir}/instancias_coloracion/{semana}` y `.../resultados_coloracion/{semana}`.

    n_turnos: 3=1día, 6=2días, 21=semana completa (default).
    """
    resultados_dir     = Path(resultados_dir)
    inst_color_root    = resultados_dir / "instancias_coloracion"
    res_color_root     = resultados_dir / "resultados_coloracion"
    inst_gruas_root    = resultados_dir / "instancias_gruas"

    static_sheets, BLOQUES = build_static_sheets()

    for semana in semanas:
        carpeta_inst = inst_color_root / semana
        carpeta_res  = res_color_root  / semana
        out_dir      = inst_gruas_root / f"instancias_turno_{semana}"
        out_dir.mkdir(parents=True, exist_ok=True)

        sufijo_k = "_K" if usar_ki_flujo else ""
        file_instancia = carpeta_inst / f"Instancia_{semana}{sufijo_k}.xlsx"
        file_resultado = carpeta_res  / f"resultado_{semana}{sufijo_k}.xlsx"

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

        try:
            base = _leer_instancia_base(file_instancia)
        except Exception as e:
            print(f"Error al leer base de instancia: {e}. Abortando.")
            sys.exit(1)

        try:
            resc = _leer_resultado_coloracion(
                file_resultado, base["s_low_set"], base["map_segtext_to_code"]
            )
        except Exception as e:
            print(f"Error al leer resultados: {e}. Abortando.")
            sys.exit(1)

        all_s_low = sorted(base["df_S"]["S_low"].unique())
        grid_bs = pd.MultiIndex.from_product(
            [BLOQUES, all_s_low], names=["Bloque", "S_low"]
        ).to_frame(index=False)

        print(f"Iniciando generación de archivos por turno (1..{n_turnos})...")
        for turno in range(1, n_turnos + 1):
            h_ini = (turno - 1) * HORAS_POR_TURNO + 1
            h_fin = turno * HORAS_POR_TURNO
            print(f"   Procesando Turno {turno:02d} (Horas {h_ini}-{h_fin})...")

            Gs   = compute_Gs(resc["df_recibir"], base["df_S_E"], turno)
            AEbs, AIbs = compute_AEbs_AIbs(
                turno, base["df_i0"], resc["df_volumen"],
                base["df_S_E"], base["df_S_I"],
                base["teu_factor_by_S_low"], grid_bs,
            )
            EIbs = compute_EIbs(resc["df_entregar"], grid_bs, turno)
            DMEst = compute_DMEst(base["dpar"], base["df_S_E"], h_ini, h_fin)
            DMIst = compute_DMIst(base["dpar"], base["df_S_I"], h_ini, h_fin)
            Cbs   = compute_Cbs(
                turno, resc["df_bahias"], base["df_i0"],
                all_s_low, BLOQUES,
                base["cap_unit_map"], base["OS"], base["teu_factor_by_S_low"],
            )

            out_file = out_dir / f"Instancia_{semana}_T{turno:02d}.xlsx"
            try:
                _escribir_turno(
                    out_file, base["df_S"], base["df_S_E"], base["df_S_I"],
                    (AEbs, AIbs, EIbs, DMEst, DMIst, Cbs, Gs),
                    static_sheets,
                )
                print(f"      ✓ Instancia turno {turno:02d} guardada: {out_file.name}")
            except Exception as e:
                print(f"      ✗ Error al escribir turno {turno:02d}: {e}")
