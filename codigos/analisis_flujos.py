import os
import pandas as pd
import numpy as np
from pathlib import Path

"""
Procesa Flujos_w{semana}.xlsx y genera:
- analisis_flujos_w{semana}_0.xlsx (salida principal, insumo para la instancia)
- analisis_flujos_w{semana}_debug.xlsx (diagnóstico de filas descartadas/inconsistencias)

"""

# ==== FLAGS (comportamiento global) ====
MODO_TODO_ENTRA = False
FILTRA_IME = True  # OFF => pasan TODOS los IME


def run_analysis_flujos(semana, resultados_dir, estaticos_flujos_dir, criterio_flujos="criterio_ii", debug=True):
    """Pipeline de análisis de flujos:
    - Lee y normaliza IME/tiempos dentro de la semana.
    - Filtra por categoría y, según flags, por prefijos/IME.
    - Asigna turnos (3 por día → 21 en 7 días) y genera bases canónicas:
        * FlujosAll_sbt / st / s
        * Versiones “_P” (colapsa IME de patio a 'Patio')
        * Series horarias 168h (consistentes con turnos)
    - Calcula inventario inicial y un resumen “flujos_vs” (movimientos vinculados a Patio).
    - Exporta Excel principal y otro de depuración con las filas descartadas/alertas.
    """

    # ── 1) Parámetros estáticos de validación/normalización ──
    valid_iu_category = ['IMPRT', 'EXPRT', 'TRSHP']
    BLOQUES_PATIO = [
        'C1','C2','C3','C4','C5','C6','C7','C8','C9',
        'H1','H2','H3','H4','H5','T1','T2','T3','T4','I1','I2'
    ]
    valid_ime_values = BLOQUES_PATIO + [
        'GATE', 'BUQUE', 'Y-SAI-1', 'Y-SAI-2', 'Y-SAI-M10', 'Y-SAI-ANDEN', 'Y-SAI-???',
        #'PA','Y-SAI-T1100','F1','Y-SAI-RAMP','V-M10','T-BDHC41-TIP-Y',
        #'SK','T-HKFF77-TIP-U', 'Y-SAI-3', 'T-PRYS76-TIP-Y',
        #'T-PG2775', 'T-YY4257', 'T-T118', 'Y-SAI-BR',
        #'T-UF2977 (TIP)','T-UF2977-TIP'
    ]
    valid_prefixes = (
        "expo-dry-","expo-empty-","expo-reefer-","expo-imo-",
        "impo-dry-","impo-empty-","impo-reefer-","impo-imo-",
    )

    # ── 2) Rutas + lectura de archivo fuente ──
    estaticos_flujos_dir = os.path.join(estaticos_flujos_dir, "flujos")
    input_file = os.path.join(estaticos_flujos_dir, f"Flujos_w{semana}.xlsx")

    df = pd.read_excel(input_file).reset_index(drop=True)
    df["__rowid__"] = np.arange(len(df), dtype=int)
    if debug:
        print(f"Archivo leído: {input_file}")
        print(f"Filas iniciales: {len(df)}")

    debug_rows = {}

    def _pack_for_debug(_df, reason):
        """Recorta columnas y anota motivo de descarte para el Excel de depuración."""
        cols = [
            "__rowid__", "ime_time", "ime_fm", "ime_to", "ime_move_kind",
            "criterio_i", "criterio_ii", "criterio_iii", "iu_category",
            "iu_freight_kind", "ret_nominal_length"
        ]
        cols = [c for c in cols if c in _df.columns]
        out = _df[cols].copy()
        out.insert(1, "__motivo__", reason)
        return out

    # ── 3) Normalización temporal + recorte a ventana semanal ──
    df["ime_time"] = pd.to_datetime(df["ime_time"], errors="coerce")
    df = df.sort_values("ime_time")
    start_date = pd.to_datetime(semana) + pd.Timedelta(hours=8)
    end_date = start_date + pd.Timedelta(days=7)
    if debug:
        print(f"Ventana de planificación: [{start_date} , {end_date})")

    mask_in_week = (df["ime_time"] >= start_date) & (df["ime_time"] < end_date)
    fuera_semana = df[~mask_in_week]
    if len(fuera_semana) > 0:
        debug_rows["01_fuera_semana"] = _pack_for_debug(fuera_semana, "fuera_de_semana")
    df = df[mask_in_week].copy()
    if debug:
        print(f"Tras recorte a semana: {len(df)} (eliminadas: {len(fuera_semana)})\n")

    # ── 3.1) Normalización de IME (completa vacíos por tipo de movimiento) ──
    for col in ["ime_fm", "ime_to"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(str).str.strip()
        df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    mask_recv = df["ime_move_kind"] == "RECV"
    mask_load = df["ime_move_kind"] == "LOAD"
    mask_dsch = df["ime_move_kind"] == "DSCH"
    mask_dlvr = df["ime_move_kind"] == "DLVR"
    df.loc[mask_recv & df["ime_fm"].isna(), "ime_fm"] = "GATE"
    df.loc[mask_load & df["ime_to"].isna(), "ime_to"] = "GATE"
    df.loc[mask_dsch & df["ime_fm"].isna(), "ime_fm"] = "BUQUE"
    df.loc[mask_dlvr & df["ime_to"].isna(), "ime_to"] = "GATE"
    df["ime_fm"].fillna("IME_DESCONOCIDO", inplace=True)
    df["ime_to"].fillna("IME_DESCONOCIDO", inplace=True)

    # ── 4) Filtro por categoría  ──
    mask_cat = df["iu_category"].isin(valid_iu_category)
    cat_invalidas = df[~mask_cat]
    if len(cat_invalidas) > 0:
        debug_rows["02_cat_invalidas"] = _pack_for_debug(cat_invalidas, "iu_category_no_valida")
    df = df[mask_cat].copy()

    # ── 5) Criterio/carrier + prefijos ──
    """
    valida que criterio_flujos exista, crea criterio, derivar carrier, aplica prefijos, manda a debug lo no válido.
    """
    if criterio_flujos not in df.columns:
        raise ValueError(f"La columna '{criterio_flujos}' no existe en Flujos_w{semana}.xlsx")
    df["criterio"] = df[criterio_flujos]
    df["carrier"]  = df["criterio"].apply(lambda c: c.split("-")[-1] if isinstance(c, str) else "")
    if MODO_TODO_ENTRA:
        df["criterio"] = df["criterio"].fillna("criterio_desconocido")
        df["carrier"]  = df["carrier"].fillna("carrier_desconocido")
    else:
        mask_crit_notna = df["criterio"].notna()
        criterio_nan = df[~mask_crit_notna]
        if len(criterio_nan) > 0:
            debug_rows["03_criterio_nan"] = _pack_for_debug(criterio_nan, "criterio_nan")
        df_notna = df[mask_crit_notna].copy()
        mask_pref = df_notna["criterio"].str.startswith(valid_prefixes)
        prefijo_no_valido = df_notna[~mask_pref]
        if len(prefijo_no_valido) > 0:
            debug_rows["04_prefijo_no_valido"] = _pack_for_debug(prefijo_no_valido, "criterio_prefijo_no_valido")
        df = df_notna[mask_pref].copy()

    # ── 6) Asignación de turnos (21 en la semana) ──
    def assign_period(time):
        """Mapea ime_time → turno [1..21] (3 turnos/día de 8h)."""
        if pd.isnull(time) or time < start_date or time >= end_date:
            return None
        delta = time - start_date
        days_passed = delta.days
        hours_passed = (delta.seconds // 3600) % 24
        period = days_passed * 3
        if 0 <= hours_passed < 8:   return period + 1
        if 8 <= hours_passed < 16:  return period + 2
        return period + 3

    df["shift"] = df["assign_shift"] = df["ime_time"].apply(assign_period)
    turno_fuera = df[df["shift"].isna()]
    if len(turno_fuera) > 0:
        debug_rows["05_turno_fuera"] = _pack_for_debug(turno_fuera, "turno_fuera_horizonte")
    df = df.dropna(subset=["shift"]).copy()
    df["shift"] = df["shift"].astype(int)

    # ── 6b) Foto global de IME fuera de lista (solo diagnóstico) ──
    invalid_ime_mask = (~df['ime_fm'].isin(valid_ime_values)) | (~df['ime_to'].isin(valid_ime_values))
    invalid_ime_global = df[invalid_ime_mask].copy()
    if len(invalid_ime_global) > 0:
        dbg = _pack_for_debug(invalid_ime_global, "ime_no_valido_global")
        dbg["__fm_invalido__"] = ~invalid_ime_global['ime_fm'].isin(valid_ime_values)
        dbg["__to_invalido__"] = ~invalid_ime_global['ime_to'].isin(valid_ime_values)
        debug_rows["07_ime_no_validos_global"] = dbg  # diagnóstico, no filtra

    # ── 7) Filtro por (criterio, carrier) si no está en modo “todo entra” ──
    if not MODO_TODO_ENTRA:
        interes = ['RECV','LOAD','DSCH','DLVR','OTHR','YARD','SHFT']
        df_before_pairs = df.copy()
        valid_criterio_idx = (
            df[df['ime_move_kind'].isin(interes)]
            .groupby(['criterio','carrier'])['ime_move_kind']
            .count()
        )
        valid_criterio_idx = valid_criterio_idx[valid_criterio_idx > 0].index
        df = df.set_index(['criterio','carrier']).loc[valid_criterio_idx].reset_index()
        mask_pairs_kept = df_before_pairs.set_index(['criterio','carrier']).index.isin(valid_criterio_idx)
        cc_sin_mov = df_before_pairs[~mask_pairs_kept]
        if len(cc_sin_mov) > 0:
            debug_rows["06_cc_sin_mov"] = _pack_for_debug(cc_sin_mov, "criterio_carrier_sin_mov_interes")

    # ── 7.5) Base canónica común para 168h y All_sbt_P ──
    interes = ['RECV','LOAD','DSCH','DLVR']

    def to_patio(x):  # colapsa IME de patio a etiqueta 'Patio'
        return 'Patio' if x in set(BLOQUES_PATIO) else x

    df_core = df.copy()
    df_core['ime_move_kind'] = df_core['ime_move_kind'].astype(str).str.strip().str.upper()
    df_core = df_core[df_core['ime_move_kind'].isin(interes)].copy()
    df_core['T'] = ((df_core['ime_time'] - start_date).dt.total_seconds() // 3600).astype(int) + 1
    df_core = df_core[(df_core['T'] >= 1) & (df_core['T'] <= 168)].copy()
    df_core['shift'] = df_core['assign_shift'].astype(int)
    df_core['ime_fm_g'] = df_core['ime_fm'].apply(to_patio)
    df_core['ime_to_g'] = df_core['ime_to'].apply(to_patio)

    def pivot_counts(local, group_cols):
        """Agrupa por columnas + tipo de movimiento y pivotea a RECV/LOAD/DSCH/DLVR."""
        tmp = (local.groupby(group_cols + ['ime_move_kind'])
                     .size().reset_index(name='count'))
        out = (tmp.pivot_table(values='count', index=group_cols,
                               columns='ime_move_kind', fill_value=0)
                  .reset_index())
        for c in interes:
            if c not in out.columns:
                out[c] = 0
        return out

    # == Turno (SBT-P): misma base, distinta granularidad == 
    flujos_all_sbt_p = pivot_counts(
        df_core,
        ['criterio','carrier','ime_fm_g','ime_to_g','shift']
    ).rename(columns={'ime_fm_g':'ime_fm','ime_to_g':'ime_to'})

    # == 168h (horario) ==
    flujos_168h = pivot_counts(
        df_core,
        ['criterio','T']
    ).rename(columns={'criterio':'Segregacion'})

    # (Diagnóstico) 168h con IME y carrier
    flujos_168h_P = pivot_counts(
        df_core,
        ['criterio','carrier','ime_fm_g','ime_to_g','T']
    ).rename(columns={'ime_fm_g':'ime_fm','ime_to_g':'ime_to'})

    # Chequeo de consistencia: totales por criterio turno vs 168h
    try:
        a = (flujos_all_sbt_p.groupby('criterio')[['RECV','LOAD','DSCH','DLVR']].sum().sort_index())
        b = (flujos_168h.rename(columns={'Segregacion':'criterio'})
                       .groupby('criterio')[['RECV','LOAD','DSCH','DLVR']].sum().sort_index())
        diff = a.reindex(b.index).fillna(0) - b.fillna(0)
        bad = diff[(diff != 0).any(axis=1)]
        if len(bad) > 0:
            print("⚠️ Inconsistencias (turno vs hora) por criterio:")
            print(bad)
    except Exception as _e:
        print(f"(Aviso) Chequeo de consistencia no ejecutado: {_e}")

    # ── 8) Agrupadores utilitarios para distintas vistas ──
    def create_grouped_df(data,
                          include_ime=True,
                          filter_move_kind=False,
                          only_visitas=False,
                          exclude_patio=False,
                          group_patio=False,
                          _label=""):
        """Genera vistas agregadas (SBT/ST/S) con switches:
        - include_ime: incluye columnas ime_fm/ime_to
        - filter_move_kind: limita a movimientos de interés ampliados
        - only_visitas: agrega sin turno (por visita)
        - exclude_patio: excluye filas ya colapsadas a 'Patio'
        - group_patio: colapsa IME de patio → 'Patio'
        - _label: sufijo para etiquetar diagnósticos
        """
        local = data
        if FILTRA_IME:
            mask_ime = local['ime_fm'].isin(valid_ime_values) & local['ime_to'].isin(valid_ime_values)
            inv = local[~mask_ime].copy()
            if debug and len(inv) > 0:
                dbg = _pack_for_debug(inv, f"ime_no_valido_{_label}")
                dbg["__fm_invalido__"] = ~inv['ime_fm'].isin(valid_ime_values)
                dbg["__to_invalido__"] = ~inv['ime_to'].isin(valid_ime_values)
                sheet = f"07_ime_inv_{_label}"[:31]
                debug_rows[sheet] = dbg
            local = local[mask_ime].copy()

        if exclude_patio:
            mask_patio = (local['ime_fm'] != 'Patio') & (local['ime_to'] != 'Patio')
            local = local[mask_patio].copy()

        if filter_move_kind:
            mask_kind = local['ime_move_kind'].isin(['RECV','LOAD','DSCH','DLVR','OTHR','YARD','SHFT'])
            local = local[mask_kind].copy()

        if group_patio and include_ime:
            patio_set = set(BLOQUES_PATIO)
            def to_patio_local(x): return 'Patio' if x in patio_set else x
            local.loc[:, 'ime_fm'] = local['ime_fm'].apply(to_patio_local)
            local.loc[:, 'ime_to'] = local['ime_to'].apply(to_patio_local)

        group_cols = ['criterio','carrier']
        if include_ime:
            group_cols += ['ime_fm','ime_to']
        if not only_visitas:
            group_cols += ['shift']
        group_cols += ['ime_move_kind']

        result = (local.groupby(group_cols, as_index=False)
                        .size().rename(columns={'size':'count'}))
        index_cols = [c for c in group_cols if c != 'ime_move_kind']
        result = result.pivot_table(values='count', index=index_cols,
                                    columns='ime_move_kind', fill_value=0).reset_index()

        # Completa grilla de turnos para evitar huecos
        if not only_visitas and 'shift' in index_cols:
            key_cols = [c for c in index_cols if c != 'shift']
            all_keys = result[key_cols].drop_duplicates() if len(result) > 0 else local[key_cols].drop_duplicates()
            all_shifts = pd.DataFrame({'shift': range(1, 22)})
            if len(all_keys) == 0:
                result = pd.DataFrame(columns=key_cols + ['shift'])
            else:
                full_grid = (all_keys.assign(_k=1).merge(all_shifts.assign(_k=1), on='_k').drop(columns='_k'))
                result = full_grid.merge(result, on=key_cols + ['shift'], how='left')
                numeric_cols = result.columns.difference(key_cols + ['shift'])
                result[numeric_cols] = result[numeric_cols].fillna(0)

        return result

    # ── 9) Vistas finales (además del canónico flujos_all_sbt_p ya calculado) ──
    flujos_all_sbt   = create_grouped_df(df, include_ime=True,  _label="all_sbt")
    flujos_all_st    = create_grouped_df(df, include_ime=False, _label="all_st")
    flujos_all_s     = create_grouped_df(df, include_ime=False, only_visitas=True, _label="all_s")

    flujos_sbt       = create_grouped_df(df, include_ime=True,  filter_move_kind=False, exclude_patio=False, _label="sbt")
    flujos_st        = create_grouped_df(df, include_ime=False, filter_move_kind=False, exclude_patio=False, _label="st")
    flujos_s         = create_grouped_df(df, include_ime=False, filter_move_kind=False, only_visitas=True, exclude_patio=False, _label="s")

    flujos_all_sb_p  = create_grouped_df(df, include_ime=True,  only_visitas=True, group_patio=True, _label="all_sb_P")

    # ── 9.1) Diagnósticos opcionales si no filtramos IME ──
    if not FILTRA_IME:
        invalids_df = df[(~df['ime_fm'].isin(valid_ime_values)) | (~df['ime_to'].isin(valid_ime_values))].copy()
        if len(invalids_df) > 0:
            debug_rows["ZZ_invalids_in_df"] = _pack_for_debug(invalids_df, "persisten_en_df")
        if not flujos_all_sbt.empty:
            mask_out_invalid = (~flujos_all_sbt['ime_fm'].isin(valid_ime_values)) | (~flujos_all_sbt['ime_to'].isin(valid_ime_values))
            invalids_out = flujos_all_sbt[mask_out_invalid].copy()
            if len(invalids_out) > 0:
                invalids_out = invalids_out.sort_values(['criterio','carrier']).reset_index(drop=True)
                debug_rows["ZZ_invalids_in_outputs_all_sbt"] = invalids_out

    # ── 9.2) Inventario inicial (LOAD/DLVR) y ajuste por 20/40 ──
    inventario_inicial_s = (
        df[df['ime_move_kind'].isin(['LOAD','DLVR'])]
          .groupby(['criterio','carrier'])['ime_move_kind']
          .count().reset_index()
          .rename(columns={'ime_move_kind':'inventario_inicial'})
    )
    inventario_inicial = (
        df[df['ime_move_kind'].isin(['LOAD','DLVR'])]
          .groupby(['criterio','carrier','ime_fm'])['ime_move_kind']
          .count().reset_index()
          .rename(columns={'ime_move_kind':'inventario_inicial'})
    )

    def ajustar_inventario(row):
        """Convierte unidades a capacidad: 20'×1, 40'×2 según el nombre del criterio."""
        crit = str(row['criterio'])
        if '-40-' in crit: return row['inventario_inicial'] * 2
        if '-20-' in crit: return row['inventario_inicial'] * 1
        return row['inventario_inicial']

    inventario_inicial_s['capacidad_ajustada'] = inventario_inicial_s.apply(ajustar_inventario, axis=1)
    inventario_inicial['capacidad_ajustada']   = inventario_inicial.apply(ajustar_inventario, axis=1)
    inventario_inicial_s = inventario_inicial_s[['criterio','carrier','inventario_inicial','capacidad_ajustada']]
    inventario_inicial   = inventario_inicial[['criterio','carrier','ime_fm','inventario_inicial','capacidad_ajustada']]

    def create_flujos_vs(data):
        """Cuenta movimientos que tocan Patio (fm/to ∈ BLOQUES_PATIO) por (criterio, carrier)."""
        patio_set = set(BLOQUES_PATIO)
        all_criterios = data[['criterio','carrier']].drop_duplicates()
        data_filtered = data[data['ime_move_kind'].isin(['RECV','LOAD','DSCH','DLVR'])].copy()
        mvsp = (
            data_filtered[
                (data_filtered['ime_fm'].isin(patio_set) | data_filtered['ime_to'].isin(patio_set))
            ].groupby(['criterio','carrier']).size().reset_index(name='MvsP')
        )
        return all_criterios.merge(mvsp, on=['criterio','carrier'], how='left').fillna(0)

    flujos_vs = create_flujos_vs(df)

    # ── 10) Escritura de salidas ──
    out_dir = os.path.join(resultados_dir, "instancias_magdalena", f"{semana}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(out_dir, f"analisis_flujos_w{semana}_0.xlsx")
    debug_file  = os.path.join(out_dir, f"analisis_flujos_w{semana}_debug.xlsx")

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            flujos_all_sbt.to_excel(writer, sheet_name='FlujosAll_sbt', index=False)
            flujos_all_st.to_excel(writer, sheet_name='FlujosAll_st', index=False)
            flujos_all_s.to_excel(writer, sheet_name='FlujosAll_s', index=False)
            flujos_sbt.to_excel(writer, sheet_name='Flujos_sbt', index=False)
            flujos_st.to_excel(writer, sheet_name='Flujos_st', index=False)
            flujos_s.to_excel(writer, sheet_name='Flujos_s', index=False)
            flujos_all_sbt_p.to_excel(writer, sheet_name='FlujosAll_sbt_P', index=False)
            flujos_all_sb_p.to_excel(writer, sheet_name='FlujosAll_sb_P', index=False)
            inventario_inicial_s.to_excel(writer, sheet_name='Inventario_Inicial_s', index=False)
            inventario_inicial.to_excel(writer, sheet_name='Inventario_Inicial_sb', index=False)
            flujos_vs.to_excel(writer, sheet_name='Flujos_vs', index=False)
            flujos_168h.to_excel(writer, sheet_name='Flujos_168h', index=False)
            flujos_168h_P.to_excel(writer, sheet_name='Flujos_168h_P', index=False)  # diagnóstico opcional
        print(f"El análisis principal se ha guardado en: {output_file}")
    except PermissionError:
        print(f"No se pudo escribir en {output_file}. Cierra el archivo si está abierto y verifica permisos.")
    except Exception as e:
        print(f"Ocurrió un error al escribir el archivo principal: {str(e)}")

    try:
        if len(debug_rows) > 0:
            with pd.ExcelWriter(debug_file, engine='openpyxl') as writer:
                for sheet, d in debug_rows.items():
                    sname = sheet[:31]
                    if "__rowid__" in d.columns:
                        d.sort_values("__rowid__", inplace=True)
                    d.to_excel(writer, sheet_name=sname, index=False)
            print(f"Depuración guardada en: {debug_file}")
        else:
            print("No hubo filas eliminadas para depuración.")
    except Exception as e:
        print(f"Error al escribir el archivo de depuración: {str(e)}")

