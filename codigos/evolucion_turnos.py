import os, glob
import pandas as pd
from collections import defaultdict

def criterioII_a_evolucion(semana, input_path, output_file, criterio="criterio_ii"):
    """
    Lee archivos CriterioII-YYYY-MM-DD_{turno}.csv (turnos = 08-00, 15-30, 23-00)
    y genera un Excel "evolucion_turnos_w{semana}.xlsx" con varias hojas:
    - Volumen_Bloques  (per-bloque + total Patio)
    - Visita_Volumen
    - Visita_Flujos
    - Bloques_Seg_Volumen
    - Bloques_Seg_Flujos
    - Bloques_Seg_Flujos2
    - Bloques_Flujos
    """

    # ------------------ BLOQUES DE INTERÉS (UN SOLO PATIO) ------------------
    BLOQUES_PATIO = [
        "C1","C2","C3","C4","C5","C6","C7","C8","C9",
        "H1","H2","H3","H4","H5",
        "T1","T2","T3","T4",
        "I1","I2"
    ]
    turnos_posibles = ["08-00", "15-30", "23-00"]

    # ------------------ LECTURA DE ARCHIVOS ------------------
    
    pref = criterio.strip().lower().replace("_", "")
    # Buscamos todos los CSV y filtramos por prefijo de forma case-insensitive
    all_csv = glob.glob(os.path.join(input_path, "*.csv"))
    all_files = [f for f in all_csv if os.path.basename(f).lower().startswith(pref + "-")]
    if not all_files:
        raise FileNotFoundError(
            f"No se encontraron CSV con prefijo '{pref}-*.csv' (cualquier mayúscula/minúscula) en {input_path}."
        )

    # Acumuladores:
    # - normal: unidades contenedor
    data_sum_normal = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # - multip: TEU, multiplicando 40' x 2
    data_sum_multip = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def extraer_fecha_turno(filename):
        base = os.path.basename(filename)
        sin_ext = os.path.splitext(base)[0]
        partes = sin_ext.split('_')
        if len(partes) < 2:
            return None, None
        first = partes[0]
        if "-" not in first:
            return None, None
        fecha_part = first.split("-", 1)[1]  # YYYY-MM-DD
        turno_part = partes[1]
        return fecha_part, turno_part

    for fpath in all_files:
        fecha_str, turno_str = extraer_fecha_turno(fpath)
        if not fecha_str or not turno_str:
            continue
        if turno_str not in turnos_posibles:
            continue

        df = pd.read_csv(fpath, sep=';', encoding='utf-8')
        # columnas esperadas: Bloque ; Segregacion ; Cantidad de Contenedores
        for _, row in df.iterrows():
            bloque = str(row.get("Bloque", "")).strip()
            seg = str(row.get("Segregacion", "")).strip()
            try:
                cantidad = int(row.get("Cantidad de Contenedores", 0))
            except Exception:
                continue

            if bloque not in BLOQUES_PATIO:
                continue

            # Suma normal (conteo de contenedores)
            data_sum_normal[(fecha_str, turno_str)][seg][bloque] += cantidad

            # Suma multiplicada (40' x2)
            partes_seg = seg.split('-')
            factor = 2 if (len(partes_seg) > 2 and partes_seg[2] == "40") else 1
            data_sum_multip[(fecha_str, turno_str)][seg][bloque] += (cantidad * factor)

    # ------------------ (1) Volumen_Bloques ------------------
    filas_volumen_bloques = []
    all_fecha_turno = sorted(data_sum_multip.keys())

    for (fecha_str, turno_str) in all_fecha_turno:
        # TEU multiplicados por bloque
        sumas = {b: 0 for b in BLOQUES_PATIO}
        dict_seg = data_sum_multip[(fecha_str, turno_str)]
        for seg, bloques_dict in dict_seg.items():
            for b, val in bloques_dict.items():
                sumas[b] += val

        row_bloques = [sumas[b] for b in BLOQUES_PATIO]
        total_patio = sum(row_bloques)  # total sobre todos los bloques del patio

        filas_volumen_bloques.append([fecha_str, turno_str] + row_bloques + [total_patio])

    columns_vol_bloques = ["Fecha","Turno"] + BLOQUES_PATIO + ["Patio"]
    volumen_bloques_df = pd.DataFrame(filas_volumen_bloques, columns=columns_vol_bloques)

    # ------------------ (2) Visita_Volumen ------------------
    all_segregaciones = set()
    for _, seg_dict in data_sum_normal.items():
        for seg, bloques_dict in seg_dict.items():
            if sum(bloques_dict.values()) > 0:
                all_segregaciones.add(seg)
    all_segregaciones = sorted(all_segregaciones)

    ft_sorted = sorted(data_sum_normal.keys())
    col_map = {ft: str(i) for i, ft in enumerate(ft_sorted, start=1)}

    seg_vol_df = pd.DataFrame({"Segregacion": all_segregaciones})
    for ft in ft_sorted:
        col_name = col_map[ft]
        dict_seg = data_sum_normal[ft]
        # suma TODOS los bloques (Patio completo)
        seg2total = {seg: sum(bloques_dict.values()) for seg, bloques_dict in dict_seg.items()}
        seg_vol_df[col_name] = [seg2total.get(s, 0) for s in all_segregaciones]

    col_nums = [col_map[ft] for ft in ft_sorted]
    seg_vol_df["Total"] = seg_vol_df[col_nums].sum(axis=1) if col_nums else 0
    seg_vol_df = seg_vol_df[seg_vol_df["Total"] > 0].copy()
    seg_vol_df.insert(0, "S", [f"S{i}" for i in range(1, len(seg_vol_df)+1)])

    # ------------------ (3) Visita_Flujos ------------------
    visita_flujos_df = seg_vol_df.copy()
    for i in range(len(col_nums)-1):
        c_actual = col_nums[i]
        c_next   = col_nums[i+1]
        visita_flujos_df[c_actual] = seg_vol_df[c_next] - seg_vol_df[c_actual]
    cols_for_sum = col_nums[:-1]
    visita_flujos_df["Total"] = visita_flujos_df[cols_for_sum].abs().sum(axis=1) if cols_for_sum else 0

    # ------------------ (4) Bloques_Seg_Volumen ------------------
    bloques_rows = []
    for _, row in seg_vol_df.iterrows():
        s_id  = row["S"]
        seg   = row["Segregacion"]
        for block in BLOQUES_PATIO:
            vals_periodo = []
            for ft in ft_sorted:
                dict_seg = data_sum_normal[ft]
                val_block = dict_seg.get(seg, {}).get(block, 0)
                vals_periodo.append(val_block)
            total_bloque = sum(vals_periodo)
            bloques_rows.append([s_id, seg, block] + vals_periodo + [total_bloque])

    col_bloques_segVol = ["S","Segregacion","Bloque"] + [col_map[ft] for ft in ft_sorted] + ["Total"]
    bloques_segVol_df = pd.DataFrame(bloques_rows, columns=col_bloques_segVol)

    # ------------------ (5) Bloques_Seg_Flujos ------------------
    bloques_segFlujos_df = bloques_segVol_df.copy()
    col_nums_bloques = [col_map[ft] for ft in ft_sorted]
    for i in range(len(col_nums_bloques)-1):
        c_actual = col_nums_bloques[i]
        c_next   = col_nums_bloques[i+1]
        bloques_segFlujos_df[c_actual] = bloques_segVol_df[c_next] - bloques_segVol_df[c_actual]
    cols_num_bloques = col_nums_bloques[:-1]
    bloques_segFlujos_df["Total"] = bloques_segFlujos_df[cols_num_bloques].abs().sum(axis=1) if cols_num_bloques else 0

    # ------------------ (6) Bloques_Seg_Flujos2 (absoluto por periodo) ------------------
    bloques_segFlujos2_df = bloques_segFlujos_df.copy()
    idx_start = 3  # desde primera columna de periodos
    idx_end   = len(bloques_segFlujos2_df.columns)-1  # antes de Total
    if idx_end > idx_start:
        bloques_segFlujos2_df.iloc[:, idx_start:idx_end] = bloques_segFlujos2_df.iloc[:, idx_start:idx_end].abs()

    # ------------------ (7) Bloques_Flujos (sumario por bloque) ------------------
    bloques_flujos_df = bloques_segFlujos2_df.groupby("Bloque").sum(numeric_only=True).reset_index()
    columnas_bloques_flujos = ["Bloque"] + col_nums_bloques + ["Total"] if col_nums_bloques else ["Bloque","Total"]
    bloques_flujos_df = bloques_flujos_df[columnas_bloques_flujos]

    # ------------------ GUARDAR ------------------
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        volumen_bloques_df.to_excel(writer, sheet_name='Volumen_Bloques', index=False)
        seg_vol_df.to_excel(writer, sheet_name='Visita_Volumen', index=False)
        visita_flujos_df.to_excel(writer, sheet_name='Visita_Flujos', index=False)
        bloques_segVol_df.to_excel(writer, sheet_name='Bloques_Seg_Volumen', index=False)
        bloques_segFlujos_df.to_excel(writer, sheet_name='Bloques_Seg_Flujos', index=False)
        bloques_segFlujos2_df.to_excel(writer, sheet_name='Bloques_Seg_Flujos2', index=False)
        bloques_flujos_df.to_excel(writer, sheet_name='Bloques_Flujos', index=False)

    print(f"¡Listo! Se creó el archivo Excel: {output_file}")

if __name__ == "__main__":
    # Ejemplo de uso:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    estaticos_dir = os.path.join(base_dir, "archivos_estaticos")
    resultados_dir = os.path.join(base_dir, "resultados_generados")

    semana_ejemplo = "2022-08-29"
    salida_excel = os.path.join(resultados_dir, f"evolucion_turnos_w{semana_ejemplo}.xlsx")
    criterioII_a_evolucion(semana_ejemplo, estaticos_dir, salida_excel)
