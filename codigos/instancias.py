import pandas as pd
import numpy as np
import warnings
import math
import os
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

def generar_instancias(semana, resultados_dir, estaticos_dir, participacion_C, cap_mode="pila"):
    
    """
    Genera archivos de instancia (Excel) para el modelo Magdalena con granularidad HORARIA.
    """

    # ───────────── Parámetros base y estáticos ─────────────
    # ### CAMBIO ###: Horizonte de tiempo 1 a 168 (horas)
    T = list(range(1, 25))
    
    SERVICIOS_REGULARES = ['EU', 'MSC', 'MK', 'HAP', 'ACSA', 'CMA']

    B = [
        'C1','C2','C3','C4','C5','C6','C7','C8','C9',
        'H1','H2','H3','H4','H5',
        'T1','T2','T3','T4',
        'I1','I2'
    ]

    ROWS_POR_BLOQUE = {
        'C1':7,'C2':7,'C3':7,'C4':7,'C5':7,'C6':7,'C7':7,'C8':7,'C9':7,
        'H1':7,'H2':7,'H3':6,'H4':7,'H5':7,
        'T1':11,'T2':5,'T3':6,'T4':11,
        'I1':4,'I2':4
    }

    BAHIAS_POR_BLOQUE = {
        'C1': 33, 'C2': 35, 'C3': 40, 'C4': 40, 'C5': 14,
        'C6': 29, 'C7': 29, 'C8': 28, 'C9': 12,
        'H1': 17, 'H2': 25, 'H3': 27, 'H4': 24, 'H5': 20,
        'T1': 31, 'T2': 32, 'T3': 29, 'T4': 28,
        'I1': 13, 'I2': 12
    }

    BAHIAS_REEFER_BLOQUE = {
        'C1': 4, 'C2': 4, 'C3': 4, 'C4': 4, 'C5': 7, 'C6': 0, 'C7': 0, 'C8': 14, 'C9': 6,
        'H1': 0, 'H2': 0, 'H3': 0, 'H4': 12, 'H5': 8,
        'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0,
        'I1': 0, 'I2': 0
    }

    NIVELES_POR_BLOQUE = {b: 5 for b in B}

    # ────────── Capacidades derivadas (bahía/pila) ──────────
    C_POR_BAHIA = {b: ROWS_POR_BLOQUE[b] * NIVELES_POR_BLOQUE[b] for b in B}
    C_REF_BAHIA = int(round(np.mean(list(C_POR_BAHIA.values()))))
    THR_UNIDADES_BAHIA = [4, 8, 12, 17]

    C_POR_PILA = {b: NIVELES_POR_BLOQUE[b] for b in B}
    PILAS_TOTALES_POR_BLOQUE = {b: BAHIAS_POR_BLOQUE[b] * ROWS_POR_BLOQUE[b] for b in B}
    PILAS_REEFER_POR_BLOQUE = {b: BAHIAS_REEFER_BLOQUE[b] * ROWS_POR_BLOQUE[b] for b in B}
    PAIRS_TOTALES_POR_BLOQUE = {b: BAHIAS_POR_BLOQUE[b] * (ROWS_POR_BLOQUE[b] // 2) for b in B}
    PAIRS_REEFER_POR_BLOQUE = {b: BAHIAS_REEFER_BLOQUE[b] * (ROWS_POR_BLOQUE[b] // 2) for b in B}

    # ───────────── Carga de insumos de la semana ─────────────
    ruta_base_semana = os.path.join(resultados_dir, "instancias_magdalena", f"{semana}")
    ruta_analisis = os.path.join(ruta_base_semana, f"analisis_flujos_w{semana}_0.xlsx")

    def _leer_hoja(path, nombre):
        return pd.read_excel(path, sheet_name=nombre)

    df_flujos_all_sb = _leer_hoja(ruta_analisis, 'FlujosAll_sb_P')
    df_flujos_all_sbt = _leer_hoja(ruta_analisis, 'FlujosAll_sbt_P')
    df_flujos_168h = _leer_hoja(ruta_analisis, 'Flujos_168h')

    ruta_evolucion = os.path.join(ruta_base_semana, f"evolucion_turnos_w{semana}.xlsx")
    if Path(ruta_evolucion).is_file():
        df_evolucion_sb = pd.read_excel(ruta_evolucion, sheet_name='Bloques_Seg_Volumen')
    else:
        raise FileNotFoundError(f"No existe {ruta_evolucion}. Genera primero la evolución (criterioII_a_evolucion).")

    df_distancias = pd.read_excel(os.path.join(estaticos_dir, "Distancias_GranPatio.xlsx"), sheet_name='Distancias')

    # ───────────── Utilidades de ajuste y heurísticas ─────────────
    def calcular_ajuste_necesario(I0, flujos, tipo):
        # ### CAMBIO ###: Detectar columna temporal (T o period)
        col_t = 'T' if 'T' in flujos.columns else 'period' if 'period' in flujos.columns else 'shift'
        f = flujos.sort_values(col_t).fillna(0)
        
        if tipo == 'expo':
            delta = f['RECV'] - f['LOAD']
        elif tipo == 'impo':
            delta = f['DSCH'] - f['DLVR']
        else:
            delta = (f['RECV'] + f['DSCH']) - (f['LOAD'] + f['DLVR'])
        
        inv, inv_min = I0, I0
        for d in delta:
            inv += d
            inv_min = min(inv_min, inv)
        return max(0, -inv_min)

    def num_bloques_heur(inventario_total, _cap_unidad_ref_ignored=None):
        unidades_equiv_bahia = math.ceil(inventario_total / C_REF_BAHIA)
        if unidades_equiv_bahia <= THR_UNIDADES_BAHIA[0]:   return 1
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[1]: return 2
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[2]: return 3
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[3]: return 4
        else:                                               return 5

    def redistribuir_bahia(inventario_total, es_reefer, es_40_pies, bahias_oc, bahias_reefer_oc):
        inv_aj = np.zeros(len(B), dtype=int)
        restante = inventario_total
        def libres_tot(b):  return max(0, BAHIAS_POR_BLOQUE[b] - bahias_oc[b])
        def libres_ree(b):  return max(0, BAHIAS_REEFER_BLOQUE[b] - bahias_reefer_oc[b])
        orden = sorted(B, key=lambda x: (libres_ree(x) if es_reefer else libres_tot(x), libres_tot(x)), reverse=True)
        k = min(num_bloques_heur(inventario_total), len(orden))
        usar = orden[:k]
        def asignar(b, q):
            nonlocal restante
            if q <= 0 or restante <= 0: return 0
            cap = C_POR_BAHIA[b]
            factor = 2 if es_40_pies else 1
            unidades = math.ceil(q / cap) * factor
            disp_total_v_teu = libres_tot(b)
            if es_reefer:
                disp_reefer_v = libres_ree(b)
                disp_reefer_v_teu = disp_reefer_v * factor 
                disp = min(disp_total_v_teu, disp_reefer_v_teu)
            else:
                disp = disp_total_v_teu
            if disp <= 0: return 0
            if unidades > disp:
                unidades = disp
                q = (unidades // factor) * cap
            if q <= 0: return 0
            inv_aj[B.index(b)] += q
            bahias_oc[b] += unidades
            if es_reefer: bahias_reefer_oc[b] += (unidades // factor)
            restante -= q
            return q
        for b in usar:
            if restante <= 0: break
            asignar(b, min(restante, inventario_total // k))
        for b in usar:
            if restante <= 0: break
            asignar(b, restante)
        for b in orden[k:]:
            if restante <= 0: break
            asignar(b, restante)
        return inv_aj, bahias_oc, bahias_reefer_oc

    def redistribuir_pila(inventario_total, es_reefer, es_40_pies, pilas_oc, pilas_reefer_oc, pares_oc, pares_reefer_oc, permitir_fallback_40Como20=True):
        # [Implementación original sin cambios]
        inv_aj = np.zeros(len(B), dtype=int)
        restante = int(inventario_total)
        def libres_pilas_tot(b): return max(0, PILAS_TOTALES_POR_BLOQUE[b] - pilas_oc[b])
        def libres_pilas_ree(b): return max(0, PILAS_REEFER_POR_BLOQUE[b] - pilas_reefer_oc[b])
        def libres_pairs_tot(b):
            by_pairs = max(0, PAIRS_TOTALES_POR_BLOQUE[b] - pares_oc[b])
            by_pilas = libres_pilas_tot(b) // 2
            return min(by_pairs, by_pilas)
        def libres_pairs_ree(b):
            by_pairs = max(0, PAIRS_REEFER_POR_BLOQUE[b] - pares_reefer_oc[b])
            by_pilas = libres_pilas_ree(b) // 2
            return min(by_pairs, by_pilas)
        if es_40_pies:
            candidatos = [b for b in B if (libres_pairs_ree(b) if es_reefer else libres_pairs_tot(b)) > 0]
            orden = sorted(candidatos, key=lambda x: (libres_pairs_ree(x), libres_pilas_ree(x)) if es_reefer else (libres_pairs_tot(x), libres_pilas_tot(x)), reverse=True)
        else:
            candidatos = [b for b in B if (libres_pilas_ree(b) if es_reefer else libres_pilas_tot(b)) > 0]
            orden = sorted(candidatos, key=lambda x: (libres_pilas_ree(x), libres_pilas_tot(x)) if es_reefer else libres_pilas_tot(x), reverse=True)
        if not orden:
            return inv_aj, pilas_oc, pilas_reefer_oc, pares_oc, pares_reefer_oc
        k = min(num_bloques_heur(inventario_total), len(orden))
        usar = orden[:k]
        def asignar_40(b, q_cont):
            nonlocal restante
            if q_cont <= 0 or restante <= 0: return 0
            cap_par = NIVELES_POR_BLOQUE[b]
            pares_nec = math.ceil(q_cont / cap_par)
            disp = libres_pairs_ree(b) if es_reefer else libres_pairs_tot(b)
            if disp <= 0: return 0
            if pares_nec > disp:
                pares_nec = disp
                q_cont = pares_nec * cap_par
            if q_cont <= 0: return 0
            idx = B.index(b)
            inv_aj[idx] += q_cont
            pares_oc[b] += pares_nec
            pilas_oc[b] += 2 * pares_nec
            if es_reefer:
                pares_reefer_oc[b] += pares_nec
                pilas_reefer_oc[b] += 2 * pares_nec
            restante -= q_cont
            return q_cont
        def asignar_20(b, q_cont):
            nonlocal restante
            if q_cont <= 0 or restante <= 0: return 0
            cap_pila = C_POR_PILA[b]
            pilas_nec = math.ceil(q_cont / cap_pila)
            disp_tot = libres_pilas_tot(b)
            disp = min(disp_tot, libres_pilas_ree(b)) if es_reefer else disp_tot
            if disp <= 0: return 0
            if pilas_nec > disp:
                pilas_nec = disp
                q_cont = pilas_nec * cap_pila
            if q_cont <= 0: return 0
            idx = B.index(b)
            inv_aj[idx] += q_cont
            pilas_oc[b] += pilas_nec
            if es_reefer:
                pilas_reefer_oc[b] += pilas_nec
            restante -= q_cont
            return q_cont
        if es_40_pies:
            cuota = max(1, inventario_total // max(1, k))
            for b in usar:
                if restante <= 0: break
                asignar_40(b, min(restante, cuota))
            for b in usar:
                if restante <= 0: break
                asignar_40(b, restante)
            for b in orden[k:]:
                if restante <= 0: break
                asignar_40(b, restante)
        if not es_40_pies:
            cuota = max(1, inventario_total // max(1, k))
            for b in usar:
                if restante <= 0: break
                asignar_20(b, min(restante, cuota))
            for b in usar:
                if restante <= 0: break
                asignar_20(b, restante)
            for b in orden[k:]:
                if restante <= 0: break
                asignar_20(b, restante)
        if es_40_pies and permitir_fallback_40Como20 and restante > 0:
            q20_equiv = 2 * restante
            asig20 = 0
            orden20 = sorted([b for b in B if (libres_pilas_ree(b) if es_reefer else libres_pilas_tot(b)) > 0], key=lambda x: (libres_pilas_ree(x), libres_pilas_tot(x)) if es_reefer else libres_pilas_tot(x), reverse=True)
            for b in orden20:
                if q20_equiv <= 0: break
                asig20 += asignar_20(b, q20_equiv)
                q20_equiv -= asig20
            cont40_equiv = asig20 // 2
            restante = max(0, restante - cont40_equiv)
        return inv_aj, pilas_oc, pilas_reefer_oc, pares_oc, pares_reefer_oc

    # ... (analyze_column y analyze_dlvr se mantienen IGUALES) ...
    def analyze_column(df, column_name, location_column, min_value=0):
        df_grouped = df.groupby('criterio')[column_name].sum()
        df_filtered = df_grouped[df_grouped >= min_value]
        def calc_patio_ratio(group):
            patio_sum = group[group[location_column] == 'Patio'][column_name].sum()
            total_sum = group[column_name].sum()
            return patio_sum / total_sum if total_sum != 0 else 0
        patio_values = df[df['criterio'].isin(df_filtered.index)].groupby('criterio').apply(calc_patio_ratio)
        return patio_values[patio_values >= (participacion_C / 100)].index.tolist()

    def analyze_dlvr(df, min_value=0):
        return analyze_column(df, 'DLVR', 'ime_fm', min_value)

    # ... (Funciones de normalidad se mantienen IGUALES) ...
    def obtener_visita(segregacion):
        return segregacion.split('-')[-1] if '-' in segregacion else ''
    def es_servicio_regular(visita):
        return any(visita.startswith(servicio) for servicio in SERVICIOS_REGULARES)
    def determinar_visitas_normales(df_flujos_all_sb):
        tmp = df_flujos_all_sb.copy()
        tmp['visita'] = tmp['criterio'].apply(obtener_visita)
        tmp['es_regular'] = tmp['visita'].apply(es_servicio_regular)
        flujos_por_visita = tmp[tmp['es_regular']].groupby('visita').agg({'RECV': 'sum', 'LOAD': 'sum', 'DSCH': 'sum', 'DLVR': 'sum'})
        if flujos_por_visita.empty: return set()
        flujos_por_visita['total_flujos'] = flujos_por_visita.sum(axis=1)
        visitas_normales = set()
        for servicio in SERVICIOS_REGULARES:
            v = flujos_por_visita[flujos_por_visita.index.str.startswith(servicio)]
            if not v.empty:
                vp = v['total_flujos'].idxmax()
                visitas_normales.add(vp)
                if vp[2:].isdigit():
                    n = int(vp[2:])
                    visitas_normales.add(f"{vp[:2]}{n-1:03d}")
                    visitas_normales.add(f"{vp[:2]}{n+1:03d}")
        return visitas_normales
    def determinar_normalidad(segregacion, df_flujos_all_sb, inventario_inicial, visitas_normales):
        es_expo = segregacion.startswith("expo")
        es_impo = segregacion.startswith("impo")
        visita = obtener_visita(segregacion)
        if not (es_expo or es_impo) or not visita: return "No aplica"
        if not es_servicio_regular(visita): return "Anormal (servicio no regular)"
        flujos_seg = df_flujos_all_sb[df_flujos_all_sb['criterio'] == segregacion]
        if visita in visitas_normales:
            if flujos_seg[['RECV', 'LOAD', 'DSCH', 'DLVR']].sum().sum() > 0 or inventario_inicial > 0:
                return "Normal"
        elif es_expo and flujos_seg['RECV'].sum() > 0: return "Anormal (adelantada)"
        elif es_impo and flujos_seg['DLVR'].sum() > 0: return "Anormal (atrasada)"
        if flujos_seg.empty and inventario_inicial == 0: return "No participa"
        return "Anormal"
    def obtener_razon_detallada(segregacion, load_criterios, dsch_criterios, recv_criterios, dlvr_criterios, df_flujos_all_sb, inventario_inicial_por_bloque, segregaciones_finales):
        # [Implementación original]
        tipo = "Exportación" if segregacion.startswith("expo") else "Importación" if segregacion.startswith("impo") else ""
        flujos_seg = df_flujos_all_sb[df_flujos_all_sb['criterio'] == segregacion]
        criterios = {'LOAD': (load_criterios, 'ime_fm', 15), 'DSCH': (dsch_criterios, 'ime_to', 0), 'RECV': (recv_criterios, 'ime_to', 25), 'DLVR': (dlvr_criterios, 'ime_fm', 35)}
        ok, no_ok = [], []
        for mov, (lista_ok, loc_col, min_val) in criterios.items():
            valores = flujos_seg.groupby(loc_col)[mov].sum()
            total = valores.sum()
            if total > min_val:
                patio_pct = (valores.get('Patio', 0) / total) * 100
                if segregacion in lista_ok: ok.append(mov)
                else: no_ok.append(f"{mov} ({patio_pct:.0f}%)")
            else: no_ok.append(f"{mov} (0)")
        inv_total = inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == segregacion, B].sum().sum() if segregacion in inventario_inicial_por_bloque['Segregacion'].values else 0
        if inv_total > 0: ok.append("Inventario")
        razones = [tipo] if tipo else []
        if ok: razones.append(f"Cumple: {', '.join(ok)}")
        if no_ok: razones.append(f"No cumple: {', '.join(no_ok)}")
        razones.append("Incluida en seg. finales" if segregacion in segregaciones_finales else "No incluida en seg. finales")
        return " | ".join(razones)

    # ───────── Inventario inicial ─────────
    df_evolucion_sb.columns = df_evolucion_sb.columns.astype(str)
    # Aquí seguimos usando columna '1' porque corresponde al inicio de la semana
    inventario_inicial_por_bloque = df_evolucion_sb.pivot_table(
        values='1', index='Segregacion', columns='Bloque', fill_value=0
    ).reset_index()

    # ───────── Construcción de la lista final de segregaciones ─────────
    load_criterios = analyze_column(df_flujos_all_sb,  'LOAD', 'ime_fm', min_value=15)
    dsch_criterios = analyze_column(df_flujos_all_sb,  'DSCH', 'ime_to', min_value=0)
    recv_criterios = analyze_column(df_flujos_all_sb,  'RECV', 'ime_to', min_value=25)
    dlvr_criterios = analyze_dlvr  (df_flujos_all_sb,  min_value=35)

    def unique_preserve(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen: seen.add(x); out.append(x)
        return out

    candidatas = unique_preserve(load_criterios + dsch_criterios + recv_criterios + dlvr_criterios)

    presion = {}
    for seg in candidatas:
        inv_ini = int(inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == seg, B].sum().sum()) if seg in inventario_inicial_por_bloque['Segregacion'].values else 0
        # Aquí no hay problema con usar shift porque sumamos todo el horizonte
        fseg = df_flujos_all_sbt[df_flujos_all_sbt['criterio'] == seg]
        flujo = int(fseg['RECV'].sum() + fseg['DSCH'].sum())
        presion[seg] = inv_ini + flujo

    segregaciones_finales = sorted(candidatas, key=lambda s: (0 if 'reefer' in s.lower() else 1, 0 if '-40-' in s else 1, -presion[s], s))

    # ───────── Reparto final de I0 por bloques ─────────
    inventario_ajustado = {}
    ajustes = {}
    bahias_ocupadas_global = {b: 0 for b in B}
    bahias_reefer_ocupadas_global = {b: 0 for b in B}
    pilas_ocupadas_global = {b: 0 for b in B}
    pilas_reefer_ocupadas_global = {b: 0 for b in B}
    pares_ocupados_global = {b: 0 for b in B}
    pares_reefer_ocupados_global = {b: 0 for b in B}

    for segregacion in segregaciones_finales:
        inv_ini = inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == segregacion, B].values[0].sum() if segregacion in inventario_inicial_por_bloque['Segregacion'].values else 0
        flujos_seg = df_flujos_all_sbt[df_flujos_all_sbt['criterio'] == segregacion]
        tipo = 'expo' if segregacion.startswith('expo') else 'impo' if segregacion.startswith('impo') else 'otro'
        ajuste_nec = calcular_ajuste_necesario(inv_ini, flujos_seg, tipo)
        inv_total = int(round(inv_ini + ajuste_nec))
        es_reefer = 'reefer' in segregacion.lower()
        es_40 = '-40-' in segregacion

        if cap_mode == "bahia":
            inv_aj, bahias_ocupadas_global, bahias_reefer_ocupadas_global = redistribuir_bahia(inv_total, es_reefer, es_40, bahias_ocupadas_global, bahias_reefer_ocupadas_global)
        elif cap_mode == "pila":
            inv_aj, pilas_ocupadas_global, pilas_reefer_ocupadas_global, pares_ocupados_global, pares_reefer_ocupados_global = redistribuir_pila(inv_total, es_reefer, es_40, pilas_ocupadas_global, pilas_reefer_ocupadas_global, pares_ocupados_global, pares_reefer_ocupados_global)
        else:
            raise ValueError("cap_mode debe ser 'bahia' o 'pila'")
        inventario_ajustado[segregacion] = inv_aj
        ajustes[segregacion] = ajuste_nec

    # ───────── Armado de tablas de salida ─────────
    df_B = pd.DataFrame({'B': B})
    df_S = pd.DataFrame({'S': [f'S{i+1}' for i in range(len(segregaciones_finales))], 'Segregacion': segregaciones_finales})
    df_T = pd.DataFrame({'T': T})

    tipo_por_seg = {seg: ('expo' if seg.startswith('expo') else 'impo' if seg.startswith('impo') else 'otro') for seg in df_S['Segregacion']}

    if cap_mode == "bahia":
        df_Cb  = pd.DataFrame({'B': B, 'C': [C_POR_BAHIA[b] for b in B]})
        df_VSb = pd.DataFrame({'B': B, 'VS': [BAHIAS_POR_BLOQUE[b] for b in B]})
        df_VSRb= pd.DataFrame({'B': B, 'VSR':[BAHIAS_REEFER_BLOQUE[b] for b in B]})
    else:
        df_Cb  = pd.DataFrame({'B': B, 'C': [C_POR_PILA[b] for b in B]})
        df_VSb = pd.DataFrame({'B': B, 'VS': [PILAS_TOTALES_POR_BLOQUE[b] for b in B]})
        df_VSRb= pd.DataFrame({'B': B, 'VSR':[PILAS_REEFER_POR_BLOQUE[b] for b in B]})

    df_teu = pd.DataFrame({'S': df_S['S'], 'Segregacion': df_S['Segregacion'], 'TEU': [2 if ('-40' in str(seg)) else 1 for seg in df_S['Segregacion']]})

    data_i0 = []
    for i, seg in enumerate(segregaciones_finales):
        for j, bloque in enumerate(B):
            data_i0.append({'S': df_S.loc[i, 'S'], 'Segregacion': seg, 'B': bloque, 'I0': int(inventario_ajustado[seg][j])})
    df_i0 = pd.DataFrame(data_i0)

    # ### CAMBIO ###: Generación de d_params HORARIA
    # Ahora 'shift' es 'T' (o 'period') y va de 1 a 168.
    d_params_data = []
    
    # Detectamos la columna de tiempo en el dataframe que vino de analisis_flujos
    time_col = 'T' if 'T' in df_flujos_all_sbt.columns else 'period' if 'period' in df_flujos_all_sbt.columns else 'shift'

    for _, srow in df_S.iterrows():
        seg = srow['Segregacion']; s_id = srow['S']
        tipo = tipo_por_seg[seg]
        flujos_seg = df_flujos_all_sbt[df_flujos_all_sbt['criterio'] == seg]
        
        for t in T:
            # Filtramos por el periodo T (1 a 168)
            f = flujos_seg[flujos_seg[time_col] == t]
            
            if tipo == 'expo':
                DR = int(f['RECV'].sum()); DC = int(f['LOAD'].sum())
                DD = 0;                    DE = 0
            elif tipo == 'impo':
                DR = 0;                    DC = 0
                DD = int(f['DSCH'].sum());    DE = int(f['DLVR'].sum())
            else:
                DR = int(f['RECV'].sum());    DC = int(f['LOAD'].sum())
                DD = int(f['DSCH'].sum());    DE = int(f['DLVR'].sum())
            
            d_params_data.append({'S': s_id, 'Segregacion': seg, 'T': t, 'DR': DR, 'DC': DC, 'DD': DD, 'DE': DE})
    
    d_params = pd.DataFrame(d_params_data)

    # d_params_168h se vuelve un poco redundante porque d_params ahora ES 168h, pero lo mantenemos por compatibilidad.
    df_flujos_168h_filtrado = df_flujos_168h[df_flujos_168h['Segregacion'].isin(segregaciones_finales)]
    d_params_h_data = []
    for _, row in df_flujos_168h_filtrado.iterrows():
        seg = row['Segregacion']; s_id = df_S.loc[df_S['Segregacion'] == seg, 'S'].iat[0]
        d_params_h_data.append({
            'S': s_id, 'Segregacion': seg, 'T': int(row['T']),
            'DR': int(row['RECV']), 'DC': int(row['LOAD']),
            'DD': int(row['DSCH']), 'DE': int(row['DLVR'])
        })
    d_params_168h = pd.DataFrame(d_params_h_data)
    
    
    # ───────── DATOS EXTRA PARA MODELO INTEGRADO (CAMILA + MAGDALENA) ─────────

    # 1) Conjuntos de bloques por patio (Costanera, H, T, I)
    B_C = [b for b in B if b.startswith("C")]
    B_H = [b for b in B if b.startswith("H")]
    B_T = [b for b in B if b.startswith("T")]
    B_I = [b for b in B if b.startswith("I")]

    df_BC = pd.DataFrame({"BC": B_C})
    df_BH = pd.DataFrame({"BH": B_H})
    df_BT = pd.DataFrame({"BT": B_T})
    df_BI = pd.DataFrame({"BI": B_I})

    # Por ahora B_E y B_I (expo/impo) = todos los bloques.
    # Si después decides separar bloques de export/import, ajustas aquí solamente.
    df_B_E = pd.DataFrame({"B_E": B})
    df_B_I = pd.DataFrame({"B_I": B})

    # 2) Conjuntos de grúas
    N_RTG = 14
    N_RS  = 11

    G_RTG = [f"rtg{i}" for i in range(1, N_RTG + 1)]
    G_RS  = [f"rs{i}"  for i in range(1, N_RS + 1)]
    G_ALL = G_RTG + G_RS

    df_G   = pd.DataFrame({"G": G_ALL})
    df_GRT = pd.DataFrame({"GRT": G_RTG})
    df_GRS = pd.DataFrame({"GRS": G_RS})

    # 3) Productividades por tipo (movimientos/hora)
    MU_RTG = 30
    MU_RS  = 20
    df_PROD = pd.DataFrame(
        [
            {"Tipo": "RTG", "Prod": MU_RTG},
            {"Tipo": "RS",  "Prod": MU_RS},
        ]
    )

    # 4) Límite de grúas simultáneas por bloque (W) y por tipo (Rmax)
    # En el modelo integrado usamos W en la restricción:
    #  2*#RTG + #RS <= W
    W_global = 6
    df_W = pd.DataFrame({"W": [W_global]})

    df_Wb = pd.DataFrame({"B": B, "W_b": [W_global] * len(B)})  # opcional si quieres por bloque

    df_Rmax_rtg = pd.DataFrame({"Rmax_rtg": [len(G_RTG)]})
    df_Rmax_rs  = pd.DataFrame({"Rmax_rs":  [len(G_RS)]})

    # 5) K_g (permanencia mínima por grúa)
    # Ejemplo: RTG = 2 horas, RS = 1 hora
    df_Kg = pd.DataFrame({
        "G": G_ALL,
        "K": ([2] * len(G_RTG)) + ([1] * len(G_RS))
    })

    # 6) C_bs: capacidad máxima de contenedores por (b, s)
    # Definimos una cota simple:
    #   C_bs = VS_b * C_b
    # donde VS_b está en df_VSb y C_b en df_Cb
    cap_b = df_Cb.set_index("B")["C"].to_dict()
    vs_b  = df_VSb.set_index("B")["VS"].to_dict()

    C_bs_rows = []
    for _, srow in df_S.iterrows():
        s_id = srow["S"]
        for b_id in B:
            C_bs_rows.append({
                "B": b_id,
                "S": s_id,
                "Cbs": int(vs_b[b_id] * cap_b[b_id])
            })
    df_Cbs = pd.DataFrame(C_bs_rows)

    # 7) Compatibilidades y exclusividad entre bloques
    # Matrices completas con 1 por defecto (todo permitido).
    def build_compat_df(blocks, colname):
        rows = [{"b1": b1, "b2": b2, colname: 1} for b1 in blocks for b2 in blocks]
        return pd.DataFrame(rows)

    df_CBR = build_compat_df(B, "CBR")  # RTG
    df_CBS = build_compat_df(B, "CBS")  # RS

    # Exclusividad "legacy" para RTG en Costanera (si quieres mantener la lógica vieja)
    ADYAC_RTG_B = {
        ('C1','C1'), ('C1','C3'),
        ('C2','C2'), ('C2','C4'),
        ('C3','C3'), ('C6','C3'),
        ('C4','C4'), ('C7','C4'),
        ('C5','C5'), ('C5','C8'),
        ('C6','C6'),
        ('C7','C7'),
        ('C8','C8'),
        ('C9','C9'),
    }

    def build_ex_rtg(blocks_costanera, blocks_all, allowed_pairs):
        rows = []
        allowed_sym = allowed_pairs | {(b2, b1) for (b1, b2) in allowed_pairs}
        for b1 in blocks_all:
            for b2 in blocks_all:
                if (b1 in blocks_costanera) and (b2 in blocks_costanera) and ((b1, b2) in allowed_sym):
                    ex = 2
                elif (b1 == b2) and (b1 in blocks_costanera):
                    ex = 2
                else:
                    ex = 1
                rows.append({"b1": b1, "b2": b2, "EX": ex})
        return pd.DataFrame(rows)

    df_EX_RTG = build_ex_rtg(B_C, B, ADYAC_RTG_B)

    

    # ───────── Parámetros K ─────────
    # KI se calcula sobre el flujo TOTAL, así que no cambia
    def calcular_ki(flujo):
        if flujo <= 140:   return 1
        elif flujo <= 280: return 2
        elif flujo <= 420: return 3
        elif flujo <= 595: return 4
        else:              return 5

    df_KS = pd.DataFrame({'S': df_S['S'], 'Segregacion': df_S['Segregacion'], 'KS': [9] * len(df_S)})
    df_KI = pd.DataFrame({'S': df_S['S'], 'Segregacion': df_S['Segregacion'], 'KI': [1] * len(df_S)})
    flujos_por_seg = d_params.groupby('Segregacion')[['DR','DD']].sum()
    flujos_por_seg['Total_Flujo'] = flujos_por_seg['DR'] + flujos_por_seg['DD']
    flujos_por_seg['KI'] = flujos_por_seg['Total_Flujo'].apply(calcular_ki)
    df_KI_K = pd.DataFrame({
        'S': df_S['S'],
        'Segregacion': df_S['Segregacion'],
        'KI': [int(flujos_por_seg.get('KI', {}).get(seg, 1)) for seg in df_S['Segregacion']]
    })

    # ───────── Distancias y banderas ─────────
    df_dist_carga = df_distancias[(df_distancias['ime_fm'].isin(B)) & (df_distancias['ime_to'].isin(['Y-SAI-1','Y-SAI-2']))]
    sitios_carga = df_flujos_all_sbt[df_flujos_all_sbt['LOAD'] > 0].groupby('criterio')['ime_to'].first()

    lc_sb_data = []
    for _, srow in df_S.iterrows():
        seg = srow['Segregacion']; s_id = srow['S']
        sitio = sitios_carga.get(seg)
        for bloque in B:
            if sitio in ['Y-SAI-1','Y-SAI-2']:
                dist = df_dist_carga[(df_dist_carga['ime_fm']==bloque)&(df_dist_carga['ime_to']==sitio)]['Distancia[m]'].values
                val = int(dist[0]) if len(dist)>0 else 0
            else: val = 0
            lc_sb_data.append({'S': s_id, 'Segregacion': seg, 'B': bloque, 'LC': val})
    df_lc_sb = pd.DataFrame(lc_sb_data)

    df_le_b = pd.DataFrame({
        'B': B,
        'LE': [int(df_distancias[(df_distancias['ime_fm']==b)&(df_distancias['ime_to']=='GATE')]['Distancia[m]'].values[0]) if len(df_distancias[(df_distancias['ime_fm']==b)&(df_distancias['ime_to']=='GATE')])>0 else 0 for b in B]
    })

    df_r_s = pd.DataFrame({'S': df_S['S'], 'Segregacion': df_S['Segregacion'], 'R': [1 if 'reefer' in str(seg).lower() else 0 for seg in df_S['Segregacion']]})

    # ───────── Resumen por segregación ─────────
    visitas_normales = determinar_visitas_normales(df_flujos_all_sb)
    data_segs = []
    for seg in segregaciones_finales:
        inv_ini = inventario_inicial_por_bloque[inventario_inicial_por_bloque['Segregacion']==seg][B].sum().sum() if seg in inventario_inicial_por_bloque['Segregacion'].values else 0
        fseg = df_flujos_all_sbt[df_flujos_all_sbt['criterio']==seg]
        recv = int(fseg['RECV'].sum()); load = int(fseg['LOAD'].sum())
        dsch = int(fseg['DSCH'].sum()); dlvr = int(fseg['DLVR'].sum())
        inv_adj = int(sum(inventario_ajustado.get(seg, [0]*len(B))))
        ajuste = int(ajustes.get(seg, 0))
        tipo = tipo_por_seg[seg]
        if tipo == 'expo': inv_fin = max(0, inv_adj + recv - load)
        elif tipo == 'impo': inv_fin = max(0, inv_adj + dsch - dlvr)
        else: inv_fin = max(0, inv_adj + recv + dsch - load - dlvr)

        razon = obtener_razon_detallada(seg, load_criterios, dsch_criterios, recv_criterios, dlvr_criterios, df_flujos_all_sb, inventario_inicial_por_bloque, segregaciones_finales)
        norm = determinar_normalidad(seg, df_flujos_all_sb, inv_adj, visitas_normales)

        data_segs.append({
            'Segregacion': seg, 'En_S': seg in segregaciones_finales, 'Razon': razon, 'Normalidad': norm,
            'Inventario_Inicial': int(inv_ini), 'Ajuste': ajuste, 'Inventario_Ajustado': inv_adj,
            'RECV': recv, 'LOAD': load, 'DSCH': dsch, 'DLVR': dlvr, 'Inventario_Final': int(inv_fin)
        })
    df_segregaciones_detalle = pd.DataFrame(data_segs).sort_values('Segregacion')

    # ───────── Exportación ─────────
    out_dir = os.path.join(resultados_dir, "instancias_magdalena", semana)
    os.makedirs(out_dir, exist_ok=True)
    instancia_path   = os.path.join(out_dir, f"Instancia_{semana}_{participacion_C}.xlsx")
    instancia_k_path = os.path.join(out_dir, f"Instancia_{semana}_{participacion_C}_K.xlsx")

    with pd.ExcelWriter(instancia_path, engine='openpyxl') as w:
        df_B.to_excel(w, sheet_name='B', index=False)
        df_S.to_excel(w, sheet_name='S', index=False)
        df_T.to_excel(w, sheet_name='T', index=False)
        df_Cb.to_excel(w, sheet_name='C_b', index=False)
        df_VSb.to_excel(w, sheet_name='VS_b', index=False)
        df_VSRb.to_excel(w, sheet_name='VSR_b', index=False)
        df_KS.to_excel(w, sheet_name='KS_s', index=False)
        df_KI.to_excel(w, sheet_name='KI_s', index=False)
        df_teu.to_excel(w, sheet_name='TEU_s', index=False)
        d_params.to_excel(w, sheet_name='D_params', index=False)
        df_i0.to_excel(w, sheet_name='I0_sb', index=False)
        df_lc_sb.to_excel(w, sheet_name='LC_sb', index=False)
        df_le_b.to_excel(w, sheet_name='LE_b', index=False)
        df_r_s.to_excel(w, sheet_name='R_s', index=False)
        df_segregaciones_detalle.to_excel(w, sheet_name='Segregaciones_Detalle', index=False)
        d_params_168h.to_excel(w, sheet_name='D_params_168h', index=False)
        
        # ───── Hojas extra para modelo integrado ─────
        df_G.to_excel(w, sheet_name="G", index=False)
        df_GRT.to_excel(w, sheet_name="GRT", index=False)
        df_GRS.to_excel(w, sheet_name="GRS", index=False)
        
        df_BC.to_excel(w, sheet_name="BC", index=False)
        df_BH.to_excel(w, sheet_name="BH", index=False)
        df_BT.to_excel(w, sheet_name="BT", index=False)
        df_BI.to_excel(w, sheet_name="BI", index=False)
        df_B_E.to_excel(w, sheet_name="B_E", index=False)
        df_B_I.to_excel(w, sheet_name="B_I", index=False)

        df_PROD.to_excel(w, sheet_name="PROD", index=False)
        df_W.to_excel(w, sheet_name="W", index=False)
        df_Wb.to_excel(w, sheet_name="W_b", index=False)
        df_Rmax_rtg.to_excel(w, sheet_name="Rmax_rtg", index=False)
        df_Rmax_rs.to_excel(w, sheet_name="Rmax_rs", index=False)
        df_Kg.to_excel(w, sheet_name="K_g", index=False)

        df_Cbs.to_excel(w, sheet_name="C_bs", index=False)
        df_CBR.to_excel(w, sheet_name="CBR", index=False)
        df_CBS.to_excel(w, sheet_name="CBS", index=False)
        df_EX_RTG.to_excel(w, sheet_name="EX_RTG", index=False)

    print(f"\nArchivo Excel {instancia_path} generado/actualizado.")

    with pd.ExcelWriter(instancia_k_path, engine='openpyxl') as w:
        df_B.to_excel(w, sheet_name='B', index=False)
        df_S.to_excel(w, sheet_name='S', index=False)
        df_T.to_excel(w, sheet_name='T', index=False)
        df_Cb.to_excel(w, sheet_name='C_b', index=False)
        df_VSb.to_excel(w, sheet_name='VS_b', index=False)
        df_VSRb.to_excel(w, sheet_name='VSR_b', index=False)
        df_KS.to_excel(w, sheet_name='KS_s', index=False)
        df_KI_K.to_excel(w, sheet_name='KI_s', index=False)
        df_teu.to_excel(w, sheet_name='TEU_s', index=False)
        d_params.to_excel(w, sheet_name='D_params', index=False)
        df_i0.to_excel(w, sheet_name='I0_sb', index=False)
        df_lc_sb.to_excel(w, sheet_name='LC_sb', index=False)
        df_le_b.to_excel(w, sheet_name='LE_b', index=False)
        df_r_s.to_excel(w, sheet_name='R_s', index=False)
        df_segregaciones_detalle.to_excel(w, sheet_name='Segregaciones_Detalle', index=False)
        d_params_168h.to_excel(w, sheet_name='D_params_168h', index=False)
        
        # ───── Hojas extra para modelo integrado ─────
        df_G.to_excel(w, sheet_name="G", index=False)
        df_GRT.to_excel(w, sheet_name="GRT", index=False)
        df_GRS.to_excel(w, sheet_name="GRS", index=False)

        df_BC.to_excel(w, sheet_name="BC", index=False)
        df_BH.to_excel(w, sheet_name="BH", index=False)
        df_BT.to_excel(w, sheet_name="BT", index=False)
        df_BI.to_excel(w, sheet_name="BI", index=False)
        df_B_E.to_excel(w, sheet_name="B_E", index=False)
        df_B_I.to_excel(w, sheet_name="B_I", index=False)

        df_PROD.to_excel(w, sheet_name="PROD", index=False)
        df_W.to_excel(w, sheet_name="W", index=False)
        df_Wb.to_excel(w, sheet_name="W_b", index=False)
        df_Rmax_rtg.to_excel(w, sheet_name="Rmax_rtg", index=False)
        df_Rmax_rs.to_excel(w, sheet_name="Rmax_rs", index=False)
        df_Kg.to_excel(w, sheet_name="K_g", index=False)

        df_Cbs.to_excel(w, sheet_name="C_bs", index=False)
        df_CBR.to_excel(w, sheet_name="CBR", index=False)
        df_CBS.to_excel(w, sheet_name="CBS", index=False)
        df_EX_RTG.to_excel(w, sheet_name="EX_RTG", index=False)

    print(f"Archivo Excel {instancia_k_path} generado/actualizado.\n")