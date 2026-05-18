import pandas as pd
import numpy as np
import warnings
import math
import os
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)


def generar_instancias(semana, resultados_dir, estaticos_dir, cap_mode, aux_ki,
                       n_turnos: int = 21, umbral_agrupacion: int = 0):

    # ───────────── Parámetros base y estáticos ─────────────
    T = list(range(1, n_turnos + 1))
    SERVICIOS_REGULARES = ['EU', 'MSC', 'MK', 'HAP', 'ACSA', 'CMA']

    B = [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        'H1', 'H2', 'H3', 'H4', 'H5',
        'T1', 'T2', 'T3', 'T4',
        'I1', 'I2'
    ]

    ROWS_POR_BLOQUE = {
        'C1': 7, 'C2': 7, 'C3': 7, 'C4': 7, 'C5': 7, 'C6': 7, 'C7': 7, 'C8': 7, 'C9': 7,
        'H1': 7, 'H2': 7, 'H3': 6, 'H4': 7, 'H5': 7,
        'T1': 11, 'T2': 5, 'T3': 6, 'T4': 11,
        'I1': 4, 'I2': 4
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
    PAIRS_TOTALES_POR_BLOQUE = {b: (BAHIAS_POR_BLOQUE[b] // 2) * ROWS_POR_BLOQUE[b] for b in B}
    UNIDADES_REEFER_POR_BLOQUE = {b: BAHIAS_REEFER_BLOQUE[b] * ROWS_POR_BLOQUE[b] for b in B }

    # ───────────── Carga de insumos de la semana ─────────────
    ruta_base_semana = os.path.join(resultados_dir, "instancias_coloracion", f"{semana}")
    ruta_analisis = os.path.join(ruta_base_semana, f"analisis_flujos_w{semana}_0.xlsx")

    def _leer_hoja(path, nombre):
        return pd.read_excel(path, sheet_name=nombre)

    df_flujos_all_sb = _leer_hoja(ruta_analisis, 'FlujosAll_sb_P')
    df_flujos_all_sbt = _leer_hoja(ruta_analisis, 'FlujosAll_sbt_P')
    df_flujos_168h = _leer_hoja(ruta_analisis, 'Flujos_168h')

    # ✅ FIX CRÍTICO: shift/T siempre numéricos para que calce con t=int del modelo
    if "shift" in df_flujos_all_sbt.columns:
        df_flujos_all_sbt["shift"] = pd.to_numeric(df_flujos_all_sbt["shift"], errors="coerce").fillna(0).astype(int)
    if "T" in df_flujos_168h.columns:
        df_flujos_168h["T"] = pd.to_numeric(df_flujos_168h["T"], errors="coerce").fillna(0).astype(int)

    ruta_evolucion = os.path.join(ruta_base_semana, f"evolucion_turnos_w{semana}.xlsx")
    if Path(ruta_evolucion).is_file():
        df_evolucion_sb = pd.read_excel(ruta_evolucion, sheet_name='Bloques_Seg_Volumen')
    else:
        raise FileNotFoundError(
            f"No existe {ruta_evolucion}. Genera primero la evolución (criterioII_a_evolucion)."
        )

    df_distancias = pd.read_excel(os.path.join(estaticos_dir, "Distancias_GranPatio.xlsx"), sheet_name='Distancias')

    # ───────── Consolidación de segregaciones impo: eliminar "visita" ─────────
    # impo-{tipo}-{tamaño}-{visita}-{DIRECTO|INDIRECTO} → impo-{tipo}-{tamaño}-{DIRECTO|INDIRECTO}
    def _consolidar_impo(name):
        if not isinstance(name, str) or not name.startswith('impo-'):
            return name
        parts = name.split('-')
        if len(parts) < 5:
            return name
        return '-'.join(parts[:3] + parts[4:])

    df_flujos_all_sb['criterio']   = df_flujos_all_sb['criterio'].apply(_consolidar_impo)
    df_flujos_all_sbt['criterio']  = df_flujos_all_sbt['criterio'].apply(_consolidar_impo)
    df_flujos_168h['Segregacion']  = df_flujos_168h['Segregacion'].apply(_consolidar_impo)
    df_evolucion_sb['Segregacion'] = df_evolucion_sb['Segregacion'].apply(_consolidar_impo)

    # df_flujos_168h se itera fila-a-fila aguas abajo: hay que agrupar duplicados.
    flow_cols_168 = [c for c in ('RECV', 'LOAD', 'DSCH', 'DLVR') if c in df_flujos_168h.columns]
    df_flujos_168h = (
        df_flujos_168h.groupby(['Segregacion', 'T'], as_index=False)[flow_cols_168].sum()
    )

    # df_evolucion_sb se pivotea por (Segregacion × Bloque): consolidar antes.
    if 'S' in df_evolucion_sb.columns:
        df_evolucion_sb = df_evolucion_sb.drop(columns=['S'])
    turno_cols = [c for c in df_evolucion_sb.columns if c not in ('Segregacion', 'Bloque')]
    df_evolucion_sb = (
        df_evolucion_sb.groupby(['Segregacion', 'Bloque'], as_index=False)[turno_cols].sum(numeric_only=True)
    )

    # ───────────── Utilidades de ajuste y heurísticas ─────────────
    # Evita inventario negativo a lo largo del horizonte (ajuste mínimo requerido)
    def calcular_ajuste_necesario(I0, flujos, tipo, T=range(1, 22)):
        f = flujos.copy()

        # 1) asegurar shift numérico (y orden real)
        if "shift" in f.columns:
            f["shift"] = pd.to_numeric(f["shift"], errors="coerce").fillna(0).astype(int)
        else:
            # si por alguna razón no existe, no podemos ajustar correctamente
            return 0

        # 2) agregar por turno (el MILP trabaja por turno, no por fila)
        cols = [c for c in ["RECV", "LOAD", "DSCH", "DLVR"] if c in f.columns]
        if not cols:
            return 0

        g = (
            f.groupby("shift")[cols]
            .sum()
            .reindex(list(T), fill_value=0)
            .sort_index()
        )

        if tipo == "expo":
            delta = g.get("RECV", 0) - g.get("LOAD", 0)
        elif tipo == "impo":
            delta = g.get("DSCH", 0) - g.get("DLVR", 0)
        else:
            delta = (g.get("RECV", 0) + g.get("DSCH", 0)) - (g.get("LOAD", 0) + g.get("DLVR", 0))

        inv = int(I0)
        inv_min = inv
        for d in delta.values:
            inv += int(d)
            inv_min = min(inv_min, inv)

        return max(0, -inv_min)

    # Decide cuántos bloques usar para repartir un inventario total (escala por capacidad promedio de bahía)
    def num_bloques_heur(inventario_total, _cap_unidad_ref_ignored=None):
        unidades_equiv_bahia = math.ceil(inventario_total / max(1, C_REF_BAHIA))
        if unidades_equiv_bahia <= THR_UNIDADES_BAHIA[0]:
            return 1
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[1]:
            return 2
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[2]:
            return 3
        elif unidades_equiv_bahia <= THR_UNIDADES_BAHIA[3]:
            return 4
        else:
            return 5
        
        # ───────── Helpers: estimar bahías necesarias/faltantes (modo bahía) ─────────
    def _req_bahias_tot(inv_total, es_40, cap_ref=C_REF_BAHIA):
        # Total bahías equivalentes: 40' consume 2
        factor = 2 if es_40 else 1
        return int(math.ceil(int(inv_total) / max(1, int(cap_ref))) * factor)

    def _req_bahias_reefer(inv_total, cap_ref=C_REF_BAHIA):
        # OJO: esto sigue tu misma "unidad reefer" (BAHIAS_REEFER_BLOQUE y bahias_reefer_oc)
        # En tu lógica, el contador reefer crece con (unidades_bahia // factor),
        # lo que deja el requerimiento reefer ~ ceil(inv_total / cap_ref), independiente de 20/40.
        return int(math.ceil(int(inv_total) / max(1, int(cap_ref))))

    def _faltante_bahias_para_restantes(seg_infos_restantes, bahias_oc, bahias_ree_oc):
        disp_total = sum(max(0, int(BAHIAS_POR_BLOQUE[b]) - int(bahias_oc.get(b, 0))) for b in B)
        disp_ree   = sum(max(0, int(BAHIAS_REEFER_BLOQUE[b]) - int(bahias_ree_oc.get(b, 0))) for b in B)

        req_total = sum(_req_bahias_tot(x["inv_total"], x["es_40"]) for x in seg_infos_restantes)
        req_ree   = sum(_req_bahias_reefer(x["inv_total"]) for x in seg_infos_restantes if x["es_reefer"])

        add_total = max(0, req_total - disp_total)
        add_ree   = max(0, req_ree   - disp_ree)

        return {
            "disp_total": disp_total, "disp_ree": disp_ree,
            "req_total": req_total,   "req_ree": req_ree,
            "add_total": add_total,   "add_ree": add_ree
        }

    # ───────── Reparto de inventarios por modo de capacidad ─────────
    # Modo bahía: asigna por bahías disponibles (con/sin reefer), priorizando bloques con mayor disponibilidad.
    def redistribuir_bahia(inventario_total, es_reefer, es_40_pies, bahias_oc, bahias_reefer_oc):
        inv_aj = np.zeros(len(B), dtype=int)
        restante = int(inventario_total)

        # Disponibilidades residuales por bloque
        def libres_tot(b): return max(0, int(BAHIAS_POR_BLOQUE[b]) - int(bahias_oc[b]))
        def libres_ree(b): return max(0, int(BAHIAS_REEFER_BLOQUE[b]) - int(bahias_reefer_oc[b]))

        # Orden de preferencia: más bahías disponibles primero (reefer o total según corresponda)
        orden = sorted(
            B,
            key=lambda x: (libres_ree(x) if es_reefer else libres_tot(x), libres_tot(x)),
            reverse=True
        )

        # Cuántos bloques intentar usar (cota por heurística y por disponibilidad)
        k = min(num_bloques_heur(inventario_total), len(orden))
        usar = orden[:k]

        def asignar(b, q):
            nonlocal restante
            if q <= 0 or restante <= 0:
                return 0

            cap = int(C_POR_BAHIA[b])  # contenedores por bahía (aprox)
            factor = 2 if es_40_pies else 1  # 40' consume 2 bahías equivalentes
            unidades_bahia = int(math.ceil(q / max(1, cap)) * factor)

            disp_total = libres_tot(b)
            if es_reefer:
                disp_reefer = libres_ree(b) * factor
                disp = min(disp_total, disp_reefer)
            else:
                disp = disp_total

            if disp <= 0:
                return 0

            if unidades_bahia > disp:
                unidades_bahia = disp
                q = (unidades_bahia // factor) * cap

            q = int(q)
            if q <= 0:
                return 0

            inv_aj[B.index(b)] += q
            bahias_oc[b] += unidades_bahia
            if es_reefer:
                bahias_reefer_oc[b] += (unidades_bahia // factor)

            restante -= q
            return q

        # 1 pasada: cuota
        cuota = max(1, inventario_total // max(1, k))
        for b in usar:
            if restante <= 0:
                break
            asignar(b, min(restante, cuota))

        # 2 pasada: remanente en los mismos
        for b in usar:
            if restante <= 0:
                break
            asignar(b, restante)

        # 3 pasada: resto del orden
        for b in orden[k:]:
            if restante <= 0:
                break
            asignar(b, restante)

        return inv_aj, bahias_oc, bahias_reefer_oc, restante

    # Modo pila: asigna por pilas (20’) o pares de pilas (40’), respetando disponibilidad y reefer.
    def redistribuir_pila(
        inventario_total, es_reefer, es_40_pies,
        pilas_oc, pares_oc,
        pilas_reefer20_oc, pares_reefer40_oc
    ):
        inv_aj = np.zeros(len(B), dtype=int)
        restante = int(inventario_total)
    
        def ceil_div(a, b):
            return 0 if a <= 0 else (a + b - 1) // b
    
        def libres_pilas_tot(b):
            return max(0, int(PILAS_TOTALES_POR_BLOQUE[b]) - int(pilas_oc[b]))
    
        def libres_pairs_tot(b):
            by_pairs = max(0, int(PAIRS_TOTALES_POR_BLOQUE[b]) - int(pares_oc[b]))
            by_pilas = libres_pilas_tot(b) // 2
            return min(by_pairs, by_pilas)
    
        # ---- uso de enchufes reefer por tipo ----
        def plugs20_usados(b):
            return ceil_div(int(pilas_reefer20_oc[b]), int(ROWS_POR_BLOQUE[b]))
    
        def plugs40_usados(b):
            return ceil_div(int(pares_reefer40_oc[b]), int(ROWS_POR_BLOQUE[b]))
    
        def plugs_libres(b):
            return max(
                0,
                int(BAHIAS_REEFER_BLOQUE[b]) - plugs20_usados(b) - plugs40_usados(b)
            )
    
        def libres_pilas_ree_20(b):
            rows = int(ROWS_POR_BLOQUE[b])
            usadas = int(pilas_reefer20_oc[b])
            plugs20 = plugs20_usados(b)
            slack_mismo_tipo = plugs20 * rows - usadas
            return slack_mismo_tipo + plugs_libres(b) * rows
    
        def libres_pairs_ree_40(b):
            rows = int(ROWS_POR_BLOQUE[b])
            usados = int(pares_reefer40_oc[b])
            plugs40 = plugs40_usados(b)
            slack_mismo_tipo = plugs40 * rows - usados
            return slack_mismo_tipo + plugs_libres(b) * rows
    
        # ---- orden preferente ----
        if es_40_pies:
            candidatos = [b for b in B if (libres_pairs_ree_40(b) if es_reefer else     libres_pairs_tot(b)) > 0]
            orden = sorted(
                candidatos,
                key=lambda x: (libres_pairs_ree_40(x), libres_pairs_tot(x)) if es_reefer
                else libres_pairs_tot(x),
                reverse=True
            )
        else:
            candidatos = [b for b in B if (libres_pilas_ree_20(b) if es_reefer else     libres_pilas_tot(b)) > 0]
            orden = sorted(
                candidatos,
                key=lambda x: (libres_pilas_ree_20(x), libres_pilas_tot(x)) if es_reefer
                else libres_pilas_tot(x),
                reverse=True
            )
    
        if not orden:
            return inv_aj, pilas_oc, pares_oc, pilas_reefer20_oc, pares_reefer40_oc,     restante
    
        k = min(num_bloques_heur(inventario_total), len(orden))
        usar = orden[:k]
    
        def asignar_20(b, q_cont):
            nonlocal restante
            if q_cont <= 0 or restante <= 0:
                return 0
    
            cap_pila = int(C_POR_PILA[b])   # 5 contenedores por pila
            pilas_nec = int(math.ceil(q_cont / max(1, cap_pila)))
    
            disp = min(libres_pilas_tot(b), libres_pilas_ree_20(b)) if es_reefer else     libres_pilas_tot(b)
            if disp <= 0:
                return 0
    
            if pilas_nec > disp:
                pilas_nec = disp
                q_cont = pilas_nec * cap_pila
    
            q_cont = int(q_cont)
            if q_cont <= 0:
                return 0
    
            idx = B.index(b)
            inv_aj[idx] += q_cont
            pilas_oc[b] += pilas_nec
            if es_reefer:
                pilas_reefer20_oc[b] += pilas_nec
    
            restante -= q_cont
            return q_cont
    
        def asignar_40(b, q_cont):
            nonlocal restante
            if q_cont <= 0 or restante <= 0:
                return 0
    
            cap_par = int(C_POR_PILA[b])    # 5 contenedores por par
            pares_nec = int(math.ceil(q_cont / max(1, cap_par)))
    
            disp = min(libres_pairs_tot(b), libres_pairs_ree_40(b)) if es_reefer else     libres_pairs_tot(b)
            if disp <= 0:
                return 0
    
            if pares_nec > disp:
                pares_nec = disp
                q_cont = pares_nec * cap_par
    
            q_cont = int(q_cont)
            if q_cont <= 0:
                return 0
    
            idx = B.index(b)
            inv_aj[idx] += q_cont
            pares_oc[b] += pares_nec
            pilas_oc[b] += 2 * pares_nec
            if es_reefer:
                pares_reefer40_oc[b] += pares_nec
    
            restante -= q_cont
            return q_cont
    
        cuota = max(1, inventario_total // max(1, k))
    
        if es_40_pies:
            for b in usar:
                if restante <= 0:
                    break
                asignar_40(b, min(restante, cuota))
            for b in usar:
                if restante <= 0:
                    break
                asignar_40(b, restante)
            for b in orden[k:]:
                if restante <= 0:
                    break
                asignar_40(b, restante)
        else:
            for b in usar:
                if restante <= 0:
                    break
                asignar_20(b, min(restante, cuota))
            for b in usar:
                if restante <= 0:
                    break
                asignar_20(b, restante)
            for b in orden[k:]:
                if restante <= 0:
                    break
                asignar_20(b, restante)
    
        return inv_aj, pilas_oc, pares_oc, pilas_reefer20_oc, pares_reefer40_oc, restante
    
    

    # ───────── Selección de segregaciones (criterio Patio + prioridad) ─────────
    def analyze_column(df, column_name, location_column, min_value=0):
        df_grouped = df.groupby('criterio')[column_name].sum()
        df_filtered = df_grouped[df_grouped >= min_value]

        def calc_patio_ratio(group):
            patio_sum = group[group[location_column] == 'Patio'][column_name].sum()
            total_sum = group[column_name].sum()
            return patio_sum / total_sum if total_sum != 0 else 0

        patio_values = df[df['criterio'].isin(df_filtered.index)].groupby('criterio').apply(calc_patio_ratio)
        return patio_values.index.tolist()

    def analyze_dlvr(df, min_value=0):
        return analyze_column(df, 'DLVR', 'ime_fm', min_value)

    # Reglas de “normalidad” por visitas regulares
    def obtener_visita(segregacion):
        return segregacion.split('-')[-1] if '-' in segregacion else ''

    def es_servicio_regular(visita):
        return any(visita.startswith(servicio) for servicio in SERVICIOS_REGULARES)

    def determinar_visitas_normales(df_flujos_all_sb):
        tmp = df_flujos_all_sb.copy()
        tmp['visita'] = tmp['criterio'].apply(obtener_visita)
        tmp['es_regular'] = tmp['visita'].apply(es_servicio_regular)

        flujos_por_visita = tmp[tmp['es_regular']].groupby('visita').agg({
            'RECV': 'sum', 'LOAD': 'sum', 'DSCH': 'sum', 'DLVR': 'sum'
        })
        if flujos_por_visita.empty:
            return set()

        flujos_por_visita['total_flujos'] = flujos_por_visita.sum(axis=1)

        visitas_normales = set()
        for servicio in SERVICIOS_REGULARES:
            v = flujos_por_visita[flujos_por_visita.index.str.startswith(servicio)]
            if not v.empty:
                vp = v['total_flujos'].idxmax()
                visitas_normales.add(vp)
                if vp[2:].isdigit():
                    n = int(vp[2:])
                    visitas_normales.add(f"{vp[:2]}{n - 1:03d}")
                    visitas_normales.add(f"{vp[:2]}{n + 1:03d}")
        return visitas_normales

    def determinar_normalidad(segregacion, df_flujos_all_sb, inventario_inicial, visitas_normales):
        es_expo = segregacion.startswith("expo")
        es_impo = segregacion.startswith("impo")
        visita = obtener_visita(segregacion)
        if not (es_expo or es_impo) or not visita:
            return "No aplica"
        if not es_servicio_regular(visita):
            return "Anormal (servicio no regular)"
        flujos_seg = df_flujos_all_sb[df_flujos_all_sb['criterio'] == segregacion]
        if visita in visitas_normales:
            if flujos_seg[['RECV', 'LOAD', 'DSCH', 'DLVR']].sum().sum() > 0 or inventario_inicial > 0:
                return "Normal"
        elif es_expo and flujos_seg['RECV'].sum() > 0:
            return "Anormal (adelantada)"
        elif es_impo and flujos_seg['DLVR'].sum() > 0:
            return "Anormal (atrasada)"
        if flujos_seg.empty and inventario_inicial == 0:
            return "No participa"
        return "Anormal"

    def obtener_razon_detallada(
        segregacion,
        load_criterios, dsch_criterios, recv_criterios, dlvr_criterios,
        df_flujos_all_sb, inventario_inicial_por_bloque, segregaciones_finales
    ):
        tipo = "Exportación" if segregacion.startswith("expo") else "Importación" if segregacion.startswith("impo") else ""
        flujos_seg = df_flujos_all_sb[df_flujos_all_sb['criterio'] == segregacion]
        criterios = {
            'LOAD': (load_criterios, 'ime_fm', 15),
            'DSCH': (dsch_criterios, 'ime_to', 0),
            'RECV': (recv_criterios, 'ime_to', 25),
            'DLVR': (dlvr_criterios, 'ime_fm', 35)
        }
        ok, no_ok = [], []
        for mov, (lista_ok, loc_col, min_val) in criterios.items():
            valores = flujos_seg.groupby(loc_col)[mov].sum()
            total = valores.sum()
            if total > min_val:
                patio_pct = (valores.get('Patio', 0) / total) * 100
                if segregacion in lista_ok:
                    ok.append(mov)
                else:
                    no_ok.append(f"{mov} ({patio_pct:.0f}%)")
            else:
                no_ok.append(f"{mov} (0)")

        inv_total = (
            inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == segregacion, B]
            .sum().sum()
            if segregacion in inventario_inicial_por_bloque['Segregacion'].values else 0
        )
        if inv_total > 0:
            ok.append("Inventario")

        razones = [tipo] if tipo else []
        if ok:
            razones.append(f"Cumple: {', '.join(ok)}")
        if no_ok:
            razones.append(f"No cumple: {', '.join(no_ok)}")
        razones.append("Incluida en seg. finales" if segregacion in segregaciones_finales else "No incluida en seg. finales")
        return " | ".join(razones)

    # ───────── Inventario inicial (matriz Segregación×Bloque) ─────────
    df_evolucion_sb.columns = df_evolucion_sb.columns.astype(str)
    inventario_inicial_por_bloque = df_evolucion_sb.pivot_table(
        values='1', index='Segregacion', columns='Bloque', fill_value=0
    ).reset_index()

    # ───────── Construcción de la lista final de segregaciones ─────────
    load_criterios = analyze_column(df_flujos_all_sb, 'LOAD', 'ime_fm', min_value=15) #original 15
    dsch_criterios = analyze_column(df_flujos_all_sb, 'DSCH', 'ime_to', min_value=0) #original 0
    recv_criterios = analyze_column(df_flujos_all_sb, 'RECV', 'ime_to', min_value=25) #original 25
    dlvr_criterios = analyze_dlvr(df_flujos_all_sb, min_value=35) #original 35

    def unique_preserve(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    candidatas = unique_preserve(load_criterios + dsch_criterios + recv_criterios + dlvr_criterios)

    presion = {}
    for seg in candidatas:
        inv_ini = int(
            inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == seg, B].sum().sum()
        ) if seg in inventario_inicial_por_bloque['Segregacion'].values else 0

        fseg = df_flujos_all_sbt[df_flujos_all_sbt['criterio'] == seg]
        flujo = int(fseg.get('RECV', 0).sum() + fseg.get('DSCH', 0).sum())
        presion[seg] = inv_ini + flujo

    segregaciones_finales = sorted(
        candidatas,
        key=lambda s: (
            0 if 'reefer' in str(s).lower() else 1,
            0 if '-40-' in str(s) else 1,
            -presion.get(s, 0),
            s
        )
    )

    # ───────── Agrupación de segregaciones de bajo volumen (RUMA) ─────────
    if umbral_agrupacion > 0:
        vol_sbt = df_flujos_all_sbt.groupby('criterio')[['RECV', 'LOAD', 'DSCH', 'DLVR']].sum()
        vol_sbt['total'] = vol_sbt.sum(axis=1)
        inv_tot_s = (
            inventario_inicial_por_bloque.set_index('Segregacion')[B].sum(axis=1)
            if len(inventario_inicial_por_bloque) > 0 else pd.Series(dtype=float)
        )

        def _prefix3(name):
            parts = str(name).split('-')
            return '-'.join(parts[:3]) if len(parts) >= 3 else name

        grandes_segs, grupos_ruma = [], {}
        for seg in segregaciones_finales:
            flujo = int(vol_sbt.loc[seg, 'total']) if seg in vol_sbt.index else 0
            inv = int(inv_tot_s.get(seg, 0))
            if flujo + inv <= umbral_agrupacion:
                grupos_ruma.setdefault(_prefix3(seg), []).append(seg)
            else:
                grandes_segs.append(seg)

        if grupos_ruma:
            nuevas_ruma = []
            for prefix, segs_g in sorted(grupos_ruma.items()):
                nombre_agg = f"{prefix}-RUMA"
                nuevas_ruma.append(nombre_agg)

                # Agregar flows en df_flujos_all_sbt
                mask_sbt = df_flujos_all_sbt['criterio'].isin(segs_g)
                flujos_agg = (
                    df_flujos_all_sbt[mask_sbt]
                    .groupby('shift')[['RECV', 'LOAD', 'DSCH', 'DLVR']]
                    .sum().reset_index()
                )
                flujos_agg['criterio'] = nombre_agg
                flujos_agg['carrier'] = 'RUMA'
                flujos_agg['ime_fm'] = 'RUMA'
                flujos_agg['ime_to'] = 'RUMA'
                df_flujos_all_sbt = pd.concat(
                    [df_flujos_all_sbt[~mask_sbt], flujos_agg], ignore_index=True
                )

                # Agregar flows en df_flujos_168h
                mask_h = df_flujos_168h['Segregacion'].isin(segs_g)
                if mask_h.any():
                    h_agg = (
                        df_flujos_168h[mask_h]
                        .groupby('T')[['RECV', 'LOAD', 'DSCH', 'DLVR']]
                        .sum().reset_index()
                    )
                    h_agg['Segregacion'] = nombre_agg
                    df_flujos_168h = pd.concat(
                        [df_flujos_168h[~mask_h], h_agg], ignore_index=True
                    )

                # Agregar inventario
                mask_inv = inventario_inicial_por_bloque['Segregacion'].isin(segs_g)
                if mask_inv.any():
                    inv_agg = inventario_inicial_por_bloque[mask_inv][B].sum()
                    inv_row = pd.DataFrame([{'Segregacion': nombre_agg, **inv_agg.to_dict()}])
                    inventario_inicial_por_bloque = pd.concat(
                        [inventario_inicial_por_bloque[~mask_inv], inv_row], ignore_index=True
                    )

            nuevas_ruma_sorted = sorted(
                nuevas_ruma,
                key=lambda s: (0 if 'reefer' in s else 1, 0 if '-40-' in s else 1, s)
            )
            segregaciones_finales = grandes_segs + nuevas_ruma_sorted
            print(f"  [RUMA] umbral={umbral_agrupacion}: {len(grandes_segs) + sum(len(v) for v in grupos_ruma.values())} → {len(segregaciones_finales)} segs ({len(grupos_ruma)} grupos RUMA)")

    # ───────── Reparto final de I0 por bloques (según modo) ─────────
    inventario_ajustado = {}
    ajustes = {}

    bahias_ocupadas_global = {b: 0 for b in B}
    bahias_reefer_ocupadas_global = {b: 0 for b in B}
    pilas_ocupadas_global = {b: 0 for b in B}
    pares_ocupados_global = {b: 0 for b in B}
    pilas_reefer20_ocupadas_global = {b: 0 for b in B}
    pares_reefer40_ocupados_global = {b: 0 for b in B}

    # 1) Precalcular info por segregación (para poder estimar "restantes")
    seg_infos = []
    for segregacion in segregaciones_finales:
        inv_ini = (
            inventario_inicial_por_bloque.loc[inventario_inicial_por_bloque['Segregacion'] == segregacion, B]
            .values[0].sum()
            if segregacion in inventario_inicial_por_bloque['Segregacion'].values else 0
        )
        inv_ini = int(inv_ini)

        flujos_seg = df_flujos_all_sbt[df_flujos_all_sbt['criterio'] == segregacion]
        tipo = 'expo' if segregacion.startswith('expo') else 'impo' if segregacion.startswith('impo') else 'otro'

        ajuste_nec = calcular_ajuste_necesario(inv_ini, flujos_seg, tipo, T=T)
        inv_total = int(round(inv_ini + ajuste_nec))

        es_reefer = 'reefer' in str(segregacion).lower()
        es_40 = '-40-' in str(segregacion)

        seg_infos.append({
            "seg": segregacion,
            "tipo": tipo,
            "inv_ini": inv_ini,
            "ajuste": int(ajuste_nec),
            "inv_total": inv_total,
            "es_reefer": es_reefer,
            "es_40": es_40
        })

    # 2) Asignar en orden (y si falla: reportar bahías necesarias para las restantes)
    for idx, info in enumerate(seg_infos):
        segregacion = info["seg"]
        inv_total   = info["inv_total"]
        es_reefer   = info["es_reefer"]
        es_40       = info["es_40"]

        if cap_mode == "bahia":
            # usar copias: si falla, no contaminas la ocupación global
            bahias_oc_tmp = bahias_ocupadas_global.copy()
            bahias_ree_tmp = bahias_reefer_ocupadas_global.copy()

            inv_aj, bahias_oc_tmp, bahias_ree_tmp, restante = redistribuir_bahia(
                inv_total, es_reefer, es_40,
                bahias_oc_tmp, bahias_ree_tmp
            )

        elif cap_mode == "pila":
            pilas_oc_tmp = pilas_ocupadas_global.copy()
            pares_oc_tmp = pares_ocupados_global.copy()
            pilas_ree20_tmp = pilas_reefer20_ocupadas_global.copy()
            pares_ree40_tmp = pares_reefer40_ocupados_global.copy()
            
            inv_aj, pilas_oc_tmp, pares_oc_tmp, pilas_ree20_tmp, pares_ree40_tmp, restante = redistribuir_pila(
                inv_total, es_reefer, es_40,
                pilas_oc_tmp, pares_oc_tmp,
                pilas_ree20_tmp, pares_ree40_tmp
            )
            
        else:
            raise ValueError("cap_mode debe ser 'bahia' o 'pila'")

        inv_asignado = int(inv_aj.sum())
        
        

        if restante > 0 or inv_asignado != inv_total:
            # 👉 Aquí va el reporte “bahías necesarias” para todas las segregaciones restantes
            if cap_mode == "bahia":
                restantes = seg_infos[idx:]  # desde la que falla hasta el final
                falt = _faltante_bahias_para_restantes(
                    restantes,
                    bahias_ocupadas_global,          # ocupación REAL (sin el intento fallido)
                    bahias_reefer_ocupadas_global
                )
                nombres = [x["seg"] for x in restantes]
                sample = ", ".join(nombres[:10]) + (" ..." if len(nombres) > 10 else "")

                raise RuntimeError(
                    f"[ERROR I0] Semana {semana} infactible por falta de bahías en cap_mode=bahia.\n"
                    f"Falla al intentar incluir: {segregacion} | inv_total={inv_total} | inv_asignado={inv_asignado} | restante={restante}\n"
                    f"Segregaciones restantes sin incluir: {len(restantes)} (muestra: {sample})\n"
                    f"Bahías disponibles (total/reefer): {falt['disp_total']} / {falt['disp_ree']}\n"
                    f"Bahías requeridas aprox (total/reefer): {falt['req_total']} / {falt['req_ree']}\n"
                    f"👉 Bahías ADICIONALES necesarias (total/reefer): {falt['add_total']} / {falt['add_ree']}\n"
                    f"Nota: estimación usando capacidad promedio por bahía C_REF_BAHIA={C_REF_BAHIA} y 40' como 2 bahías."
                )
            else:
                # Si también quieres algo similar en pila, se puede hacer, pero tú pediste bahías.
                raise RuntimeError(
                    f"[ERROR I0] {segregacion}: inv_total={inv_total}, inv_asignado={inv_asignado}, restante={restante}. "
                    f"Falta capacidad/heurística en cap_mode={cap_mode}. No exporto instancia inválida."
                )

        # ✅ Commit de ocupación solo si fue exitoso
        if cap_mode == "bahia":
            bahias_ocupadas_global = bahias_oc_tmp
            bahias_reefer_ocupadas_global = bahias_ree_tmp
        else:
            pilas_ocupadas_global = pilas_oc_tmp
            pares_ocupados_global = pares_oc_tmp
            pilas_reefer20_ocupadas_global = pilas_ree20_tmp
            pares_reefer40_ocupados_global = pares_ree40_tmp

        inventario_ajustado[segregacion] = inv_aj
        ajustes[segregacion] = int(info["ajuste"])

    # ───────── Armado de tablas de salida ─────────
    df_MODE = pd.DataFrame({'mode': [cap_mode]})
    df_ROWSb = pd.DataFrame({'B': B, 'ROWS': [ROWS_POR_BLOQUE[b] for b in B]})
    df_Eb = pd.DataFrame({'B': B, 'E': [BAHIAS_REEFER_BLOQUE[b] for b in B]})
    df_VP = pd.DataFrame({'B': B, 'VP': [PAIRS_TOTALES_POR_BLOQUE[b] for b in B]})
    df_B = pd.DataFrame({'B': B})
    df_S = pd.DataFrame({'S': [f'S{i + 1}' for i in range(len(segregaciones_finales))], 'Segregacion': segregaciones_finales})
    df_T = pd.DataFrame({'T': T})

    tipo_por_seg = {
        seg: ('expo' if str(seg).startswith('expo') else 'impo' if str(seg).startswith('impo') else 'otro')
        for seg in df_S['Segregacion']
    }

    if cap_mode == "bahia":
        df_Cb = pd.DataFrame({'B': B, 'C': [C_POR_BAHIA[b] for b in B]})
        df_VSb = pd.DataFrame({'B': B, 'VS': [BAHIAS_POR_BLOQUE[b] for b in B]})
        df_VSRb = pd.DataFrame({'B': B, 'VSR': [BAHIAS_REEFER_BLOQUE[b] for b in B]})
    else:
        df_Cb = pd.DataFrame({'B': B, 'C': [C_POR_PILA[b] for b in B]})
        df_VSb = pd.DataFrame({'B': B, 'VS': [PILAS_TOTALES_POR_BLOQUE[b] for b in B]})
        df_VSRb = pd.DataFrame({'B': B, 'VSR': [UNIDADES_REEFER_POR_BLOQUE[b] for b in B]})

    df_teu = pd.DataFrame({
        'S': df_S['S'],
        'Segregacion': df_S['Segregacion'],
        'TEU': [2 if ('-40' in str(seg)) else 1 for seg in df_S['Segregacion']]
    })

    data_i0 = []
    for i, seg in enumerate(segregaciones_finales):
        for j, bloque in enumerate(B):
            data_i0.append({
                'S': df_S.loc[i, 'S'],
                'Segregacion': seg,
                'B': bloque,
                'I0': int(inventario_ajustado[seg][j])
            })
    df_i0 = pd.DataFrame(data_i0)

    # ✅ Asegura shift numérico para el filtro por t
    tmp_all_sbt = df_flujos_all_sbt.copy()
    tmp_all_sbt["shift"] = pd.to_numeric(tmp_all_sbt["shift"], errors="coerce").fillna(0).astype(int)

    d_params_data = []
    for _, srow in df_S.iterrows():
        seg = srow['Segregacion']
        s_id = srow['S']
        tipo = tipo_por_seg[seg]
        flujos_seg = tmp_all_sbt[tmp_all_sbt['criterio'] == seg]

        for t in T:
            f = flujos_seg[flujos_seg['shift'] == int(t)]
            if tipo == 'expo':
                DR = int(f.get('RECV', 0).sum()); DC = int(f.get('LOAD', 0).sum())
                DD = 0; DE = 0
            elif tipo == 'impo':
                DR = 0; DC = 0
                DD = int(f.get('DSCH', 0).sum()); DE = int(f.get('DLVR', 0).sum())
            else:
                DR = int(f.get('RECV', 0).sum()); DC = int(f.get('LOAD', 0).sum())
                DD = int(f.get('DSCH', 0).sum()); DE = int(f.get('DLVR', 0).sum())

            d_params_data.append({
                'S': s_id, 'Segregacion': seg, 'T': int(t),
                'DR': DR, 'DC': DC, 'DD': DD, 'DE': DE
            })
    d_params = pd.DataFrame(d_params_data)

    df_flujos_168h_filtrado = df_flujos_168h[df_flujos_168h['Segregacion'].isin(segregaciones_finales)]
    d_params_h_data = []
    for _, row in df_flujos_168h_filtrado.iterrows():
        seg = row['Segregacion']
        s_id = df_S.loc[df_S['Segregacion'] == seg, 'S'].iat[0]
        d_params_h_data.append({
            'S': s_id, 'Segregacion': seg, 'T': int(row['T']),
            'DR': int(row['RECV']), 'DC': int(row['LOAD']),
            'DD': int(row['DSCH']), 'DE': int(row['DLVR'])
        })
    d_params_168h = pd.DataFrame(d_params_h_data)

    # ───────── Auditoría de factibilidad por prefijo (mismo input del MILP) ─────────
    def audit_feasibility_by_prefix(df_i0, d_params, df_S, T):
        i0_tot = df_i0.groupby("S")["I0"].sum().to_dict()
        g = (
            d_params.groupby(["S", "T"])[["DR", "DD", "DC", "DE"]]
            .sum()
            .reindex(pd.MultiIndex.from_product([df_S["S"].tolist(), T], names=["S", "T"]), fill_value=0)
            .reset_index()
        )

        bad = []
        for s in df_S["S"]:
            inv = int(i0_tot.get(s, 0))
            inv_min = inv
            for _, row in g[g["S"] == s].sort_values("T").iterrows():
                inv += int(row["DR"] + row["DD"] - row["DC"] - row["DE"])
                inv_min = min(inv_min, inv)
            if inv_min < 0:
                bad.append((s, inv_min, int(i0_tot.get(s, 0))))
        return bad

    bad = audit_feasibility_by_prefix(df_i0, d_params, df_S, T)
    if bad:
        sample = bad[:20]
        msg = "\n".join([f"  {s}: inv_min={inv_min}, I0={i0}" for (s, inv_min, i0) in sample])
        raise RuntimeError(
            "🚨 Instancia infactible por prefijo (I0 + entradas - salidas < 0 en algún turno).\n"
            f"{msg}\n"
            "No exporto instancia inválida."
        )

    # ───────── Parámetros K (KS fijo; KI fijo y KI por flujo) ─────────
    
    #exp=  70 y 210
        
    def calcular_ki(flujo, aux_ki):
        if flujo <= aux_ki:
            return 1
        elif flujo <= 2*aux_ki:
            return 2
        elif flujo <= 3*aux_ki:
            return 3
        elif flujo <= 4*aux_ki:
            return 4
        elif flujo <= 5*aux_ki:
            return 5
        else:
            return 6

    df_KI = pd.DataFrame({'S': df_S['S'], 'Segregacion': df_S['Segregacion'], 'KI': [1] * len(df_S)})

    flujos_por_seg = d_params.groupby('Segregacion')[['DR', 'DD']].sum()
    flujos_por_seg['Total_Flujo'] = flujos_por_seg['DR'] + flujos_por_seg['DD']
    flujos_por_seg['KI'] = flujos_por_seg['Total_Flujo'].apply(lambda x: calcular_ki(x, aux_ki))
    ki_map = flujos_por_seg['KI'].to_dict()

    df_KI_K = pd.DataFrame({
        'S': df_S['S'],
        'Segregacion': df_S['Segregacion'],
        'KI': [int(ki_map.get(seg, 1)) for seg in df_S['Segregacion']]
    })

    # ───────── Distancias y banderas (LC/LE/R) ─────────
    df_dist_carga = df_distancias[
        (df_distancias['ime_fm'].isin(B)) & (df_distancias['ime_to'].isin(['Y-SAI-1', 'Y-SAI-2']))
    ]
    sitios_carga = tmp_all_sbt[tmp_all_sbt['LOAD'] > 0].groupby('criterio')['ime_to'].first()

    lc_sb_data = []
    for _, srow in df_S.iterrows():
        seg = srow['Segregacion']; s_id = srow['S']
        sitio = sitios_carga.get(seg)
        for bloque in B:
            if sitio in ['Y-SAI-1', 'Y-SAI-2']:
                dist = df_dist_carga[
                    (df_dist_carga['ime_fm'] == bloque) & (df_dist_carga['ime_to'] == sitio)
                ]['Distancia[m]'].values
                val = int(dist[0]) if len(dist) > 0 else 0
            else:
                val = 0
            lc_sb_data.append({'S': s_id, 'Segregacion': seg, 'B': bloque, 'LC': val})
    df_lc_sb = pd.DataFrame(lc_sb_data)

    df_le_b = pd.DataFrame({
        'B': B,
        'LE': [
            int(df_distancias[(df_distancias['ime_fm'] == b) & (df_distancias['ime_to'] == 'GATE')]['Distancia[m]'].values[0])
            if len(df_distancias[(df_distancias['ime_fm'] == b) & (df_distancias['ime_to'] == 'GATE')]) > 0 else 0
            for b in B
        ]
    })

    df_r_s = pd.DataFrame({
        'S': df_S['S'],
        'Segregacion': df_S['Segregacion'],
        'R': [1 if 'reefer' in str(seg).lower() else 0 for seg in df_S['Segregacion']]
    })

    # ───────── Resumen por segregación (auditoría) ─────────
    visitas_normales = determinar_visitas_normales(df_flujos_all_sb)
    data_segs = []
    for seg in segregaciones_finales:
        inv_ini = (
            inventario_inicial_por_bloque[inventario_inicial_por_bloque['Segregacion'] == seg][B].sum().sum()
            if seg in inventario_inicial_por_bloque['Segregacion'].values else 0
        )
        fseg = tmp_all_sbt[tmp_all_sbt['criterio'] == seg]
        recv = int(fseg.get('RECV', 0).sum()); load = int(fseg.get('LOAD', 0).sum())
        dsch = int(fseg.get('DSCH', 0).sum()); dlvr = int(fseg.get('DLVR', 0).sum())
        inv_adj = int(sum(inventario_ajustado.get(seg, [0] * len(B))))
        ajuste = int(ajustes.get(seg, 0))
        tipo = tipo_por_seg[seg]

        # NO escondemos negativos: los registramos
        if tipo == 'expo':
            inv_fin_raw = inv_adj + recv - load
        elif tipo == 'impo':
            inv_fin_raw = inv_adj + dsch - dlvr
        else:
            inv_fin_raw = inv_adj + recv + dsch - load - dlvr

        inv_fin_clip = max(0, int(inv_fin_raw))

        razon = obtener_razon_detallada(
            seg, load_criterios, dsch_criterios, recv_criterios, dlvr_criterios,
            df_flujos_all_sb, inventario_inicial_por_bloque, segregaciones_finales
        )
        norm = determinar_normalidad(seg, df_flujos_all_sb, inv_adj, visitas_normales)

        data_segs.append({
            'Segregacion': seg,
            'En_S': seg in segregaciones_finales,
            'Razon': razon,
            'Normalidad': norm,
            'Inventario_Inicial': int(inv_ini),
            'Ajuste': ajuste,
            'Inventario_Ajustado': inv_adj,
            'RECV': recv,
            'LOAD': load,
            'DSCH': dsch,
            'DLVR': dlvr,
            'Inventario_Final_Raw': int(inv_fin_raw),
            'Inventario_Final_Clip': int(inv_fin_clip),
        })

    df_segregaciones_detalle = pd.DataFrame(data_segs).sort_values('Segregacion')

    # ───────── Exportación de archivos Excel ─────────
    out_dir = os.path.join(resultados_dir, "instancias_coloracion", semana)
    os.makedirs(out_dir, exist_ok=True)
    instancia_path = os.path.join(out_dir, f"Instancia_{semana}.xlsx")
    instancia_k_path = os.path.join(out_dir, f"Instancia_{semana}_K.xlsx")
    
    # Versión estándar (KI fijo = 1)
    with pd.ExcelWriter(instancia_path, engine='openpyxl') as w:
        df_VP.to_excel(w, sheet_name='VP_b', index=False)
        df_MODE.to_excel(w, sheet_name='MODE', index=False)
        df_ROWSb.to_excel(w, sheet_name='ROWS_b', index=False)
        df_Eb.to_excel(w, sheet_name='E_b', index=False)
        df_B.to_excel(w, sheet_name='B', index=False)
        df_S.to_excel(w, sheet_name='S', index=False)
        df_T.to_excel(w, sheet_name='T', index=False)
        df_Cb.to_excel(w, sheet_name='C_b', index=False)
        df_VSb.to_excel(w, sheet_name='VS_b', index=False)
        df_VSRb.to_excel(w, sheet_name='VSR_b', index=False)
        df_KI.to_excel(w, sheet_name='KI_s', index=False)
        df_teu.to_excel(w, sheet_name='TEU_s', index=False)
        d_params.to_excel(w, sheet_name='D_params', index=False)
        df_i0.to_excel(w, sheet_name='I0_sb', index=False)
        df_lc_sb.to_excel(w, sheet_name='LC_sb', index=False)
        df_le_b.to_excel(w, sheet_name='LE_b', index=False)
        df_r_s.to_excel(w, sheet_name='R_s', index=False)
        df_segregaciones_detalle.to_excel(w, sheet_name='Segregaciones_Detalle', index=False)
        d_params_168h.to_excel(w, sheet_name='D_params_168h', index=False)
    print(f"\nArchivo Excel {instancia_path} generado/actualizado.")

    # Versión _K (KI calculado por flujo)
    with pd.ExcelWriter(instancia_k_path, engine='openpyxl') as w:
        df_VP.to_excel(w, sheet_name='VP_b', index=False)
        df_MODE.to_excel(w, sheet_name='MODE', index=False)
        df_ROWSb.to_excel(w, sheet_name='ROWS_b', index=False)
        df_Eb.to_excel(w, sheet_name='E_b', index=False)
        df_B.to_excel(w, sheet_name='B', index=False)
        df_S.to_excel(w, sheet_name='S', index=False)
        df_T.to_excel(w, sheet_name='T', index=False)
        df_Cb.to_excel(w, sheet_name='C_b', index=False)
        df_VSb.to_excel(w, sheet_name='VS_b', index=False)
        df_VSRb.to_excel(w, sheet_name='VSR_b', index=False)
        df_KI_K.to_excel(w, sheet_name='KI_s', index=False)
        df_teu.to_excel(w, sheet_name='TEU_s', index=False)
        d_params.to_excel(w, sheet_name='D_params', index=False)
        df_i0.to_excel(w, sheet_name='I0_sb', index=False)
        df_lc_sb.to_excel(w, sheet_name='LC_sb', index=False)
        df_le_b.to_excel(w, sheet_name='LE_b', index=False)
        df_r_s.to_excel(w, sheet_name='R_s', index=False)
        df_segregaciones_detalle.to_excel(w, sheet_name='Segregaciones_Detalle', index=False)
        d_params_168h.to_excel(w, sheet_name='D_params_168h', index=False)
    print(f"Archivo Excel {instancia_k_path} generado/actualizado.\n")