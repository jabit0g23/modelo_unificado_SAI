# =========================================================
# Extracción de resultados del modelo resuelto y escritura
# de los archivos Excel de salida.
# =========================================================

import warnings
import pandas as pd
from pyomo.environ import value


def exportar_resultados(
    model,
    ctx: dict,
    semana: str,
    resultado_file_semana: str,
    resultado_distancias_file_semana: str,
):
    """
    Extrae los valores de las variables del modelo ya resuelto y genera:
        - resultado_{semana}_*.xlsx          (hoja General, flujos, inventario…)
        - Distancias_Modelo_{semana}_*.xlsx  (resumen + por segregación + detalle)

    Parámetros
    ----------
    model                          : ConcreteModel resuelto
    ctx                            : dict devuelto por construir_modelo()
    semana                         : str (fecha ISO)
    resultado_file_semana          : ruta del Excel principal
    resultado_distancias_file_semana : ruta del Excel de resumen de distancias
    """

    segregacion_map = ctx["segregacion_map"]
    bloque_id_map   = ctx["bloque_id_map"]
    seg_id_map      = ctx["seg_id_map"]

    # ── Distancias por segregación ────────────────────────
    distancia_expo = sum(
        value(model.fc[s, b, t]) * value(model.LC[s, b])
        for b in model.B for s in model.S for t in model.T
    )
    distancia_impo = sum(
        value(model.fe[s, b, t]) * value(model.LE[b])
        for b in model.B for s in model.S for t in model.T
    )

    distancia_expo_por_seg = {
        s: sum(value(model.fc[s, b, t]) * value(model.LC[s, b]) for b in model.B for t in model.T)
        for s in model.S
    }
    distancia_impo_por_seg = {
        s: sum(value(model.fe[s, b, t]) * value(model.LE[b]) for b in model.B for t in model.T)
        for s in model.S
    }
    movimientos_dlvr_por_seg = {
        s: sum(value(model.fe[s, b, t]) for b in model.B for t in model.T)
        for s in model.S
    }
    movimientos_load_por_seg = {
        s: sum(value(model.fc[s, b, t]) for b in model.B for t in model.T)
        for s in model.S
    }

    distancia_load_total = sum(distancia_expo_por_seg.values())
    distancia_dlvr_total = sum(distancia_impo_por_seg.values())

    print(
        f"Semana {semana}: {value(model.objective)}, "
        f"{distancia_load_total}, {distancia_dlvr_total}"
    )

    # ── Resúmenes para el Excel de distancias ─────────────
    resumen_semanal = [{
        'Semana':             semana,
        'Distancia Total':    value(model.objective),
        'Distancia LOAD':     distancia_load_total,
        'Distancia DLVR':     distancia_dlvr_total,
        'Movimientos_DLVR':   sum(movimientos_dlvr_por_seg.values()),
        'Movimientos_LOAD':   sum(movimientos_load_por_seg.values()),
    }]

    resultados_segregacion = [
        {
            'Semana':          semana,
            'Segregacion':     segregacion_map[s],
            'Distancia_Total': distancia_expo_por_seg[s] + distancia_impo_por_seg[s],
            'Distancia_DLVR':  distancia_impo_por_seg[s],
            'Distancia_LOAD':  distancia_expo_por_seg[s],
            'Movimientos_DLVR': movimientos_dlvr_por_seg[s],
            'Movimientos_LOAD': movimientos_load_por_seg[s],
        }
        for s in model.S
    ]

    detalle_movimientos = [
        {
            'Semana':           semana,
            'Segregacion':      segregacion_map[s],
            'Bloque':           b,
            'Movimientos DLVR': sum(value(model.fe[s, b, t]) for t in model.T),
            'Movimientos LOAD': sum(value(model.fc[s, b, t]) for t in model.T),
        }
        for s in model.S for b in model.B
        if (
            sum(value(model.fe[s, b, t]) for t in model.T) > 0
            or sum(value(model.fc[s, b, t]) for t in model.T) > 0
        )
    ]

    # ── DataFrames de variables ───────────────────────────
    df_k = pd.DataFrame(
        [(s, model.k[s].value) for s in model.S],
        columns=["Segregación", "Total bloques asignadas"],
    )
    df_w = pd.DataFrame(
        [(b, t, model.w[b, t].value, bloque_id_map[b]) for b in model.B for t in model.T],
        columns=["Bloque", "Periodo", "Carga de trabajo", "BloqueID"],
    )
    df_pq = pd.DataFrame(
        [(j, t, model.p[j, t].value, model.q[j, t].value)
         for j in model.YARDS for t in model.T],
        columns=["Patio", "Periodo", "Carga máxima", "Carga mínima"],
    )
    df_r = pd.DataFrame(
        [(j, model.r[j].value) for j in model.YARDS],
        columns=["Patio", "Variación Carga de trabajo"],
    )

    # ── DataFrame general (todos los campos) ──────────────
    df_gen = pd.DataFrame(
        [
            (s, b, t,
             model.fr[s, b, t].value, model.fc[s, b, t].value,
             model.fd[s, b, t].value, model.fe[s, b, t].value,
             model.y[s, b, t].value,
             model.i[s, b, t].value * model.TEU[s],
             model.v[s, b, t].value * model.TEU[s],
             bloque_id_map[b], seg_id_map[s], model.VS[b])
            for s in model.S for b in model.B for t in model.T
        ],
        columns=[
            "Segregación", "Bloque", "Periodo",
            "Recepción", "Carga", "Descarga", "Entrega",
            "Asignado", "Volumen (TEUs)", "Bahías Ocupadas",
            "BloqueID", "SegregaciónID", "Bahías",
        ],
    )

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df_gen = (
        df_gen.groupby(['Segregación', 'Bloque'])
        .apply(_calcular_incremento_bahias)
        .reset_index(drop=True)
    )

    # ── Capacidades por bloque ────────────────────────────
    df_c_b = pd.DataFrame(
        [
            (s, b, t,
             model.C[b] * model.VS[b] * model.OS.value,
             model.i[s, b, t].value * model.TEU[s],
             sum(model.C[b2] * model.VS[b2] * model.OS.value for b2 in model.B),
             bloque_id_map[b], seg_id_map[s], model.VS[b])
            for s in model.S for b in model.B for t in model.T
        ],
        columns=[
            "Segregación", "Bloque", "Periodo", "Capacidad Bloque",
            "Volumen bloques (TEUs)", "Cap Patio",
            "BloqueID", "SegregaciónID", "Bahías",
        ],
    )

    # ── Contenedores por turno-bloque (pivot) ─────────────
    datos_turno_bloque = [
        {'Turno': t, 'Bloque': b,
         'Contenedores': sum(value(model.i[s, b, t]) for s in model.S)}
        for t in model.T for b in model.B
    ]
    df_pivot_turno_bloque = (
        pd.DataFrame(datos_turno_bloque)
        .pivot(index='Turno', columns='Bloque', values='Contenedores')
        .fillna(0)
    )

    # ── Escritura de archivos ─────────────────────────────
    with pd.ExcelWriter(resultado_file_semana, engine='openpyxl') as writer:
        df_gen.to_excel(writer,              sheet_name="General",                    index=False)
        df_c_b.to_excel(writer,              sheet_name="Ocupación Bloques",          index=False)
        df_k.to_excel(writer,                sheet_name="Total bloques",              index=False)
        df_w.to_excel(writer,                sheet_name="Workload bloques",           index=False)
        df_pq.to_excel(writer,               sheet_name="Carga máx-min",             index=False)
        df_r.to_excel(writer,                sheet_name="Variación Carga de trabajo", index=False)
        df_pivot_turno_bloque.to_excel(writer, sheet_name="Contenedores Turno-Bloque", index=True)
    print(f"Resultados principales para {semana} guardados en {resultado_file_semana}")

    try:
        with pd.ExcelWriter(resultado_distancias_file_semana, engine='openpyxl') as writer:
            pd.DataFrame(resumen_semanal).to_excel(
                writer, sheet_name='Resumen Semanal', index=False
            )
            pd.DataFrame(resultados_segregacion).to_excel(
                writer, sheet_name='Resultados por Segregación', index=False
            )
            pd.DataFrame(detalle_movimientos).to_excel(
                writer, sheet_name='Detalle de Movimientos', index=False
            )
        print(f"Resumen de distancias para {semana} guardado en {resultado_distancias_file_semana}")
    except Exception as e:
        print(f"Error al guardar el archivo de resumen de distancias para {semana}: {e}")

    return {
        "distancia_load_total":    distancia_load_total,
        "distancia_dlvr_total":    distancia_dlvr_total,
        "movimientos_dlvr_total":  sum(movimientos_dlvr_por_seg.values()),
        "movimientos_load_total":  sum(movimientos_load_por_seg.values()),
    }


# ---------------------------------------------------------
# Helper interno
# ---------------------------------------------------------

def _calcular_incremento_bahias(group):
    group = group.sort_values('Periodo')
    group['Incremento Bahías'] = (
        group['Bahías Ocupadas'].diff().fillna(group['Bahías Ocupadas'])
    )
    group['Incremento Bahías'] = group['Incremento Bahías'].apply(lambda x: max(0, x))
    return group
