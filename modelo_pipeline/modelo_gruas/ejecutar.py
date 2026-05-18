"""
Orquestador del modelo de grúas: itera semanas × turnos, construye el modelo,
lo resuelve y registra métricas.
"""

import logging
import os

import pandas as pd

from modelo_pipeline.utils.telemetria import telemetry_pack, append_metrics_row, objective_value_safe

from .construir_modelo import construir_modelo
from .exportar import exportar_resultados
from .resolver import resolver_modelo

logger = logging.getLogger("gruas")


def ejecutar_instancias_gruas(semanas, turnos, base_instancias, base_resultados, diagnostico_inf=True):
    """
    Resuelve el modelo de grúas para cada combinación (semana, turno).
    Escribe un Excel por turno y acumula telemetría en metrics_gruas.csv.
    """
    metrics_csv = os.path.join(base_resultados, "metrics", "metrics_gruas.csv")
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

    for semana in semanas:
        out_dir = os.path.join(base_resultados, f"resultados_turno_{semana}")
        os.makedirs(out_dir, exist_ok=True)

        for turno in turnos:
            t2 = str(turno).zfill(2)
            logger.info(f"--- INICIANDO TURNO {turno} / SEMANA {semana} ---")

            instancia_path = os.path.join(
                base_instancias, f"instancias_turno_{semana}",
                f"Instancia_{semana}_T{t2}.xlsx",
            )
            datos = pd.read_excel(instancia_path, sheet_name=None)

            m = construir_modelo(datos)

            resultado_xlsx = os.path.join(
                out_dir, f"resultados_{semana}_T{t2}.xlsx"
            )

            out = resolver_modelo(m, resultado_xlsx, logger=logger, diagnostico_inf=diagnostico_inf)
            fase = out["fase"]
            res_for_log = out["res2"] if out["res2"] is not None else out["res"]

            meta = {
                "modelo":         "gruas",
                "semana":         semana,
                "turno":          int(turno),
                "fase":           fase,
                "resultado_xlsx": resultado_xlsx,
                "resultado_dir":  out_dir,
            }

            if not out["feasible"]:
                row = telemetry_pack(m, meta=meta, solve_elapsed=out["elapsed"],
                                     res=res_for_log, objective=None)
                append_metrics_row(metrics_csv, row)
                logger.info("Turno %s completado.", turno)
                continue

            exportar_resultados(m, resultado_xlsx)
            logger.info("Turno %s completado.", turno)

            obj_val = objective_value_safe(m, obj_name="obj")
            row = telemetry_pack(m, meta=meta, solve_elapsed=out["elapsed"],
                                 res=res_for_log, objective=obj_val)
            append_metrics_row(metrics_csv, row)
