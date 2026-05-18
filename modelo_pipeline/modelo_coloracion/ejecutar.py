# =========================================================
# Orquestador principal del modelo de coloración (Magdalena).
# Itera semana por semana: lee instancia → construye modelo
# → resuelve → exporta resultados → registra métricas.
# =========================================================

import logging
import os
import time

import pandas as pd
from pyomo.environ import value

from modelo_pipeline.utils.telemetria import telemetry_pack, append_metrics_row, objective_value_safe

from .construir_modelo import construir_modelo
from .resolver         import resolver_modelo
from .exportar         import exportar_resultados

logger = logging.getLogger("coloracion")

logging.basicConfig(level=logging.INFO)


def ejecutar_instancias_coloracion(
    semanas: list[str],
    resultados_dir: str,
    usar_ki_flujo: bool = True,
    diagnostico_inf: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Resuelve el modelo de coloración para cada semana.

    Retorna:
        semanas_filtradas   : semanas con solución factible
        semanas_infactibles : semanas sin solución
    """
    sufijo_k = "_K" if usar_ki_flujo else ""

    # ── Directorios base ──────────────────────────────────
    resultados_base_path = os.path.join(resultados_dir, "resultados_coloracion")
    os.makedirs(resultados_base_path, exist_ok=True)

    for semana in semanas:
        os.makedirs(os.path.join(resultados_base_path, semana), exist_ok=True)

    metrics_csv = os.path.join(resultados_dir, "metrics", "metrics_coloracion.csv")

    semanas_infactibles: list[str] = []
    print("Iniciando procesamiento de optimización para múltiples semanas...")

    for semana_actual in semanas:
        print(f"\n--- Procesando Semana: {semana_actual} ---")

        directorio_datos_semanal = os.path.join(
            resultados_dir, "instancias_coloracion", semana_actual
        )
        os.makedirs(directorio_datos_semanal, exist_ok=True)

        archivo_instancia = os.path.join(
            directorio_datos_semanal,
            f"Instancia_{semana_actual}{sufijo_k}.xlsx",
        )
        resultado_file = os.path.join(
            resultados_base_path, semana_actual,
            f"resultado_{semana_actual}{sufijo_k}.xlsx",
        )
        resultado_distancias_file = os.path.join(
            resultados_base_path, semana_actual,
            f"Distancias_Modelo_{semana_actual}.xlsx",
        )

        if not os.path.exists(archivo_instancia):
            print(
                f"ADVERTENCIA: Archivo de instancia no encontrado para {semana_actual}: "
                f"{archivo_instancia}. Saltando."
            )
            continue

        try:
            # 1) Leer instancia
            t_build0 = time.perf_counter()
            df = pd.read_excel(archivo_instancia, sheet_name=None)

            # 2) Construir modelo (sets + params + vars + restricciones + objetivo)
            model, ctx = construir_modelo(df)
            t_build1   = time.perf_counter()
            build_seconds = t_build1 - t_build0

            # 3) Resolver
            solver_result = resolver_modelo(
                model,
                semana_actual,
                directorio_datos_semanal,
                resultados_base_path,
                diagnostico_inf=diagnostico_inf,
            )

            if solver_result is None:
                # Infactible o sin incumbente → ya registrado en resolver_modelo
                semanas_infactibles.append(semana_actual)
                continue

            solve_seconds  = solver_result["solve_seconds"]
            gurobi_version = solver_result["gurobi_version"]
            mip_gap        = solver_result["mip_gap"]
            node_count     = solver_result["node_count"]
            threads        = solver_result["threads"]
            res            = solver_result["res"]

            # 4) Exportar resultados
            exportar_resultados(
                model, ctx, semana_actual,
                resultado_file,
                resultado_distancias_file,
            )

            # 5) Métricas
            meta = {
                "modelo":               "coloracion",
                "semana":               semana_actual,
                "fase":                 "final",
                "resultado_xlsx":       resultado_file,
                "resultado_distancias": resultado_distancias_file,
                "build_seconds":        build_seconds,
                "gurobi_version":       gurobi_version,
                "threads":              threads,
                "mip_gap":              mip_gap,
                "node_count":           node_count,
            }

            obj_val = objective_value_safe(model, obj_name="objective")
            row = telemetry_pack(model, meta=meta, solve_elapsed=solve_seconds, res=res, objective=obj_val)
            row.update({
                "build_seconds":  build_seconds,
                "gurobi_version": gurobi_version,
                "threads":        threads,
                "mip_gap":        mip_gap,
                "node_count":     node_count,
            })
            append_metrics_row(metrics_csv, row)

        except Exception as e:
            print(f"Error procesando semana {semana_actual}: {e}")
            continue

    print("\nProceso completado para todas las semanas.")

    semanas_filtradas = [s for s in semanas if s not in semanas_infactibles]

    print("\nsemanas_a_procesar = [")
    for s in semanas_filtradas:
        print(f'    "{s}",')
    print("]")

    return semanas_filtradas, semanas_infactibles
