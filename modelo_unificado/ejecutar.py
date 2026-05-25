"""Orquestador semanal del modelo unificado (lee Instancia_*.xlsx por semana)."""

import logging
import os
import time

import pandas as pd

from .construir_modelo import construir_modelo
from .resolver         import resolver_modelo
from .exportar         import exportar_resultados
from .utils.telemetria import telemetry_pack, append_metrics_row, objective_value_safe

logger = logging.getLogger("unificado")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def ejecutar_instancias_unificado(
    semanas: list[str],
    participacion,
    resultados_dir: str,
    *,
    usar_ki_flujo: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Resuelve el modelo unificado (ε-constraint Pareto) para cada semana.

    Lee: {resultados_dir}/instancias_unificadas/{semana}/Instancia_{semana}_{p}_K.xlsx
    Escribe: {resultados_dir}/resultados_unificados/{semana}/
    """
    sufijo_k = "_K" if usar_ki_flujo else ""
    instancias_base = os.path.join(resultados_dir, "instancias_unificadas")
    resultados_base = os.path.join(resultados_dir, "resultados_unificados")
    os.makedirs(resultados_base, exist_ok=True)

    metrics_csv = os.path.join(resultados_dir, "metrics", "metrics_unificado.csv")

    semanas_infactibles: list[str] = []
    print("Iniciando modelo unificado (Magdalena + Camila, horario)...")

    for semana in semanas:
        print(f"\n--- Semana {semana} ---")
        sem_dir_inst = os.path.join(instancias_base, semana)
        sem_dir_res  = os.path.join(resultados_base, semana)
        os.makedirs(sem_dir_res, exist_ok=True)

        archivo = os.path.join(sem_dir_inst, f"Instancia_{semana}_{participacion}{sufijo_k}.xlsx")
        if not os.path.exists(archivo):
            print(f"ADVERTENCIA: no se encontró {archivo}. Salto semana.")
            continue

        resultado_xlsx = os.path.join(sem_dir_res, f"resultado_{semana}_{participacion}_Unificado.xlsx")
        pareto_csv     = os.path.join(sem_dir_res, f"pareto_{semana}_{participacion}.csv")

        try:
            logger.info("[%s] Leyendo instancia: %s", semana, os.path.basename(archivo))
            t0 = time.perf_counter()
            df = pd.read_excel(archivo, sheet_name=None)
            read_sec = time.perf_counter() - t0
            logger.info("[%s] Hojas leídas: %s  (%.1f s)", semana, list(df.keys()), read_sec)

            logger.info("[%s] Construyendo modelo…", semana)
            t1 = time.perf_counter()
            model, ctx = construir_modelo(df)
            build_sec = time.perf_counter() - t1
            logger.info("[%s] Modelo listo en %.1f s  (modo=%s  horas=%d)",
                        semana, build_sec, "pila" if ctx["es_pila"] else "bahia", ctx["horas"])

            logger.info("[%s] Resolviendo (Pareto ε-constraint)…", semana)
            out = resolver_modelo(model, semana=semana, log_dir=sem_dir_res)
            if not out["ok"]:
                logger.error("[%s] Infactible; LP/IIS en %s", semana, sem_dir_res)
                semanas_infactibles.append(semana)
                continue

            logger.info("[%s] Exportando resultados…", semana)
            exportar_resultados(
                model, ctx,
                semana=semana, participacion=participacion,
                resultado_xlsx=resultado_xlsx,
                pareto_csv=pareto_csv,
                pareto_rows=out["pareto_rows"],
                chosen_point=out.get("chosen_point"),
                balance_valid=out.get("balance_valid", True),
            )

            meta = {
                "modelo": "unificado", "semana": semana,
                "participacion": str(participacion), "horas": ctx["horas"],
                "build_seconds": build_sec,
            }
            obj_val = (objective_value_safe(model, obj_name="obj_B")
                       if out.get("balance_valid", True) else None)
            row = telemetry_pack(model, meta=meta,
                                 solve_elapsed=out["solve_seconds"],
                                 res=out["res"], objective=obj_val)
            append_metrics_row(metrics_csv, row)
            logger.info("[%s] ✓ Semana completada  (build=%.1f s  solve=%.1f s)",
                        semana, build_sec, out["solve_seconds"])
        except Exception as e:
            logger.exception("Error en semana %s", semana)
            semanas_infactibles.append(semana)
            continue

    semanas_ok = [s for s in semanas if s not in semanas_infactibles]
    print(f"\nOK={len(semanas_ok)}  Infactibles={len(semanas_infactibles)}")
    return semanas_ok, semanas_infactibles
