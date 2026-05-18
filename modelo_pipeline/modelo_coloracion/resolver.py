
# =========================================================
# Resolución del modelo: corrida principal + análisis de Pareto.
# Maneja infactibilidad (LP + IIS) y retorna estadísticas.
# =========================================================

import logging
import os
import time

import pandas as pd
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

from . import config as _cfg
from .config import (
    ACTIVAR_R_SWEEP,
    R_SWEEP_VALORES,
    SOLVER_OPTIONS_SWEEP,
    SOLVER_OPTIONS_FINAL,
    IIS_TIME_LIMIT,
)
from modelo_pipeline.utils.gurobi_stats import (
    _gurobi_version_safe,
    _extract_gurobi_stats_from_results,
    _parse_gurobi_log_for_threads_gap_nodes,
    _has_incumbent,
)

logger = logging.getLogger("coloracion")


def resolver_modelo(
    model,
    semana: str,
    directorio_datos_semanal: str,
    resultados_base_path: str,
    diagnostico_inf: bool = True,
) -> dict | None:
    """
    Ejecuta (opcionalmente) el análisis de Pareto y luego la corrida principal.

    Retorna un dict con:
        res, solve_seconds, gurobi_version, mip_gap, node_count, threads
    o None si el modelo resultó infactible o sin incumbente.
    """

    results_dir_semana = os.path.join(resultados_base_path, semana)

    # ── Sweep R por patio (opcional) ──────────────────────
    if ACTIVAR_R_SWEEP:
        _resolver_r_sweep(
            model, semana,
            directorio_datos_semanal,
            resultados_base_path,
        )

    # ── Corrida principal ─────────────────────────────────
    print(f"\n---> Ejecutando optimización principal para la Semana {semana} con R = {_cfg.VALOR_BASE_R}")
    for j, r_val in _cfg.VALOR_BASE_R.items():
        model.r[j].set_value(r_val)
    for j, m_val in _cfg.VALOR_BASE_M.items():
        model.M[j].set_value(m_val)

    solver   = SolverFactory('gurobi')
    log_path = os.path.join(directorio_datos_semanal, f'gurobi_log_{semana}_FINAL.log')

    opts = dict(SOLVER_OPTIONS_FINAL)
    opts['LogFile'] = log_path
    solver.options.update(opts)

    t0 = time.perf_counter()
    res = solver.solve(model, tee=True, load_solutions=True)
    t1 = time.perf_counter()
    solve_seconds = t1 - t0

    # ── Estadísticas del solver ───────────────────────────
    gurobi_version = _gurobi_version_safe()
    stats_res      = _extract_gurobi_stats_from_results(res)
    stats_log      = _parse_gurobi_log_for_threads_gap_nodes(log_path)

    mip_gap    = stats_res["mip_gap"]    if stats_res["mip_gap"]    is not None else stats_log["mip_gap"]
    node_count = stats_res["node_count"] if stats_res["node_count"] is not None else stats_log["node_count"]
    threads    = stats_log["threads"]

    tc = res.solver.termination_condition

    # ── Manejo de infactibilidad ──────────────────────────
    _TC_INFACTIBLE = {
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.invalidProblem,
    }
    if tc in _TC_INFACTIBLE:
        if diagnostico_inf:
            logger.error("🚨 Infactible en %s (tc=%s): escribiendo LP + IIS…", semana, tc)
            _escribir_lp_e_iis(model, semana, results_dir_semana)
        else:
            logger.error("🚨 Infactible en %s (tc=%s). Diagnóstico desactivado.", semana, tc)
        return None  # señal de infactibilidad

    # ── Sin incumbente ────────────────────────────────────
    if not _has_incumbent(res, model):
        logger.error("⛔ %s terminó sin incumbente real (tc=%s).", semana, tc)
        if diagnostico_inf:
            os.makedirs(results_dir_semana, exist_ok=True)
            lp_path = os.path.join(results_dir_semana, f"modelo_{semana}_nolb.lp")
            try:
                model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})
                logger.error("LP guardado en %s", lp_path)
            except Exception as e:
                logger.error("No se pudo escribir el LP: %s", e)
        return None

    logger.info("✅ Semana %s con solución (tc=%s).", semana, res.solver.termination_condition)

    return {
        "res":            res,
        "solve_seconds":  solve_seconds,
        "gurobi_version": gurobi_version,
        "mip_gap":        mip_gap,
        "node_count":     node_count,
        "threads":        threads,
    }


# ---------------------------------------------------------
# Diagnóstico de infactibilidad
# ---------------------------------------------------------

def _escribir_lp_e_iis(model, semana: str, results_dir_semana: str):
    """
    Escribe el LP del modelo infactible y luego computa el IIS.

    Estrategia:
      1. Exporta el modelo a .lp con etiquetas simbólicas.
      2. Carga ese .lp directamente con gurobipy y llama computeIIS()
         → evita re-traducir el modelo Pyomo (más rápido, menos RAM).
      3. Fallback: pyomo.contrib.iis si gurobipy falla.

    Todos los errores se imprimen en consola (además del logger) para
    que sean visibles aunque el logging esté en nivel WARNING.
    """
    os.makedirs(results_dir_semana, exist_ok=True)

    lp_path  = os.path.join(results_dir_semana, f"modelo_inf_{semana}.lp")
    iis_path = os.path.join(results_dir_semana, f"modelo_inf_{semana}.ilp")
    iis_log  = os.path.join(results_dir_semana, f"iis_log_{semana}.log")

    # ── 1) Escribir LP ────────────────────────────────────
    lp_ok = False
    try:
        model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})
        lp_ok = True
        msg = f"[IIS] LP infactible escrito en: {lp_path}"
        print(msg); logger.error(msg)
    except Exception as e:
        msg = f"[IIS] ERROR escribiendo LP: {e}"
        print(msg); logger.error(msg)

    # ── 2) IIS leyendo el .lp con gurobipy directamente ──
    #    (no re-traduce desde Pyomo → mucho más rápido para modelos grandes)
    if lp_ok:
        try:
            _computar_iis_desde_lp(lp_path, iis_path, iis_log)
            return
        except Exception as e:
            msg = f"[IIS] gurobipy directo falló: {e}. Intentando fallback Pyomo…"
            print(msg); logger.error(msg)

    # ── 3) Fallback: pyomo.contrib.iis ───────────────────
    try:
        from pyomo.contrib.iis import write_iis as pyomo_write_iis
        pyomo_write_iis(model, iis_path, solver="gurobi")
        msg = f"[IIS] IIS (fallback pyomo) escrito en: {iis_path}"
        print(msg); logger.error(msg)
    except Exception as e:
        msg = (
            f"[IIS] Fallback pyomo también falló: {e}\n"
            f"      Usa el LP en {lp_path} para diagnóstico manual "
            f"(ej: gurobi_cl {lp_path} y luego computeIIS())."
        )
        print(msg); logger.error(msg)


def _computar_iis_desde_lp(lp_path: str, iis_path: str, iis_log: str):
    """
    Lee el LP ya exportado con gurobipy y computa el IIS.
    Al leer desde archivo se evita re-traducir el modelo Pyomo completo.
    """
    import gurobipy as gp

    env = gp.Env(iis_log)           # log de Gurobi al archivo
    grb = gp.read(lp_path, env)     # carga el LP directamente

    grb.Params.LogToConsole = 0
    grb.Params.TimeLimit    = float(IIS_TIME_LIMIT)

    grb.computeIIS()

    if not iis_path.lower().endswith(".ilp"):
        iis_path = iis_path + ".ilp"
    grb.write(iis_path)

    is_min = bool(getattr(grb, "IISMinimal", 0))
    msg = f"[IIS] IIS escrito en: {iis_path}  (IISMinimal={int(is_min)})"
    print(msg)
    logger.error(msg)


# ---------------------------------------------------------
# Sweep R por patio (interno)
# ---------------------------------------------------------

def _resolver_r_sweep(model, semana, directorio_datos_semanal, resultados_base_path):
    import itertools

    yards  = list(R_SWEEP_VALORES.keys())
    combos = list(itertools.product(*[R_SWEEP_VALORES[j] for j in yards]))
    n      = len(combos)
    print(f"\n---> Iniciando R-Sweep para Semana {semana}: {n} combinaciones")

    solver = SolverFactory('gurobi')
    solver.options.update(SOLVER_OPTIONS_SWEEP)

    filas = []
    for idx, vals in enumerate(combos, 1):
        combo = dict(zip(yards, vals))
        for j, r_val in combo.items():
            model.r[j].set_value(r_val)

        label = "_".join(f"{j}{v}" for j, v in combo.items())
        log_path = os.path.join(directorio_datos_semanal, f'gurobi_log_SWEEP_{semana}_{label}.log')
        solver.options['LogFile'] = log_path

        res = solver.solve(model, tee=False, load_solutions=True)
        status = str(res.solver.termination_condition)

        fila = {'Semana': semana, **{f'R_{j}': v for j, v in combo.items()}}

        if _has_incumbent(res, model):
            distancia = value(model.objective)
            fila['Distancia_Total'] = distancia
            fila['Status'] = status
            # desbalance real por patio: max(p[j,t]-q[j,t]) sobre t
            for j in yards:
                fila[f'Desbalance_max_{j}'] = max(
                    value(model.p[j, t]) - value(model.q[j, t])
                    for t in model.T
                )
            print(f"  [{idx}/{n}] {combo} → dist={distancia:.0f}  ({status})")
        else:
            fila['Distancia_Total'] = None
            fila['Status'] = status
            for j in yards:
                fila[f'Desbalance_max_{j}'] = None
            print(f"  [{idx}/{n}] {combo} → ⛔ sin solución ({status})")

        filas.append(fila)

    out_dir = os.path.join(resultados_base_path, semana)
    os.makedirs(out_dir, exist_ok=True)
    sweep_file = os.path.join(out_dir, f"R_Sweep_{semana}.xlsx")
    pd.DataFrame(filas).to_excel(sweep_file, index=False)
    print(f"---> R-Sweep guardado en: {sweep_file}")
