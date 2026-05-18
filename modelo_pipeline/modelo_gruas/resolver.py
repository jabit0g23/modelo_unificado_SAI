"""
Resolución del modelo de grúas: doble solve (factibilidad → carga) y
manejo de infactibilidad (LP + IIS vía gurobi_persistent).
"""

import os

from pyomo.environ import SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition

from .config import IIS_TIME_LIMIT, solver_options


def _swap_ext(path, new_ext):
    base, _ = os.path.splitext(path)
    return base + new_ext


def _has_solution(model):
    try:
        _ = value(model.obj)
        return True
    except Exception:
        return False


def _dump_infeasible_artifacts(model, xlsx_path, logger=None, iis_time_limit=IIS_TIME_LIMIT, diagnostico_inf=True):
    """
    Si diagnostico_inf=True, escribe el LP e IIS del modelo infactible.
    Si diagnostico_inf=False, no escribe nada y retorna inmediatamente.
    """
    if not diagnostico_inf:
        if logger:
            logger.error("Modelo infactible. Diagnóstico desactivado (DIAGNOSTICO_INF=False).")
        return

    ilp_path = _swap_ext(xlsx_path, ".lp")
    iis_path = _swap_ext(xlsx_path, ".ilp")

    model.write(ilp_path, format="lp", io_options={"symbolic_solver_labels": True})

    iis_solver = None
    try:
        iis_solver = SolverFactory("gurobi_persistent")
        iis_solver.set_instance(model, symbolic_solver_labels=True)
        grb = iis_solver._solver_model
        grb.setParam("TimeLimit", float(iis_time_limit))
        grb.computeIIS()
        grb.write(iis_path)
        if logger:
            try:
                is_min = grb.getAttr("IISMinimal")
                logger.info(f"IIS escrito en {iis_path} | IISMinimal={is_min}")
            except Exception:
                logger.info(f"IIS escrito en {iis_path}")
    except Exception as e:
        if logger:
            logger.warning(f"No se pudo escribir IIS: {e}")
    finally:
        try:
            if iis_solver is not None and hasattr(iis_solver, "close"):
                iis_solver.close()
        except Exception:
            pass

    try:
        if os.path.exists(xlsx_path):
            os.remove(xlsx_path)
    except Exception as e:
        if logger:
            logger.warning(f"No se pudo eliminar {xlsx_path}: {e}")

    if logger:
        logger.error(f"Modelo infactible/aborted sin incumbente. Artefactos: {ilp_path}, {iis_path}")


def resolver_modelo(model, resultado_xlsx, logger=None, diagnostico_inf=True):
    """
    Ejecuta dos solves:
      1) Factibilidad sin cargar solución (detecta infactibilidad limpia).
      2) Carga de solución; si aborta sin incumbente, trata como infactible.

    Retorna:
        dict con 'res', 'res2', 'elapsed', 'feasible' (bool).
        Si infactible: feasible=False y los artefactos LP+IIS ya quedaron escritos.
    """
    import time

    solver = SolverFactory("gurobi")
    solver.options.update(solver_options())

    # ── 1) Factibilidad ───────────────────────────────────────
    t0 = time.time()
    res = solver.solve(model, tee=False, load_solutions=False)
    t1 = time.time()

    term = res.solver.termination_condition
    status = res.solver.status
    if term in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
        _dump_infeasible_artifacts(model, resultado_xlsx, logger, diagnostico_inf=diagnostico_inf)
        return {"res": res, "res2": None, "elapsed": t1 - t0, "feasible": False, "fase": "preload"}

    # Timeout sin incumbente: no tiene sentido hacer un segundo solve
    if term == TerminationCondition.maxTimeLimit and not _has_solution(model):
        if logger:
            logger.error("Timeout sin incumbente en primer solve — tratando como infactible.")
        _dump_infeasible_artifacts(model, resultado_xlsx, logger, diagnostico_inf=diagnostico_inf)
        return {"res": res, "res2": None, "elapsed": t1 - t0, "feasible": False, "fase": "timeout_no_incumbent"}

    # ── 2) Cargar solución ───────────────────────────────────
    t2 = time.time()
    res2 = solver.solve(model, tee=False, load_solutions=True)
    t3 = time.time()
    term2 = res2.solver.termination_condition
    status2 = res2.solver.status

    aborted = (
        term2 in (TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations)
        or status2 in (SolverStatus.aborted, SolverStatus.unknown)
    )
    if aborted and not _has_solution(model):
        _dump_infeasible_artifacts(model, resultado_xlsx, logger, diagnostico_inf=diagnostico_inf)
        return {"res": res, "res2": res2, "elapsed": (t1 - t0) + (t3 - t2),
                "feasible": False, "fase": "load_fail"}

    return {"res": res, "res2": res2, "elapsed": (t1 - t0) + (t3 - t2),
            "feasible": True, "fase": "final"}
