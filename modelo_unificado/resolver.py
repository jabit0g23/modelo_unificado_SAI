"""
Resolver del modelo unificado: ε-constraint Pareto (D, B).

Flujo:
    1) Ancla 1 → minimizar D → (D_min, B@Dmin)
    2) Ancla 2 → minimizar B → (B_min, D@Bmin)
    3) Barrido: para eps ∈ [D_min, D@Bmin · (1+pad)] → minimizar B s.a. D ≤ eps

Si alguna corrida es infactible, escribe LP + IIS.
"""

import logging
import os
import time

from pyomo.environ import SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition as TC

from .config import PARETO_ENABLED, PARETO_POINTS, PARETO_PAD, solver_options_for_horizon, solver_timelimit

logger = logging.getLogger("unificado")


# ─────────────────────────────────────────────────────────────

def _has_incumbent(res, model=None):
    tc = res.solver.termination_condition
    st = res.solver.status
    if tc in (TC.infeasible, TC.invalidProblem, TC.error, TC.solverFailure):
        return False
    if tc in (TC.optimal, TC.feasible):
        return True

    # Timeout (maxTimeLimit / aborted): hay incumbente solo si hay valores de variables
    from pyomo.environ import Var
    try:
        for v in model.component_data_objects(Var, active=True, descend_into=True):
            if v.value is not None:
                return True
    except Exception:
        pass
    return st in (SolverStatus.ok, SolverStatus.warning)


def _solve(model, solver, which: str, use_eps: bool, eps_val=None, *, tee=False):
    model.obj_D.deactivate()
    model.obj_B.deactivate()
    if which == "D":
        model.obj_D.activate()
    elif which == "B":
        model.obj_B.activate()
    else:
        raise ValueError("which must be 'D' or 'B'")

    model.constr_epsD.deactivate()
    if use_eps:
        if eps_val is None:
            raise ValueError("eps_val required when use_eps=True")
        model.eps_D.set_value(float(eps_val))
        model.constr_epsD.activate()

    t0 = time.perf_counter()
    res = solver.solve(model, tee=tee, load_solutions=False)
    # Cargar solución solo si el solver encontró un incumbente; ignorar si no.
    try:
        model.solutions.load_from(res)
    except Exception:
        pass
    return res, time.perf_counter() - t0


def _write_lp_iis(model, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    lp_path  = os.path.join(out_dir, f"modelo_inf_{tag}.lp")
    iis_path = os.path.join(out_dir, f"modelo_inf_{tag}.ilp")
    iis_log  = os.path.join(out_dir, f"iis_log_{tag}.log")
    try:
        model.write(lp_path, format="lp", io_options={"symbolic_solver_labels": True})
        logger.error("[IIS] LP infactible escrito: %s", lp_path)
    except Exception as e:
        logger.error("[IIS] No se pudo escribir LP: %s", e)
        return
    try:
        import gurobipy as gp
        env = gp.Env(iis_log)
        grb = gp.read(lp_path, env)
        grb.Params.LogToConsole = 0
        grb.Params.TimeLimit = 300
        grb.computeIIS()
        grb.write(iis_path)
        logger.error("[IIS] IIS escrito: %s", iis_path)
    except Exception as e:
        logger.error("[IIS] Falló cómputo del IIS: %s", e)


# ─────────────────────────────────────────────────────────────

def resolver_modelo(model, *, semana: str, log_dir: str):
    """Ejecuta el barrido Pareto. Retorna dict con pareto_rows y último res/tiempo."""
    os.makedirs(log_dir, exist_ok=True)
    solver = SolverFactory("gurobi")
    opts = solver_options_for_horizon()
    opts["TimeLimit"] = solver_timelimit()
    opts["LogFile"]   = os.path.join(log_dir, f"gurobi_log_{semana}.log")
    solver.options.update(opts)
    logger.info("[%s] TimeLimit=%d s  Method=%s  NoRelHeurTime=%s  MIPFocus=%s",
                semana, opts["TimeLimit"],
                opts.get("Method", "auto"),
                opts.get("NoRelHeurTime", "off"),
                opts["MIPFocus"])

    pareto_rows = []

    if not PARETO_ENABLED:
        logger.info("[%s] PARETO desactivado → min B directo", semana)
        res, sec = _solve(model, solver, "B", use_eps=False)
        if not _has_incumbent(res, model):
            logger.error("[%s] Sin incumbente (B_only); escribiendo LP+IIS", semana)
            _write_lp_iis(model, log_dir, f"{semana}_B_only")
            return {"pareto_rows": [], "res": res, "solve_seconds": sec, "ok": False}
        D_x = float(value(model.D)); B_x = float(value(model.B_balance))
        logger.info("[%s] Solución única: D=%.3f  B=%.3f  (%.1f s)", semana, D_x, B_x, sec)
        return {
            "pareto_rows": [{"point_type": "single", "which": "B", "eps_D": None,
                             "D_x": D_x, "B_x": B_x, "solve_seconds": sec}],
            "res": res, "solve_seconds": sec, "ok": True,
        }

    # 1) Ancla D_min
    logger.info("[%s] Pareto 1/3: ancla min-D…", semana)
    res_D, sec_D = _solve(model, solver, "D", use_eps=False)
    if not _has_incumbent(res_D, model):
        logger.error("[%s] No hay incumbente para D_min; escribiendo LP+IIS", semana)
        _write_lp_iis(model, log_dir, f"{semana}_Dmin")
        return {"pareto_rows": [], "res": res_D, "solve_seconds": sec_D, "ok": False}
    D_min     = float(value(model.D))
    B_at_Dmin = float(value(model.B_balance))
    pareto_rows.append({"point_type": "anchor", "which": "D_min", "eps_D": None,
                        "D_x": D_min, "B_x": B_at_Dmin, "solve_seconds": sec_D})
    logger.info("[%s] Ancla D_min: D=%.3f  B=%.3f  (%.1f s)", semana, D_min, B_at_Dmin, sec_D)

    # 2) Ancla B_min
    logger.info("[%s] Pareto 2/3: ancla min-B…", semana)
    res_B, sec_B = _solve(model, solver, "B", use_eps=False)
    if not _has_incumbent(res_B, model):
        logger.error("[%s] No hay incumbente para B_min; usando solo ancla D", semana)
        return {"pareto_rows": pareto_rows, "res": res_D, "solve_seconds": sec_D, "ok": True}
    B_min     = float(value(model.B_balance))
    D_at_Bmin = float(value(model.D))
    pareto_rows.append({"point_type": "anchor", "which": "B_min", "eps_D": None,
                        "D_x": D_at_Bmin, "B_x": B_min, "solve_seconds": sec_B})
    logger.info("[%s] Ancla B_min: B=%.3f  D=%.3f  (%.1f s)", semana, B_min, D_at_Bmin, sec_B)

    # 3) Barrido ε
    eps_lo = D_min
    eps_hi = D_at_Bmin * (1.0 + float(PARETO_PAD))
    if PARETO_POINTS < 2:
        eps_grid = [eps_hi]
    else:
        eps_grid = [eps_lo + (eps_hi - eps_lo) * i / (PARETO_POINTS - 1)
                    for i in range(PARETO_POINTS)]

    logger.info("[%s] Pareto 3/3: barrido ε (%d puntos, D∈[%.2f, %.2f])…",
                semana, len(eps_grid), eps_lo, eps_hi)
    last_res, last_sec = res_B, sec_B
    ok_sweep = 0
    for j, eps in enumerate(eps_grid, start=1):
        res_e, sec_e = _solve(model, solver, "B", use_eps=True, eps_val=eps)
        if not _has_incumbent(res_e, model):
            logger.warning("[%s] ε=%d/%d  eps=%.2f  infactible, sigo", semana, j, len(eps_grid), eps)
            continue
        D_x = float(value(model.D)); B_x = float(value(model.B_balance))
        ok_sweep += 1
        logger.info("[%s] ε=%d/%d  D=%.3f  B=%.3f  slack=%.3f  (%.1f s)",
                    semana, j, len(eps_grid), D_x, B_x, eps - D_x, sec_e)
        pareto_rows.append({
            "point_type": "sweep", "which": "B|epsD", "eps_D": float(eps),
            "D_x": D_x, "B_x": B_x,
            "slack_epsD": eps - D_x,
            "solve_seconds": sec_e,
        })
        last_res, last_sec = res_e, sec_e

    logger.info("[%s] Pareto completo: %d puntos totales (%d barrido).",
                semana, len(pareto_rows), ok_sweep)
    return {"pareto_rows": pareto_rows, "res": last_res, "solve_seconds": last_sec, "ok": True}
