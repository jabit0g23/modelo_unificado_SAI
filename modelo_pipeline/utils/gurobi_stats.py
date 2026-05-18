# =========================================================
# Helpers de telemetría: versión Gurobi, stats del log,
# detección de incumbente, escritura de IIS.
# =========================================================

import os
import re
import subprocess
from typing import Any, Dict, Optional

from pyomo.environ import Var, SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition as TC


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _gurobi_version_safe() -> Optional[str]:
    try:
        import gurobipy as gp
        v = gp.gurobi.version()   # (major, minor, technical)
        return ".".join(map(str, v))
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["gurobi_cl", "--version"], stderr=subprocess.STDOUT, text=True
        )
        m = re.search(r"version\s+(\d+\.\d+\.\d+)", out, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        line0 = out.strip().splitlines()[0].strip()
        return line0[:100]
    except Exception:
        return None


def _extract_gurobi_stats_from_results(res) -> Dict[str, Any]:
    mip_gap    = None
    node_count = None

    try:
        stats = getattr(res.solver, "statistics", None)
        if stats is not None:
            bb = getattr(stats, "branch_and_bound", None)
            if bb is not None:
                node_count = (
                    getattr(bb, "number_of_created_nodes", None)
                    or getattr(bb, "number_of_nodes", None)
                )
            mip_gap = getattr(stats, "mip_gap", None) or getattr(stats, "gap", None)
    except Exception:
        pass

    try:
        mip_gap = (
            mip_gap
            or getattr(res.solver, "mip_gap", None)
            or getattr(res.solver, "gap",      None)
        )
    except Exception:
        pass

    return {
        "mip_gap":    _safe_float(mip_gap),
        "node_count": None if node_count is None else int(node_count),
    }


def _parse_gurobi_log_for_threads_gap_nodes(log_path: str) -> Dict[str, Any]:
    out = {"threads": None, "mip_gap": None, "node_count": None}

    if not log_path or not os.path.exists(log_path):
        return out

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return out

    # threads
    for ln in lines[:300]:
        m = re.search(r"using up to\s+(\d+)\s+threads", ln, flags=re.IGNORECASE)
        if m:
            out["threads"] = int(m.group(1))
            break

    # node_count
    for ln in reversed(lines[-500:]):
        m = re.search(r"Explored\s+(\d+)\s+nodes", ln, flags=re.IGNORECASE)
        if m:
            out["node_count"] = int(m.group(1))
            break

    # mip_gap
    gap_candidates = []
    for ln in reversed(lines[-800:]):
        m = re.search(r"\bgap\s+([0-9]+(?:\.[0-9]+)?)\s*%", ln, flags=re.IGNORECASE)
        if m:
            gap_candidates.append(float(m.group(1)) / 100.0)
            continue

        m2 = re.search(r"\bgap\s+([0-9]+(?:\.[0-9]+)?)\b", ln, flags=re.IGNORECASE)
        if m2:
            val = float(m2.group(1))
            gap_candidates.append(val if val <= 1.0 else val / 100.0)

        m3 = re.search(
            r"best bound.*gap\s+([0-9]+(?:\.[0-9]+)?)\s*%", ln, flags=re.IGNORECASE
        )
        if m3:
            gap_candidates.append(float(m3.group(1)) / 100.0)

    if gap_candidates:
        out["mip_gap"] = gap_candidates[0]

    return out


def _has_incumbent(res, model=None) -> bool:
    tc = res.solver.termination_condition
    st = res.solver.status

    has_sol = hasattr(res, "solution") and res.solution and len(res.solution) > 0

    if not has_sol and (model is not None):
        try:
            for v in model.component_data_objects(Var, active=True, descend_into=True):
                if v.value is not None:
                    has_sol = True
                    break
        except Exception:
            pass

    if tc in (TC.infeasible, TC.invalidProblem, TC.error, TC.solverFailure):
        return False

    if tc in (TC.optimal, TC.feasible):
        return True

    user_limited = tc in tuple(
        x for x in (
            getattr(TC, "userLimit",          None),
            getattr(TC, "maxTimeLimit",       None),
            getattr(TC, "resourceInterrupt",  None),
            getattr(TC, "userInterrupt",      None),
            getattr(TC, "other",              None),
        )
        if x is not None
    )

    if user_limited and has_sol:
        return True

    if (st in (SolverStatus.aborted, SolverStatus.ok, SolverStatus.warning)) and has_sol:
        return True

    return False


def write_iis_gurobi_with_timelimit(
    pyomo_model,
    iis_file_name: str,
    timelimit_s: float = 3600,
    *,
    iis_method=None,
    log_file=None,
):
    solver = SolverFactory("gurobi_persistent")
    solver.set_instance(pyomo_model, symbolic_solver_labels=True)
    grb = solver._solver_model

    grb.Params.TimeLimit = float(timelimit_s)
    if iis_method is not None:
        grb.Params.IISMethod = int(iis_method)
    if log_file is not None:
        grb.Params.LogFile      = str(log_file)
        grb.Params.LogToConsole = 0

    grb.computeIIS()

    if not iis_file_name.lower().endswith(".ilp"):
        iis_file_name = iis_file_name + ".ilp"
    grb.write(iis_file_name)

    iis_minimal = bool(getattr(grb, "IISMinimal", 0))
    return iis_file_name, iis_minimal
