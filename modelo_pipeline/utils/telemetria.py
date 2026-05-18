import os, time, csv
import pandas as pd
from pyomo.environ import Var, Constraint, Param, Set, value
from pyomo.opt import SolverStatus, TerminationCondition as TC

def _safe_len(x):
    try:
        return len(list(x))
    except Exception:
        try:
            return len(x)
        except Exception:
            return None

def _iter_vardata(m):
    for v in m.component_objects(Var, active=True):
        # v puede ser escalar o indexado; values() recorre ambos
        for vd in v.values():
            yield v.name, vd

def _iter_constrdata(m):
    for c in m.component_objects(Constraint, active=True):
        for cd in c.values():
            yield c.name, cd

def _iter_paramdata(m):
    for p in m.component_objects(Param, active=True):
        # Para parámetros, p.values() da ParamData si indexado; si escalar, también.
        # Aun así, algunos pueden venir como literales; por eso validamos abajo.
        for pdat in p.values():
            yield p.name, pdat

def _count_sets(m, set_names=("B","S","T","G","GRT","GRS","B_I","B_E","BC","BT","BH","BI","S_E","S_I")):
    sizes = {}
    for nm in set_names:
        if hasattr(m, nm):
            sizes[f"size_{nm}"] = _safe_len(getattr(m, nm))
    return sizes

def _param_value_safe(pdat):
    # Soporta ParamData, NumericValue o literal
    try:
        return pdat.value if hasattr(pdat, "value") else float(pdat)
    except Exception:
        try:
            return value(pdat)
        except Exception:
            return None

def model_size_stats(m, *, breakdown=False):
    # Variables
    total_vars = 0; bin_vars = 0; int_vars = 0; cont_vars = 0
    by_var_component = {}

    for vname, vd in _iter_vardata(m):
        total_vars += 1
        try:
            if vd.is_binary():
                bin_vars += 1
            elif vd.is_integer():
                int_vars += 1
            elif vd.is_continuous():
                cont_vars += 1
        except Exception:
            # Si algo raro pasa con el dominio, lo contamos como continuo
            cont_vars += 1
        if breakdown:
            by_var_component[vname] = by_var_component.get(vname, 0) + 1

    # Restricciones
    total_constr = 0
    by_constr_component = {}
    for cname, _cd in _iter_constrdata(m):
        total_constr += 1
        if breakdown:
            by_constr_component[cname] = by_constr_component.get(cname, 0) + 1

    # Parámetros (robusto a literales)
    total_params = 0
    by_param_component = {}
    for pname, pdat in _iter_paramdata(m):
        val = _param_value_safe(pdat)
        if val is not None:
            total_params += 1
            if breakdown:
                by_param_component[pname] = by_param_component.get(pname, 0) + 1

    stats = {
        "vars_total": total_vars,
        "vars_bin": bin_vars,
        "vars_int": int_vars,
        "vars_cont": cont_vars,
        "constr_total": total_constr,
        "params_total": total_params,
    }
    stats.update(_count_sets(m))

    extra = {}
    if breakdown:
        extra["by_var_component"] = by_var_component
        extra["by_constr_component"] = by_constr_component
        extra["by_param_component"] = by_param_component
    return stats, extra

def append_metrics_row(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame([row_dict]).to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        pd.DataFrame([row_dict]).to_csv(csv_path, mode="a", index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)

def telemetry_pack(m, *, meta, solve_elapsed=None, res=None, objective=None):
    stats, _ = model_size_stats(m, breakdown=False)
    tc = getattr(res.solver, "termination_condition", None) if res is not None else None
    st = getattr(res.solver, "status", None) if res is not None else None

    # Obj seguro a float
    try:
        obj_val = float(objective) if objective is not None else None
    except Exception:
        obj_val = None

    row = {
        **meta,
        **stats,
        "solver_status": str(st) if st is not None else None,
        "termination": str(tc) if tc is not None else None,
        "solve_seconds": round(float(solve_elapsed), 4) if solve_elapsed is not None else None,
        "obj_value": obj_val,
    }
    return row

def objective_value_safe(m, obj_name="obj"):
    try:
        obj = getattr(m, obj_name, None)
        if obj is None:
            # toma el primer Objective que encuentre
            for comp in m.component_objects():
                if comp.__class__.__name__ == "Objective":
                    obj = comp; break
        return value(obj) if obj is not None else None
    except Exception:
        return None

# (Opcional) breakdown detallado por componente
def write_breakdown_csv(m, csv_path, meta):
    _, extra = model_size_stats(m, breakdown=True)
    rows = []
    for k, d in (("var", extra.get("by_var_component", {})),
                 ("constr", extra.get("by_constr_component", {})),
                 ("param", extra.get("by_param_component", {}))):
        for name, cnt in sorted(d.items(), key=lambda x: (-x[1], x[0])):
            rows.append({**meta, "kind": k, "component": name, "count": cnt})
    if rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(rows)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", index=False, header=False)
