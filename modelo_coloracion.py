import math
from pyomo.environ import *
import logging, sys
from pyomo.opt import TerminationCondition
import pandas as pd
import os
from util import telemetry_pack, append_metrics_row, objective_value_safe
import time
import re
import subprocess
from typing import Any, Dict, Optional

from pyomo.opt import SolverStatus, TerminationCondition as TC

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, NonNegativeIntegers, Binary, NonNegativeReals, minimize,
    SolverFactory, TerminationCondition, value
)

from pyomo.contrib.iis import write_iis
import logging, sys, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("magdalena")

# =========================================================
# HELPERS DE TELEMETRÍA (paper-ready)
# =========================================================
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _gurobi_version_safe() -> Optional[str]:
    """
    Opción 1: gurobipy
    Opción 2: gurobi_cl --version
    """
    try:
        import gurobipy as gp
        v = gp.gurobi.version()  # (major, minor, technical)
        return ".".join(map(str, v))
    except Exception:
        pass

    try:
        out = subprocess.check_output(["gurobi_cl", "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"version\s+(\d+\.\d+\.\d+)", out, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        line0 = out.strip().splitlines()[0].strip()
        return line0[:100]
    except Exception:
        return None

def _extract_gurobi_stats_from_results(res) -> Dict[str, Any]:
    """
    Intenta leer mip_gap y node_count desde Results (Pyomo).
    Si no están, devuelve None y lo sacamos del log.
    """
    mip_gap = None
    node_count = None

    try:
        stats = getattr(res.solver, "statistics", None)
        if stats is not None:
            bb = getattr(stats, "branch_and_bound", None)
            if bb is not None:
                node_count = getattr(bb, "number_of_created_nodes", None) or getattr(bb, "number_of_nodes", None)

            mip_gap = getattr(stats, "mip_gap", None) or getattr(stats, "gap", None)
    except Exception:
        pass

    try:
        mip_gap = mip_gap or getattr(res.solver, "mip_gap", None) or getattr(res.solver, "gap", None)
    except Exception:
        pass

    return {
        "mip_gap": _safe_float(mip_gap),
        "node_count": None if node_count is None else int(node_count),
    }

def _parse_gurobi_log_for_threads_gap_nodes(log_path: str) -> Dict[str, Any]:
    """
    Parsea log de Gurobi para:
    - threads: "using up to X threads"
    - node_count: "Explored N nodes"
    - mip_gap: "gap ..."
    """
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

        m3 = re.search(r"best bound.*gap\s+([0-9]+(?:\.[0-9]+)?)\s*%", ln, flags=re.IGNORECASE)
        if m3:
            gap_candidates.append(float(m3.group(1)) / 100.0)

    if gap_candidates:
        out["mip_gap"] = gap_candidates[0]

    return out



def _has_incumbent(res, model=None):
    tc = res.solver.termination_condition
    st = res.solver.status

    # 1) Pyomo-style (cuando sí viene algo en res.solution)
    has_sol = hasattr(res, "solution") and res.solution and len(res.solution) > 0

    # 2) Model-aware: si load_solutions=True, los valores están en las Vars
    if not has_sol and (model is not None):
        try:
            # toma una variable y revisa si tiene algún valor distinto de None
            for v in model.component_data_objects(Var, active=True, descend_into=True):
                if v.value is not None:
                    has_sol = True
                    break
        except Exception:
            pass

    # Casos sin solución
    from pyomo.opt import TerminationCondition as TC, SolverStatus
    if tc in (TC.infeasible, TC.invalidProblem, TC.error, TC.solverFailure):
        return False

    # Óptimo / factible explícito
    if tc in (TC.optimal, TC.feasible):
        return True

    # Limites benignos → válidos si hay algo cargado en el modelo
    user_limited = tc in tuple(x for x in (
        getattr(TC, "userLimit", None),        # p.ej. SolutionLimit
        getattr(TC, "maxTimeLimit", None),
        getattr(TC, "resourceInterrupt", None),
        getattr(TC, "userInterrupt", None),
        getattr(TC, "other", None),
    ) if x is not None)

    if user_limited and has_sol:
        return True

    # Algunos backends marcan aborted pero con solución (tu WARNING)
    if (st in (SolverStatus.aborted, SolverStatus.ok, SolverStatus.warning)) and has_sol:
        return True

    return False

from pyomo.environ import SolverFactory

def write_iis_gurobi_with_timelimit(pyomo_model, iis_file_name, timelimit_s=120, *, iis_method=None, log_file=None):
    """
    Calcula IIS con Gurobi (vía gurobi_persistent) con límite de tiempo.
    Devuelve (path_escrito, iis_minimal_bool).
    """
    solver = SolverFactory("gurobi_persistent")
    solver.set_instance(pyomo_model, symbolic_solver_labels=True)

    grb = solver._solver_model  # modelo gurobipy interno

    # Time limit para computeIIS (Gurobi lo respeta)
    grb.Params.TimeLimit = float(timelimit_s)

    # Opcional: método de IIS (depende de versión; si no sabes, no lo toques)
    if iis_method is not None:
        grb.Params.IISMethod = int(iis_method)

    # Opcional: log del IIS
    if log_file is not None:
        grb.Params.LogFile = str(log_file)
        grb.Params.LogToConsole = 0

    grb.computeIIS()

    # Gurobi decide el formato por extensión; lo típico es .ilp
    if not iis_file_name.lower().endswith(".ilp"):
        iis_file_name = iis_file_name + ".ilp"
    grb.write(iis_file_name)

    # IISMinimal = 1 si es irreducible/minimal; si se cortó por tiempo, típicamente queda 0
    iis_minimal = bool(getattr(grb, "IISMinimal", 0))
    return iis_file_name, iis_minimal


def ejecutar_instancias_coloracion(
    semanas,
    participacion,
    resultados_dir
):

    
    semanas_a_procesar = semanas
    PARTICIPACION_C = participacion
    resultados_dir_script = resultados_dir
    
    # Crear directorio base para instancias_magdalena
    resultados_magdalena_base_path = os.path.join(resultados_dir_script, "resultados_magdalena") 
    os.makedirs(resultados_magdalena_base_path, exist_ok=True)
    
    print("\n===== CREANDO CARPETAS SEMANALES PARA INSTANCIAS =====")
    for semana_folder_name in semanas_a_procesar:
        path_semana_folder = os.path.join(resultados_magdalena_base_path, semana_folder_name)
        os.makedirs(path_semana_folder, exist_ok=True)
        print(f"Directorio creado/verificado: {path_semana_folder}")
    print("===== CREACIÓN DE CARPETAS SEMANALES COMPLETADA =====\n")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # El print de cabecera general puede quedar fuera del bucle si se desea
    print("Iniciando procesamiento de optimización para múltiples semanas...")
    
    semanas_infactibles = []
    
    
    for semana_actual in semanas_a_procesar:
        print(f"\n--- Procesando Semana: {semana_actual} ---")
    
        # Inicializar listas para almacenar resultados PARA LA SEMANA ACTUAL
        # Esto asegura que cada archivo "Distancias_Modelo..." contenga solo los datos de su semana.
        resumen_semanal_actual = []
        resultados_segregacion_actual = []
        detalle_movimientos_actual = []

        directorio_datos_semanal = os.path.join(resultados_dir_script, "instancias_magdalena", semana_actual)

        # Asegurar que el directorio exista (principalmente para la salida, ya que la instancia debe existir)
        os.makedirs(directorio_datos_semanal, exist_ok=True)
    
        archivo_instancia = os.path.join(directorio_datos_semanal, f"Instancia_{semana_actual}_{PARTICIPACION_C}_K.xlsx")
        resultado_file_semana = os.path.join(resultados_magdalena_base_path, semana_actual, f"resultado_{semana_actual}_{PARTICIPACION_C}_K.xlsx")
        resultado_distancias_file_semana = os.path.join(resultados_magdalena_base_path, semana_actual, f"Distancias_Modelo_{semana_actual}_{PARTICIPACION_C}.xlsx")
        
        metrics_csv = os.path.join(resultados_dir_script, "metrics", "metrics_magdalena.csv")
    
        try:
            # Verificar si el archivo de instancia existe ANTES de intentar leerlo
            if not os.path.exists(archivo_instancia):
                print(f"ADVERTENCIA: Archivo de instancia no encontrado para la semana {semana_actual}: {archivo_instancia}. Saltando esta semana.")
                continue # Pasar a la siguiente semana
    
            t_build0 = time.perf_counter()
            model = ConcreteModel()
    
            # Leer DataFrame
            df = pd.read_excel(archivo_instancia, sheet_name=None)
            
            # Crear diccionario de mapeo de segregaciones
            segregacion_map = dict(zip(df['S']['S'], df['S']['Segregacion']))
    
            # Conjuntos
            model.B = Set(initialize=df["B"].iloc[:, 0].tolist())
            model.S = Set(initialize=df["S"].iloc[:, 0].tolist())
            model.T = Set(initialize=df["T"].iloc[:, 0].tolist())
    
            # Parámetros
            model.C = Param(model.B, initialize=df['C_b'].set_index('B')['C'].to_dict())
            model.VS = Param(model.B, initialize=df['VS_b'].set_index('B')['VS'].to_dict())
            model.VSR = Param(model.B, initialize=df['VSR_b'].set_index('B')['VSR'].to_dict())
            model.KS = Param(model.S, initialize=df['KS_s'].set_index('S')['KS'].to_dict())
            model.KI = Param(model.S, initialize=df['KI_s'].set_index('S')['KI'].to_dict())
    
            I0_dict = {(row['S'], row['B']): row['I0'] for _, row in df['I0_sb'].iterrows()}
            model.I0 = Param(model.S, model.B, initialize=I0_dict, within=NonNegativeIntegers)
    
            DR_dict = {(row['S'], row['T']): row['DR'] for _, row in df['D_params'].iterrows()}
            model.DR = Param(model.S, model.T, initialize=DR_dict, within=NonNegativeIntegers)
    
            DC_dict = {(row['S'], row['T']): row['DC'] for _, row in df['D_params'].iterrows()}
            model.DC = Param(model.S, model.T, initialize=DC_dict, within=NonNegativeIntegers)
    
            DD_dict = {(row['S'], row['T']): row['DD'] for _, row in df['D_params'].iterrows()}
            model.DD = Param(model.S, model.T, initialize=DD_dict, within=NonNegativeIntegers)
    
            DE_dict = {(row['S'], row['T']): row['DE'] for _, row in df['D_params'].iterrows()}
            model.DE = Param(model.S, model.T, initialize=DE_dict, within=NonNegativeIntegers)
    
            lc_dict = {(row['S'], row['B']): row['LC'] for _, row in df['LC_sb'].iterrows()}
            model.LC = Param(model.S, model.B, initialize=lc_dict, within=NonNegativeIntegers)
            

            # =========================
            # DISPERSIÓN (anti-concentración) - parámetros auxiliares
            # =========================
            THETA_DISPERSION = 1.4
            dispersion_mode = "prefijo"  # "global" o "prefijo"
            
            S_list = [s for s in df["S"].iloc[:, 0].tolist()]
            T_list = [int(t) for t in df["T"].iloc[:, 0].tolist()]  # asegura int para comparar tau<=t
            T_list = sorted(T_list)
            
            KI_map = df['KI_s'].set_index('S')['KI'].to_dict()
            
            # ---------
            # Global: TOTIN[s] y CAPBLOCK[s]
            # ---------
            if dispersion_mode == "global":
                TotIn_dict = {
                    s: sum(int(DR_dict[(s, t)]) + int(DD_dict[(s, t)]) for t in T_list)
                    for s in S_list
                }
                model.TOTIN = Param(model.S, initialize=TotIn_dict, within=NonNegativeIntegers)
            
                CapBlock_dict = {}
                for s in S_list:
                    tot = int(TotIn_dict[s])
                    ki = max(1, int(KI_map[s]))
                    cap = int(math.ceil(THETA_DISPERSION * tot / ki)) if tot > 0 else 0
                    cap = min(cap, tot)
                    CapBlock_dict[s] = cap
            
                model.CAPBLOCK = Param(model.S, initialize=CapBlock_dict, within=NonNegativeIntegers)
            
            # ---------
            # Prefijo: TOTINP[s,t] y CAPBLOCKP[s,t]
            # ---------
            if dispersion_mode == "prefijo":
                TotInP_dict = {}
                CapBlockP_dict = {}
            
                for s in S_list:
                    ki = max(1, int(KI_map[s]))
                    running = 0
                    for t in T_list:
                        running += int(DR_dict[(s, t)]) + int(DD_dict[(s, t)])
                        TotInP_dict[(s, t)] = running
            
                        if running > 0:
                            cap = int(math.ceil(THETA_DISPERSION * running / ki))
                            cap = min(cap, running)
                        else:
                            cap = 0
            
                        CapBlockP_dict[(s, t)] = cap
            
                model.TOTINP = Param(model.S, model.T, initialize=TotInP_dict, within=NonNegativeIntegers)
                model.CAPBLOCKP = Param(model.S, model.T, initialize=CapBlockP_dict, within=NonNegativeIntegers)

            
            model.LE = Param(model.B, initialize=df['LE_b'].set_index('B')['LE'].to_dict())
            model.TEU = Param(model.S, initialize=df['TEU_s'].set_index('S')['TEU'].to_dict())
            model.OS = Param(initialize=1, mutable=True)
            
            C_mediana = float(pd.Series(df['C_b']['C']).astype(float).median())
            es_pila = C_mediana <= 6.0
            
            if es_pila:
                # En modo 'pila' no imponemos mínimo por turno al abrir/unidad
                OI_dict = {row.B: 0.2 for _, row in df['C_b'].iterrows()}
            else:
                # En modo 'bahía' mantén tu regla original OI=1/C[b]
                OI_dict = {row.B: 1.0/float(row.C) for _, row in df['C_b'].iterrows()}
                
            model.OI = Param(model.B, within=NonNegativeReals, initialize=OI_dict)
            
            model.r = Param(initialize=348)
            model.R = Param(model.S, initialize=df['R_s'].set_index('S')['R'].to_dict())
    
            # Variables de decisión
            model.fr = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.fc = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.fd = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.fe = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            
            model.y  = Var(model.S, model.B, model.T, domain=Binary, initialize=0)
            model.u  = Var(model.S, model.B, domain=Binary, initialize=0)
            
            # v = nº de bahías asignadas a (s,b,t) → debe seguir siendo entera
            model.v  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            # k = nº de bloques usados por s → entera
            model.k  = Var(model.S, domain=NonNegativeIntegers, initialize=0)
            
            # inventario y cargas agregadas: continuas
            model.i  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.w  = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.p  = Var(model.T, domain=NonNegativeIntegers, initialize=0.0)
            model.q  = Var(model.T, domain=NonNegativeIntegers, initialize=0.0)
            
        
            # ► Restricciones opcionales (según switches)
            #   Cota inferior de flujo por turno:
            #     fr + fd >= ceil( alpha[s]*TC[s,t] ) * u[s,b]
            if usar_cota_inferior:
                def lower_flow_rule(m, s, b, t):
                    rhs = m.alpha[s] * m.TC[s, t]
                    if rhs < 1:
                        return Constraint.Skip
                    return m.fr[s, b, t] + m.fd[s, b, t] >= math.ceil(rhs) * m.y[s, b, t]
                model.constraint_lower_flow = Constraint(model.S, model.B, model.T, rule=lower_flow_rule)
            
                #   Cota de dispersión sobre el flujo ENTRANTE del turno (A):
                #     fr[s,b,t] + fd[s,b,t] ≤ gamma[s] * Σ_b' (fr[s,b',t] + fd[s,b',t])
                #   (se omite si no hay flujo total en el turno)
                if usar_cota_superior:
                    def upper_flow_dispersal_rule(m, s, b, t):
                        if m.TC[s, t] == 0:
                            return Constraint.Skip
                        return (
                            m.fr[s, b, t] + m.fd[s, b, t]
                            <= m.gamma[s] * sum(m.fr[s, bp, t] + m.fd[s, bp, t] for bp in m.B)
                        )
                    model.constraint_upper_flow = Constraint(model.S, model.B, model.T, rule=upper_flow_dispersal_rule)
           

            # =========================
            # Dispersion constraint
            # =========================
            if dispersion_mode == "global":
                def anti_concentracion_rule(m, s, b):
                    if m.TOTIN[s] == 0:
                        return Constraint.Skip
                    return (
                        sum(m.fr[s, b, t] + m.fd[s, b, t] for t in m.T)
                        <= m.CAPBLOCK[s] * m.u[s, b]
                    )
                model.constraint_dispersion = Constraint(model.S, model.B, rule=anti_concentracion_rule)
            
            elif dispersion_mode == "prefijo":
                def anti_concentracion_prefijo_rule(m, s, b, t):
                    if m.TOTINP[s, t] == 0:
                        return Constraint.Skip
                    return (
                        sum(m.fr[s, b, tau] + m.fd[s, b, tau] for tau in m.T if tau <= t)
                        <= m.CAPBLOCKP[s, t] * m.u[s, b]
                    )
                model.constraint_dispersion_prefix = Constraint(model.S, model.B, model.T, rule=anti_concentracion_prefijo_rule)
                      
           # =========================
            
            # Restricciones (2)
            model.constraint_2 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        if t == 1:
                            model.constraint_2.add(
                                expr=model.i[s, b, t] == model.I0[s, b] + model.fr[s, b, t] + model.fd[s, b, t]
                                - model.fc[s, b, t] - model.fe[s, b, t]
                            )
                        else:
                            model.constraint_2.add(
                                expr=model.i[s, b, t] == model.i[s, b, t-1] + model.fr[s, b, t] + model.fd[s, b, t]
                                - model.fc[s, b, t] - model.fe[s, b, t]
                            )
    
            # Restricciones (3)
            model.constraint_3 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_3.add(
                            expr = model.i[s, b, t] <= model.v[s, b, t] * model.OS * model.C[b]
                        )
    
            # Restricción (4)
            model.constraint_4 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_4.add(
                            expr = (model.v[s, b, t] - 1) * model.C[b] * model.OS + model.C[b] * model.OI[b] <= model.i[s, b, t]
                        )
     
            # Restricciones (5)
            model.constraint_5 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_5.add(expr=sum(model.fr[s, b, t] for b in model.B) == model.DR[s, t])
    
            # Restricciones (6)
            model.constraint_6 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_6.add(expr=sum(model.fc[s, b, t] for b in model.B) == model.DC[s, t])
    
            # Restricciones (7)
            model.constraint_7 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_7.add(expr=sum(model.fd[s, b, t] for b in model.B) == model.DD[s, t])
    
            # Restricciones (8)
            model.constraint_8 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_8.add(expr=sum(model.fe[s, b, t] for b in model.B) == model.DE[s, t])
    
            # Restricciones (9)
            model.constraint_9 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_9.add(
                            expr=model.fr[s, b, t] + model.fd[s, b, t] <= (model.DR[s, t] + model.DD[s, t]) * model.y[s, b, t]
                        )
    
            # Restricciones (10)
            model.constraint_10 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_10.add(expr=(model.fr[s, b, t] + model.fd[s, b, t]) >= model.y[s, b, t])
    
            # Restricciones (11)
            model.constraint_11 = ConstraintList()
            for b in model.B:
                for s in model.S:
                    model.constraint_11.add(expr=model.u[s, b] <= sum(model.y[s, b, t] for t in model.T))
    
            # Restricción (12)
            model.constraint_12 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_12.add(expr=model.u[s, b] >= model.y[s, b, t])
    
            # Restricciones (13)
            model.constraint_13 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_13.add(
                        expr = sum(model.v[s, b, t] * model.TEU[s] for s in model.S) <= model.VS[b]
                    )
    
            # Restricciones (14)
            model.constraint_14 = ConstraintList()
            for s in model.S:
                model.constraint_14.add(expr=model.k[s] == sum(model.u[s, b] for b in model.B))
    
            # Restricciones (15)
            model.constraint_15 = ConstraintList()
            for s in model.S:
                if sum(model.DR[s, t] for t in model.T) == 0 and sum(model.DD[s, t] for t in model.T) == 0:
                    model.constraint_15.add(model.k[s] == 0)
                else:
                    model.constraint_15.add(model.k[s] <= model.KS[s])
    
            # Restricciones (16)
            model.constraint_16 = ConstraintList()
            for s in model.S:
                if sum(model.DR[s, t] for t in model.T) == 0 and sum(model.DD[s, t] for t in model.T) == 0:
                    model.constraint_16.add(model.k[s] == 0)
                else:
                    model.constraint_16.add(model.k[s] >= model.KI[s])
    
            # Restricciones (17), (18) y (19)
            model.constraint_17 = ConstraintList()
            model.constraint_18 = ConstraintList()
            model.constraint_19 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_17.add(
                        model.w[b, t] == sum(model.fr[s, b, t] + model.fc[s, b, t] + model.fd[s, b, t] + model.fe[s, b, t]
                                            for s in model.S)
                    )
                for b in model.B:
                    model.constraint_18.add(model.p[t] >= model.w[b, t])
                    model.constraint_19.add(model.q[t] <= model.w[b, t])
    
            # Restricción (20)
            model.constraint_20 = ConstraintList()
            for t in model.T:
                model.constraint_20.add(expr=model.p[t] - model.q[t] <= model.r)
    
            # * model.TEU[s]
            # Restricción (21)
            model.constraint_21 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_21.add(
                        expr = sum(model.v[s, b, t] * model.R[s] for s in model.S) <= model.VSR[b]
                    )
    
            # Función objetivo
            def objective_rule(model):
                w1 = 1
                w2 = 1
                return (
                    w1 * sum(model.fc[s, b, t] * model.LC[s, b] for b in model.B for s in model.S for t in model.T)
                    + w2 * sum(model.fe[s, b, t] * model.LE[b] for b in model.B for s in model.S for t in model.T)
                )
    
            model.objective = Objective(rule=objective_rule, sense=minimize)
            t_build1 = time.perf_counter()
            build_seconds = t_build1 - t_build0
            
            solver = SolverFactory('gurobi')
            log_path = os.path.join(directorio_datos_semanal, f'gurobi_log_{semana_actual}.log')
            solver.options.update({
                'LogToConsole': 0,
                'LogFile': log_path,
                'MIPGap': 1e-3,
                'FeasibilityTol': 1e-5,
                'OptimalityTol': 1e-8,
                'IntFeasTol': 1e-5,
                'TimeLimit': 2000, # 2 hora
                'MIPFocus': 1,      # prioriza factibilidad
                'Heuristics': 0.5,  # más heurística
                'PumpPasses': 20,   # feasibility pump
                'SolutionLimit': 20, # detener al encontrar la primera solución factible
            })
    
            t0 = time.perf_counter()
            res = solver.solve(model, tee=True, load_solutions=True)
            t1 = time.perf_counter()
            solve_seconds = t1 - t0
            
            # ====== NUEVAS MÉTRICAS paper-ready ======
            gurobi_version = _gurobi_version_safe()
            
            stats_res = _extract_gurobi_stats_from_results(res)
            stats_log = _parse_gurobi_log_for_threads_gap_nodes(log_path)
            
            mip_gap = stats_res["mip_gap"] if stats_res["mip_gap"] is not None else stats_log["mip_gap"]
            node_count = stats_res["node_count"] if stats_res["node_count"] is not None else stats_log["node_count"]
            threads = stats_log["threads"]
            
            tc = res.solver.termination_condition
            
            
            if tc == TerminationCondition.infeasible:
                logger.error("🚨 Infactible en %s: escribiendo LP + IIS…", semana_actual)
                results_dir_semana = os.path.join(resultados_magdalena_base_path, semana_actual)
                lp_path  = os.path.join(results_dir_semana, f"modelo_inf_{semana_actual}.lp")
                model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})       
                iis_path = os.path.join(results_dir_semana, f"modelo_inf_{semana_actual}.ilp")
                iis_log  = os.path.join(results_dir_semana, f"iis_log_{semana_actual}.log")
                
                iis_path_written, is_min = write_iis_gurobi_with_timelimit(
                    model,
                    iis_path,
                    timelimit_s=60,      # << tu límite en segundos
                    iis_method=None,      # o 0/1/2 si quieres probar
                    log_file=iis_log
                )
                
                logger.error("IIS escrito en %s (IISMinimal=%s)", iis_path_written, int(is_min))
                
                
                semanas_infactibles.append(semana_actual)
                continue
            
            if not _has_incumbent(res, model):
                # Aquí ya NO escribimos LP si hay solución adjunta: sólo guarda LP cuando de verdad no hay incumbente.
                logger.error("⛔ %s terminó sin incumbente real. Guardo LP y sigo.", semana_actual)
                results_dir_semana = os.path.join(resultados_magdalena_base_path, semana_actual)
                lp_path = os.path.join(results_dir_semana, f"modelo_{semana_actual}_nolb.lp")
                model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})
                continue
            
            logger.info("✅ Semana %s con solución (tc=%s).", semana_actual, res.solver.termination_condition)
    
            # Calcular distancia para exportación (expo)
            distancia_expo = sum(
                value(model.fc[s, b, t]) * value(model.LC[s, b])
                for b in model.B for s in model.S for t in model.T
            )
            
            # Calcular distancia para importación (impo)
            distancia_impo = sum(
                value(model.fe[s, b, t]) * value(model.LE[b])
                for b in model.B for s in model.S for t in model.T
            )
            
            # Calcular distancias y movimientos
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
    
            # Agregar al resumen de la semana actual
            resumen_semanal_actual.append({
                'Semana': semana_actual, # Usar semana_actual
                'Distancia Total': value(model.objective),
                'Distancia LOAD': distancia_load_total,
                'Distancia DLVR': distancia_dlvr_total,
                'Movimientos_DLVR': sum(movimientos_dlvr_por_seg.values()),
                'Movimientos_LOAD': sum(movimientos_load_por_seg.values())
            })
    
            # Agregar a resultados por segregación de la semana actual
            for s in model.S:
                resultados_segregacion_actual.append({
                    'Semana': semana_actual, # Usar semana_actual
                    'Segregacion': segregacion_map[s],
                    'Distancia_Total': distancia_expo_por_seg[s] + distancia_impo_por_seg[s],
                    'Distancia_DLVR': distancia_impo_por_seg[s],
                    'Distancia_LOAD': distancia_expo_por_seg[s],
                    'Movimientos_DLVR': movimientos_dlvr_por_seg[s],
                    'Movimientos_LOAD': movimientos_load_por_seg[s]
                })
    
            # Agregar al detalle de movimientos de la semana actual
            for s in model.S:
                for b in model.B:
                    movimientos_dlvr = sum(value(model.fe[s, b, t]) for t in model.T)
                    movimientos_load = sum(value(model.fc[s, b, t]) for t in model.T)
                    if movimientos_dlvr > 0 or movimientos_load > 0:
                        detalle_movimientos_actual.append({
                            'Semana': semana_actual, # Usar semana_actual
                            'Segregacion': segregacion_map[s],
                            'Bloque': b,
                            'Movimientos DLVR': movimientos_dlvr,
                            'Movimientos LOAD': movimientos_load
                        })
    
            print(f"Semana {semana_actual}: {value(model.objective)}, {distancia_load_total}, {distancia_dlvr_total}")
            
            # Extraer resultados de las variables del modelo
            fr_values = [(s, b, t, model.fr[s, b, t].value) for s in model.S for b in model.B for t in model.T]
            df_fr = pd.DataFrame(fr_values, columns=["Segregación", "Bloque", "Periodo", "Recibir"])
    
            fc_values = [(s, b, t, model.fc[s, b, t].value) for s in model.S for b in model.B for t in model.T]
            df_fc = pd.DataFrame(fc_values, columns=["Segregación", "Bloque", "Periodo", "Cargar"])
    
            fd_values = [(s, b, t, model.fd[s, b, t].value) for s in model.S for b in model.B for t in model.T]
            df_fd = pd.DataFrame(fd_values, columns=["Segregación", "Bloque", "Periodo", "Descargar"])
    
            fe_values = [(s, b, t, model.fe[s, b, t].value) for s in model.S for b in model.B for t in model.T]
            df_fe = pd.DataFrame(fe_values, columns=["Segregación", "Bloque", "Periodo", "Entregar"])
    
            f_values = [(s, b, t, model.fr[s, b, t].value, model.fc[s, b, t].value,
                         model.fd[s, b, t].value, model.fe[s, b, t].value)
                        for s in model.S for b in model.B for t in model.T]
            df_f = pd.DataFrame(f_values, columns=["Segregación", "Bloque", "Periodo", "Recepción", "Carga", "Descarga", "Entregar"])
    
            y_values = [(s, b, t, model.y[s, b, t].value) for s in model.S for b in model.B for t in model.T]
            df_y = pd.DataFrame(y_values, columns=["Segregación", "Bloque", "Periodo", "Asignado"])
    
            k_values = [(s, model.k[s].value) for s in model.S]
            df_k = pd.DataFrame(k_values, columns=["Segregación", "Total bloques asignadas"])
    
            i_values = [(s, b, t, model.i[s, b, t].value * model.TEU[s]) for s in model.S for b in model.B for t in model.T]
            df_i = pd.DataFrame(i_values, columns=["Segregación", "Bloque", "Periodo", "Volumen"])
    
            v_values = [(s, b, t, model.v[s, b, t].value * model.TEU[s]) for s in model.S for b in model.B for t in model.T]
            df_v = pd.DataFrame(v_values, columns=["Segregación", "Bloque", "Periodo", "Bahías ocupadas"])
    
            bloques_orden = df['B'].iloc[:, 0].tolist()
            bloque_id_map = {b: i+1 for i, b in enumerate(bloques_orden)}
            
            segs_orden = df['S'].iloc[:, 0].tolist()
            seg_id_map = {s: i+1 for i, s in enumerate(segs_orden)}
    
            w_values = [(b, t, model.w[b, t].value, bloque_id_map[b]) for b in model.B for t in model.T]
            df_w = pd.DataFrame(w_values, columns=["Bloque", "Periodo", "Carga de trabajo", "BloqueID"])
    
            pq_values = [(t, model.p[t].value, model.q[t].value) for t in model.T]
            df_pq = pd.DataFrame(pq_values, columns=["Periodo", "Carga máxima", "Carga mínima"])
    
            r_values = [model.r.value]
            df_r = pd.DataFrame(r_values, columns=["Variación Carga de trabajo"])
    
            gen = [(s, b, t, model.fr[s, b, t].value, model.fc[s, b, t].value, model.fd[s, b, t].value,
                    model.fe[s, b, t].value, model.y[s, b, t].value, model.i[s, b, t].value * model.TEU[s],
                    model.v[s, b, t].value * model.TEU[s], bloque_id_map[b], seg_id_map[s], model.VS[b])
                   for s in model.S for b in model.B for t in model.T]
            df_gen = pd.DataFrame(gen, columns=["Segregación", "Bloque", "Periodo", "Recepción", "Carga",
                                                     "Descarga", "Entrega", "Asignado", "Volumen (TEUs)",
                                                     "Bahías Ocupadas", "BloqueID", "SegregaciónID", "Bahías"])
    
            # Calcular el incremento de bahías ocupadas
            def calcular_incremento_bahias(group):
                group = group.sort_values('Periodo')
                group['Incremento Bahías'] = group['Bahías Ocupadas'].diff().fillna(group['Bahías Ocupadas'])
                group['Incremento Bahías'] = group['Incremento Bahías'].apply(lambda x: max(0, x))
                return group
    
            import warnings # Ya importado al inicio
            warnings.filterwarnings("ignore", category=DeprecationWarning) # Ya configurado al inicio
    
            df_gen = df_gen.groupby(['Segregación', 'Bloque']).apply(calcular_incremento_bahias).reset_index(drop=True)
    
            gen = [(row['Segregación'], row['Bloque'], row['Periodo'],
                    row['Recepción'], row['Carga'], row['Descarga'], row['Entrega'],
                    row['Asignado'], row['Volumen (TEUs)'], row['Bahías Ocupadas'],
                    row['BloqueID'], row['SegregaciónID'], row['Bahías'], row['Incremento Bahías'])
                   for _, row in df_gen.iterrows()]
    
            df_gen = pd.DataFrame(gen, columns=["Segregación", "Bloque", "Periodo", "Recepción", "Carga",
                                                     "Descarga", "Entrega", "Asignado", "Volumen (TEUs)",
                                                     "Bahías Ocupadas", "BloqueID", "SegregaciónID", "Bahías",
                                                     "Incremento Bahías"])
    
            cap_bloque = [
                (s, b, t, model.C[b] * model.VS[b] * model.OS.value,
                 model.i[s, b, t].value * model.TEU[s],
                 sum(model.C[b_inner] * model.VS[b_inner] * model.OS.value for b_inner in model.B), # Corregido para sumar todos los bloques
                 bloque_id_map[b], seg_id_map[s], model.VS[b])
                for s in model.S for b in model.B for t in model.T
            ]
            df_c_b = pd.DataFrame(cap_bloque, columns=["Segregación", "Bloque", "Periodo", "Capacidad Bloque",
                                                     "Volumen bloques (TEUs)", "Cap Patio", "BloqueID", "SegregaciónID", "Bahías"])
    
    
            # Calcular la cantidad de contenedores por turno y por bloque
            datos_turno_bloque = []
            for t in model.T:
                for b in model.B:
                    total_contenedores = sum(value(model.i[s, b, t]) for s in model.S)
                    datos_turno_bloque.append({
                        'Turno': t,
                        'Bloque': b,
                        'Contenedores': total_contenedores
                    })
            
            df_turno_bloque = pd.DataFrame(datos_turno_bloque)
            df_pivot_turno_bloque = df_turno_bloque.pivot(index='Turno', columns='Bloque', values='Contenedores')
            df_pivot_turno_bloque = df_pivot_turno_bloque.fillna(0) 
            
            with pd.ExcelWriter(resultado_file_semana, engine='openpyxl') as writer:
                df_gen.to_excel(writer, sheet_name="General", index=False)
                df_c_b.to_excel(writer, sheet_name="Ocupación Bloques", index=False)
                df_k.to_excel(writer, sheet_name="Total bloques", index=False)
                df_w.to_excel(writer, sheet_name="Workload bloques", index=False)
                df_fr.to_excel(writer, sheet_name="Recibir", index=False)
                df_fc.to_excel(writer, sheet_name="Cargar", index=False)
                df_fd.to_excel(writer, sheet_name="Descargar", index=False)
                df_fe.to_excel(writer, sheet_name="Entregar", index=False)
                df_f.to_excel(writer, sheet_name="Flujos", index=False)
                df_y.to_excel(writer, sheet_name="Asignado", index=False)
                df_i.to_excel(writer, sheet_name="Volumen bloques (TEUs)", index=False)
                df_v.to_excel(writer, sheet_name="Bahías por bloques", index=False)
                df_pq.to_excel(writer, sheet_name="Carga máx-min", index=False)
                df_r.to_excel(writer, sheet_name="Variación Carga de trabajo", index=False)
                df_pivot_turno_bloque.to_excel(writer, sheet_name="Contenedores Turno-Bloque", index=True)
            print(f"Resultados principales para {semana_actual} guardados en {resultado_file_semana}")
    
        except Exception as e:
            print(f"Error procesando semana {semana_actual}: Error - {str(e)}")
            continue # Continuar con la siguiente semana en caso de error
    
        # Crear DataFrames a partir de las listas de la semana actual
        df_resumen_semanal_actual_df = pd.DataFrame(resumen_semanal_actual)
        df_resultados_segregacion_actual_df = pd.DataFrame(resultados_segregacion_actual)
        df_detalle_movimientos_actual_df = pd.DataFrame(detalle_movimientos_actual)
    
        # Guardar resultados de resumen en Excel para la semana actual
        try:
            with pd.ExcelWriter(resultado_distancias_file_semana, engine='openpyxl') as writer:
                df_resumen_semanal_actual_df.to_excel(writer, sheet_name='Resumen Semanal', index=False)
                df_resultados_segregacion_actual_df.to_excel(writer, sheet_name='Resultados por Segregación', index=False)
                df_detalle_movimientos_actual_df.to_excel(writer, sheet_name='Detalle de Movimientos', index=False)
            print(f"Resumen de distancias para {semana_actual} guardado en {resultado_distancias_file_semana}")
        except Exception as e:
            print(f"Error al guardar el archivo de resumen de distancias para {semana_actual}: {str(e)}")
            
        meta = {
            "modelo": "magdalena",
            "semana": semana_actual,
            "participacion": str(PARTICIPACION_C),
            "fase": "final",
            "usar_cota_inferior": bool(usar_cota_inferior),
            "usar_cota_superior": bool(usar_cota_superior),
            "beta_alpha": float(beta_alpha),
            "gamma_val": float(gamma_val),
            "resultado_xlsx": resultado_file_semana,
            "resultado_distancias": resultado_distancias_file_semana,
        
            # NUEVAS PAPER-METRICS
            "build_seconds": build_seconds,
            "gurobi_version": gurobi_version,
            "threads": threads,
            "mip_gap": mip_gap,
            "node_count": node_count,
        }
        
        obj_val = objective_value_safe(model, obj_name="objective")
        row = telemetry_pack(model, meta=meta, solve_elapsed=solve_seconds, res=res, objective=obj_val)
        
        # Por si telemetry_pack no expande meta a columnas, lo reforzamos:
        row["build_seconds"] = build_seconds
        row["gurobi_version"] = gurobi_version
        row["threads"] = threads
        row["mip_gap"] = mip_gap
        row["node_count"] = node_count
        
        append_metrics_row(metrics_csv, row)
        

        
    
    print("\nProceso completado para todas las semanas.")
    
    semanas_filtradas = [s for s in semanas_a_procesar 
                         if s not in semanas_infactibles]
    
    # Imprimimos en el formato literal Python que pedías
    print("\nsemanas_a_procesar = [")
    for s in semanas_filtradas:
        print(f'    "{s}",')
    print("]")
    
    return semanas_filtradas, semanas_infactibles