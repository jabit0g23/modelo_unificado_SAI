import math
from pyomo.environ import *
import logging, sys
from pyomo.opt import TerminationCondition
import pandas as pd
import os
from util import telemetry_pack, append_metrics_row, objective_value_safe
import time

from pyomo.opt import SolverStatus, TerminationCondition as TC

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList, Objective, Expression,
    NonNegativeIntegers, Binary, NonNegativeReals, minimize,
    SolverFactory, TerminationCondition, value
)

from pyomo.contrib.iis import write_iis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("magdalena")


def _has_incumbent(res, model=None):
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

    user_limited = tc in tuple(x for x in (
        getattr(TC, "userLimit", None),
        getattr(TC, "maxTimeLimit", None),
        getattr(TC, "resourceInterrupt", None),
        getattr(TC, "userInterrupt", None),
        getattr(TC, "other", None),
    ) if x is not None)

    if user_limited and has_sol:
        return True

    if (st in (SolverStatus.aborted, SolverStatus.ok, SolverStatus.warning)) and has_sol:
        return True

    return False


def ejecutar_instancias(
    semanas,
    participacion,
    resultados_dir,
    # Pareto controls (24h)
    pareto_enabled=True,
    pareto_points=12,      # cantidad de eps en el barrido
    pareto_pad=0.05,       # 5% de holgura por arriba del mínimo D
):

    semanas_a_procesar = semanas
    PARTICIPACION_C = participacion
    resultados_dir_script = resultados_dir

    resultados_magdalena_base_path = os.path.join(resultados_dir_script, "resultados_magdalena")
    os.makedirs(resultados_magdalena_base_path, exist_ok=True)

    print("\n===== CREANDO CARPETAS SEMANALES PARA INSTANCIAS =====")
    for semana_folder_name in semanas_a_procesar:
        path_semana_folder = os.path.join(resultados_magdalena_base_path, semana_folder_name)
        os.makedirs(path_semana_folder, exist_ok=True)
        print(f"Directorio creado/verificado: {path_semana_folder}")
    print("===== CREACIÓN DE CARPETAS SEMANALES COMPLETADA =====\n")

    print("Iniciando procesamiento de optimización para múltiples semanas...")
    semanas_infactibles = []

    for semana_actual in semanas_a_procesar:
        print(f"\n--- Procesando Semana: {semana_actual} ---")

        resumen_semanal_actual = []
        resultados_segregacion_actual = []
        detalle_movimientos_actual = []

        directorio_datos_semanal = os.path.join(resultados_dir_script, "instancias_magdalena", semana_actual)
        os.makedirs(directorio_datos_semanal, exist_ok=True)

        archivo_instancia = os.path.join(
            directorio_datos_semanal,
            f"Instancia_{semana_actual}_{PARTICIPACION_C}_K.xlsx"
        )
        resultado_file_semana = os.path.join(
            resultados_magdalena_base_path,
            semana_actual,
            f"resultado_{semana_actual}_{PARTICIPACION_C}_K.xlsx"
        )
        
        pareto_csv_file_semana = os.path.join(
            resultados_magdalena_base_path,
            semana_actual,
            f"pareto_{semana_actual}_{PARTICIPACION_C}.csv"
        )
        
        
        resultado_distancias_file_semana = os.path.join(
            resultados_magdalena_base_path,
            semana_actual,
            f"Distancias_Modelo_{semana_actual}_{PARTICIPACION_C}.xlsx"
        )

        metrics_csv = os.path.join(resultados_dir_script, "metrics", "metrics_magdalena.csv")

        try:
            if not os.path.exists(archivo_instancia):
                print(f"ADVERTENCIA: Archivo de instancia no encontrado: {archivo_instancia}. Saltando semana.")
                continue

            df = pd.read_excel(archivo_instancia, sheet_name=None)
            segregacion_map = dict(zip(df['S']['S'], df['S']['Segregacion']))

            # ---------------------------
            # BUILD MODEL
            # ---------------------------
            model = ConcreteModel()

            model.B = Set(initialize=df["B"].iloc[:, 0].tolist())
            model.S = Set(initialize=df["S"].iloc[:, 0].tolist())
            model.T = Set(initialize=df["T"].iloc[:, 0].tolist())

            def _safe_sheet(name, col_name):
                _df = df.get(name, pd.DataFrame({col_name: []}))
                return _df[col_name].tolist()

            model.G   = Set(initialize=_safe_sheet("G",   "G"))
            model.GRT = Set(initialize=_safe_sheet("GRT", "GRT"))
            model.GRS = Set(initialize=_safe_sheet("GRS", "GRS"))

            model.BC = Set(initialize=_safe_sheet("BC", "BC"))
            model.BH = Set(initialize=_safe_sheet("BH", "BH"))
            model.BT = Set(initialize=_safe_sheet("BT", "BT"))
            model.BI = Set(initialize=_safe_sheet("BI", "BI"))

            model.B_E = Set(initialize=_safe_sheet("B_E", "B_E"))
            model.B_I = Set(initialize=_safe_sheet("B_I", "B_I"))

            # Params (Magdalena)
            model.C   = Param(model.B, initialize=df['C_b'].set_index('B')['C'].to_dict())
            model.VS  = Param(model.B, initialize=df['VS_b'].set_index('B')['VS'].to_dict())
            model.VSR = Param(model.B, initialize=df['VSR_b'].set_index('B')['VSR'].to_dict())
            model.KS  = Param(model.S, initialize=df['KS_s'].set_index('S')['KS'].to_dict())
            model.KI  = Param(model.S, initialize=df['KI_s'].set_index('S')['KI'].to_dict())

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

            model.LE  = Param(model.B, initialize=df['LE_b'].set_index('B')['LE'].to_dict())
            model.TEU = Param(model.S, initialize=df['TEU_s'].set_index('S')['TEU'].to_dict())
            model.OS  = Param(initialize=1, mutable=True)

            C_mediana = float(pd.Series(df['C_b']['C']).astype(float).median())
            es_pila = C_mediana <= 6.0
            if es_pila:
                OI_dict = {row.B: 0.2 for _, row in df['C_b'].iterrows()}
            else:
                OI_dict = {row.B: 1.0 / float(row.C) for _, row in df['C_b'].iterrows()}
            model.OI = Param(model.B, within=NonNegativeReals, initialize=OI_dict)

            model.r = Param(initialize=45)
            model.R = Param(model.S, initialize=df['R_s'].set_index('S')['R'].to_dict())

            # Params (Camila)
            df_PROD = df.get("PROD", pd.DataFrame({"Tipo": [], "Prod": []}))
            prod_map = {
                str(row["Tipo"]).strip().upper(): float(row["Prod"])
                for _, row in df_PROD.iterrows()
            }
            model.mu_RTG = Param(initialize=prod_map.get("RTG", 30.0), mutable=True)
            model.mu_RS  = Param(initialize=prod_map.get("RS",  20.0), mutable=True)

            df_W = df.get("W", pd.DataFrame({"W": [1.0]}))
            model.W = Param(initialize=float(df_W["W"].iloc[0]), mutable=True)

            df_Wb = df.get("W_b", pd.DataFrame({"B": [], "W_b": []}))
            if df_Wb.empty:
                Wb_map = {b: 3 for b in model.B}
            else:
                Wb_map = {row["B"]: float(row["W_b"]) for _, row in df_Wb.iterrows()}
            model.Wb = Param(model.B, initialize=Wb_map, mutable=True)

            df_Rmax_rtg = df.get("Rmax_rtg", pd.DataFrame({"Rmax_rtg": [len(list(model.GRT))]}))
            df_Rmax_rs  = df.get("Rmax_rs",  pd.DataFrame({"Rmax_rs":  [len(list(model.GRS))]}))
            model.RmaxRTG = Param(initialize=int(df_Rmax_rtg["Rmax_rtg"].iloc[0]), mutable=True)
            model.RmaxRS  = Param(initialize=int(df_Rmax_rs["Rmax_rs"].iloc[0]),  mutable=True)

            df_Kg = df.get("K_g", pd.DataFrame({"G": [], "K": []}))
            if df_Kg.empty:
                Kg_map = {g: (2 if g in list(model.GRT) else 1) for g in model.G}
            else:
                Kg_map = {row["G"]: int(row["K"]) for _, row in df_Kg.iterrows()}
            model.Kg = Param(model.G, initialize=Kg_map, mutable=True)

            df_CBR = df.get("CBR", pd.DataFrame({"b1": [], "b2": [], "CBR": []}))
            df_CBS = df.get("CBS", pd.DataFrame({"b1": [], "b2": [], "CBS": []}))
            if df_CBR.empty:
                CBR_init = {(b1, b2): 1 for b1 in model.B for b2 in model.B}
            else:
                CBR_init = {(row["b1"], row["b2"]): int(row["CBR"]) for _, row in df_CBR.iterrows()}
            if df_CBS.empty:
                CBS_init = {(b1, b2): 1 for b1 in model.B for b2 in model.B}
            else:
                CBS_init = {(row["b1"], row["b2"]): int(row["CBS"]) for _, row in df_CBS.iterrows()}
            model.CBR = Param(model.B, model.B, initialize=lambda m,b1,b2: CBR_init.get((b1,b2), 1), mutable=True)
            model.CBS = Param(model.B, model.B, initialize=lambda m,b1,b2: CBS_init.get((b1,b2), 1), mutable=True)

            df_EX_RTG = df.get("EX_RTG", pd.DataFrame({"b1": [], "b2": [], "EX": []}))
            if df_EX_RTG.empty:
                EXRTG_init = {(b1, b2): 2 for b1 in model.B for b2 in model.B}
            else:
                EXRTG_init = {(row["b1"], row["b2"]): int(row["EX"]) for _, row in df_EX_RTG.iterrows()}
            model.EX_RTG = Param(model.B, model.B, initialize=lambda m,b1,b2: EXRTG_init.get((b1,b2), 2), mutable=True)

            df_Cbs = df.get("C_bs", pd.DataFrame({"B": [], "S": [], "Cbs": []}))
            Cbs_dict = {(row["B"], row["S"]): int(row["Cbs"]) for _, row in df_Cbs.iterrows()}
            model.Cbs = Param(model.B, model.S, initialize=Cbs_dict, default=0, mutable=True)

            # eps_D parameter (mutable so we can sweep it)
            model.eps_D = Param(initialize=0.0, mutable=True)

            # Vars
            model.fr = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.fc = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.fd = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.fe = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)

            model.y  = Var(model.S, model.B, model.T, domain=Binary, initialize=0)
            model.u  = Var(model.S, model.B, domain=Binary, initialize=0)

            model.v  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.k  = Var(model.S, domain=NonNegativeIntegers, initialize=0)

            model.i  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.w  = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0)
            model.p  = Var(model.T, domain=NonNegativeIntegers, initialize=0)
            model.q  = Var(model.T, domain=NonNegativeIntegers, initialize=0)

            # Camila vars (as you had)
            #model.x_gbt = Var(model.G, model.B, model.T, domain=Binary, initialize=0)
            
            # Variable principal de asignación: Grúa g en Bloque b en Tiempo t
            model.ygbt = Var(model.G, model.B, model.T, domain=Binary, initialize=0)

            # Variable de inicio de asignación (para controlar permanencia mínima Kg)
            model.alpha_gbt = Var(model.G, model.B, model.T, domain=Binary, initialize=0)

            # Variable auxiliar de uso de bloque (para exclusividad espacial)
            model.Z_gb = Var(model.G, model.B, domain=Binary, initialize=0)
            model.z_gt  = Var(model.G, model.T, domain=Binary, initialize=0)
            model.nRTG  = Var(model.T, domain=NonNegativeIntegers, initialize=0)
            model.nRS   = Var(model.T, domain=NonNegativeIntegers, initialize=0)

            # Constraints (Magdalena) - same as you had
            model.constraint_2 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        if t == 1:
                            model.constraint_2.add(
                                model.i[s, b, t] == model.I0[s, b] + model.fr[s, b, t] + model.fd[s, b, t]
                                - model.fc[s, b, t] - model.fe[s, b, t]
                            )
                        else:
                            model.constraint_2.add(
                                model.i[s, b, t] == model.i[s, b, t-1] + model.fr[s, b, t] + model.fd[s, b, t]
                                - model.fc[s, b, t] - model.fe[s, b, t]
                            )

            model.constraint_3 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_3.add(model.i[s, b, t] <= model.v[s, b, t] * model.OS * model.C[b])

            model.constraint_4 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_4.add(
                            (model.v[s, b, t] - 1) * model.C[b] * model.OS + model.C[b] * model.OI[b]
                            <= model.i[s, b, t]
                        )

            model.constraint_5 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_5.add(sum(model.fr[s, b, t] for b in model.B) == model.DR[s, t])

            model.constraint_6 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_6.add(sum(model.fc[s, b, t] for b in model.B) == model.DC[s, t])

            model.constraint_7 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_7.add(sum(model.fd[s, b, t] for b in model.B) == model.DD[s, t])

            model.constraint_8 = ConstraintList()
            for t in model.T:
                for s in model.S:
                    model.constraint_8.add(sum(model.fe[s, b, t] for b in model.B) == model.DE[s, t])

            model.constraint_9 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_9.add(
                            model.fr[s, b, t] + model.fd[s, b, t]
                            <= (model.DR[s, t] + model.DD[s, t]) * model.y[s, b, t]
                        )

            model.constraint_10 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_10.add((model.fr[s, b, t] + model.fd[s, b, t]) >= model.y[s, b, t])

            model.constraint_11 = ConstraintList()
            for b in model.B:
                for s in model.S:
                    model.constraint_11.add(model.u[s, b] <= sum(model.y[s, b, t] for t in model.T))

            model.constraint_12 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    for s in model.S:
                        model.constraint_12.add(model.u[s, b] >= model.y[s, b, t])

            model.constraint_13 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_13.add(
                        sum(model.v[s, b, t] * model.TEU[s] for s in model.S) <= model.VS[b]
                    )

            model.constraint_14 = ConstraintList()
            for s in model.S:
                model.constraint_14.add(model.k[s] == sum(model.u[s, b] for b in model.B))

            model.constraint_15 = ConstraintList()
            for s in model.S:
                if sum(model.DR[s, t] for t in model.T) == 0 and sum(model.DD[s, t] for t in model.T) == 0:
                    model.constraint_15.add(model.k[s] == 0)
                else:
                    model.constraint_15.add(model.k[s] <= model.KS[s])

            model.constraint_16 = ConstraintList()
            for s in model.S:
                if sum(model.DR[s, t] for t in model.T) == 0 and sum(model.DD[s, t] for t in model.T) == 0:
                    model.constraint_16.add(model.k[s] == 0)
                else:
                    model.constraint_16.add(model.k[s] >= model.KI[s])

            model.constraint_17 = ConstraintList()
            model.constraint_18 = ConstraintList()
            model.constraint_19 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_17.add(
                        model.w[b, t] == sum(
                            model.fr[s, b, t] + model.fc[s, b, t] + model.fd[s, b, t] + model.fe[s, b, t]
                            for s in model.S
                        )
                    )
                for b in model.B:
                    model.constraint_18.add(model.p[t] >= model.w[b, t])
                    model.constraint_19.add(model.q[t] <= model.w[b, t])

            model.constraint_20 = ConstraintList()
            for t in model.T:
                model.constraint_20.add(model.p[t] - model.q[t] <= model.r)

            model.constraint_21 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_21.add(
                        sum(model.v[s, b, t] * model.R[s] for s in model.S) <= model.VSR[b]
                    )

            # Camila constraints (as you had)
            def nRTG_def_rule(m, t):
                return m.nRTG[t] == sum(m.z_gt[g, t] for g in m.GRT)
            model.constr_nRTG_def = Constraint(model.T, rule=nRTG_def_rule)

            def nRS_def_rule(m, t):
                return m.nRS[t] == sum(m.z_gt[g, t] for g in m.GRS)
            model.constr_nRS_def = Constraint(model.T, rule=nRS_def_rule)

            def nRTG_cap_rule(m, t):
                return m.nRTG[t] <= m.RmaxRTG
            model.constr_nRTG_cap = Constraint(model.T, rule=nRTG_cap_rule)

            def nRS_cap_rule(m, t):
                return m.nRS[t] <= m.RmaxRS
            model.constr_nRS_cap = Constraint(model.T, rule=nRS_cap_rule)

            def link_xz_rule(m, g, b, t):
                return m.ygbt[g, b, t] <= m.z_gt[g, t]
            model.constr_link_xz = Constraint(model.G, model.B, model.T, rule=link_xz_rule)

            def Wb_block_rule(m, b, t):
                return sum(m.ygbt[g, b, t] for g in m.G) <= m.Wb[b]
            model.constr_Wb_block = Constraint(model.B, model.T, rule=Wb_block_rule)

            def one_block_per_crane_rule(m, g, t):
                return sum(m.ygbt[g, b, t] for b in m.B) <= 1
            model.constr_one_block_per_crane = Constraint(model.G, model.T, rule=one_block_per_crane_rule)

            def workload_served_rule(m, b, t):
                return m.w[b, t] <= (
                    m.mu_RTG * sum(m.ygbt[g, b, t] for g in m.GRT)
                    + m.mu_RS  * sum(m.ygbt[g, b, t] for g in m.GRS)
                )
            model.constr_workload_served = Constraint(model.B, model.T, rule=workload_served_rule)

            # Expressions (objectives)
            def D_expr_rule(m):
                return (
                    sum(m.fc[s, b, t] * m.LC[s, b] for b in m.B for s in m.S for t in m.T)
                    + sum(m.fe[s, b, t] * m.LE[b]     for b in m.B for s in m.S for t in m.T)
                )
            model.D = Expression(rule=D_expr_rule)

            def B_balance_rule(m):
                return sum(m.p[t] - m.q[t] for t in m.T)
            model.B_balance = Expression(rule=B_balance_rule)

            # epsilon constraint (start disabled; we enable for pareto sweep)
            model.constr_epsD = Constraint(expr=model.D <= model.eps_D)
            model.constr_epsD.deactivate()

            # Two objectives; we toggle activate/deactivate
            model.obj_D = Objective(expr=model.D, sense=minimize)
            model.obj_B = Objective(expr=model.B_balance, sense=minimize)
            model.obj_D.deactivate()
            model.obj_B.deactivate()

            # ---------------------------
            # SOLVER SETUP
            # ---------------------------
            solver = SolverFactory('gurobi')
            solver.options.update({
                'LogToConsole': 0,
                'LogFile': os.path.join(directorio_datos_semanal, f'gurobi_log_{semana_actual}.log'),
                'MIPGap': 1e-3,
                'FeasibilityTol': 1e-5,
                'OptimalityTol': 1e-8,
                'IntFeasTol': 1e-5,
                'TimeLimit': 30,
                'MIPFocus': 0,
                'Heuristics': 0.5,
                'PumpPasses': 20,
                #'SolutionLimit': 1,
            })

            def solve_with_objective(which: str, use_eps: bool, eps_val: float | None):
                # Toggle objectives
                model.obj_D.deactivate()
                model.obj_B.deactivate()
                
                if which in ("D", "B") and not use_eps:
                    solver.options["TimeLimit"] = 120   # anclas más fuertes
                else:
                    solver.options["TimeLimit"] = 20    # sweep corto
                            
                if which == "D":
                    model.obj_D.activate()
                elif which == "B":
                    model.obj_B.activate()
                else:
                    raise ValueError("which must be 'D' or 'B'")

                # Toggle epsilon constraint
                model.constr_epsD.deactivate()
                if use_eps:
                    if eps_val is None:
                        raise ValueError("eps_val required when use_eps=True")
                    model.eps_D.set_value(float(eps_val))
                    model.constr_epsD.activate()

                t0 = time.perf_counter()
                res = solver.solve(model, tee=True, load_solutions=True)
                t1 = time.perf_counter()

                return res, (t1 - t0)

            # ---------------------------
            # PARETO (24h)
            # ---------------------------
            pareto_rows = []

            if pareto_enabled:
                # 1) D_min
                logger.info("Pareto: resolviendo D_min (minimizar D)")
                res_Dmin, sec_Dmin = solve_with_objective("D", use_eps=False, eps_val=None)
                if not _has_incumbent(res_Dmin, model):
                    raise RuntimeError("No se pudo obtener incumbente para D_min")

                D_min = float(value(model.D))
                B_at_Dmin = float(value(model.B_balance))
                logger.info("Pareto: D_min=%.6f, B@Dmin=%.6f", D_min, B_at_Dmin)

                pareto_rows.append({
                    "point_type": "anchor",
                    "which": "D_min",
                    "eps_D": None,
                    "D_x": D_min,
                    "B_x": B_at_Dmin,
                    "slack_epsD": None,
                    "solver_status": str(res_Dmin.solver.status),
                    "termination": str(res_Dmin.solver.termination_condition),
                    "solve_seconds": sec_Dmin,
                })

                # 2) B_min
                logger.info("Pareto: resolviendo B_min (minimizar B)")
                res_Bmin, sec_Bmin = solve_with_objective("B", use_eps=False, eps_val=None)
                if not _has_incumbent(res_Bmin, model):
                    raise RuntimeError("No se pudo obtener incumbente para B_min")

                B_min = float(value(model.B_balance))
                D_at_Bmin = float(value(model.D))
                logger.info("Pareto: B_min=%.6f, D@Bmin=%.6f", B_min, D_at_Bmin)

                pareto_rows.append({
                    "point_type": "anchor",
                    "which": "B_min",
                    "eps_D": None,
                    "D_x": D_at_Bmin,
                    "B_x": B_min,
                    "slack_epsD": None,
                    "solver_status": str(res_Bmin.solver.status),
                    "termination": str(res_Bmin.solver.termination_condition),
                    "solve_seconds": sec_Bmin,
                })

                # 3) Sweep eps_D from D_min*(1+pad) up to D_at_Bmin (or a bit above)
                eps_lo = D_min
                eps_hi = D_at_Bmin * (1.0 + float(pareto_pad))

                if pareto_points < 2:
                    eps_grid = [eps_hi]
                else:
                    ratio = eps_hi / eps_lo if eps_lo > 0 else 1.0
                    eps_grid = [eps_lo * (ratio ** (i/(pareto_points-1))) for i in range(pareto_points)]


                logger.info("Pareto: barrido eps_D en [%0.3f, %0.3f] con %d puntos", eps_lo, eps_hi, len(eps_grid))
                
                best_B = float("inf")
                no_improve = 0
                stop_after = 3      # corta si no mejora en 3 eps seguidos
                tol_B = 1e-6        # tolerancia para comparar B
                
                for j, eps in enumerate(eps_grid, start=1):
                    logger.info("Pareto %d/%d: min B s.a. D<=eps_D (eps_D=%.6f)", j, len(eps_grid), eps)
                    res_eps, sec_eps = solve_with_objective("B", use_eps=True, eps_val=eps)

                    if not _has_incumbent(res_eps, model):
                        pareto_rows.append({
                            "point_type": "sweep",
                            "which": "B|epsD",
                            "eps_D": float(eps),
                            "D_x": None,
                            "B_x": None,
                            "slack_epsD": None,
                            "solver_status": str(res_eps.solver.status),
                            "termination": str(res_eps.solver.termination_condition),
                            "solve_seconds": sec_eps,
                        })
                        continue

                    D_x = float(value(model.D))
                    B_x = float(value(model.B_balance))
                    
                    if B_x < best_B - tol_B:
                        best_B = B_x
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= stop_after:
                            logger.info("Pareto: stop temprano (B no mejora en %d puntos seguidos).", stop_after)
                            # igual guardamos este punto y cortamos
                    
                    
                    slack = float(eps) - float(value(model.D))
                    pareto_rows.append({
                        "point_type": "sweep",
                        "which": "B|epsD",
                        "eps_D": float(eps),
                        "D_x": D_x,
                        "B_x": B_x,
                        "slack_epsD": slack,
                        "solver_status": str(res_eps.solver.status),
                        "termination": str(res_eps.solver.termination_condition),
                        "solve_seconds": sec_eps,
                    })
                    
                    if no_improve >= stop_after:
                        break

                # After sweep, keep last solution as "current" for downstream outputs
                # If you want, you could pick best feasible; for now pick last feasible sweep result
                feasible_sweeps = [r for r in pareto_rows if r["point_type"] == "sweep" and r["D_x"] is not None]
                if feasible_sweeps:
                    chosen = feasible_sweeps[-1]
                    model.eps_D.set_value(float(chosen["eps_D"]))
                    model.constr_epsD.activate()
                    model.obj_D.deactivate()
                    model.obj_B.activate()
                    # model already has values from last solve, so ok
                else:
                    # fallback: keep B_min solution
                    model.constr_epsD.deactivate()
                    model.obj_D.deactivate()
                    model.obj_B.activate()

            else:
                # Single run (your old behavior): minimize B with no eps
                res_single, sec_single = solve_with_objective("B", use_eps=False, eps_val=None)
                if not _has_incumbent(res_single, model):
                    raise RuntimeError("No incumbente en corrida single")

            # ---------------------------
            # If infeasible, dump IIS, etc (keep your logic)
            # Note: Pareto sweep already handled "no incumbente" per point;
            # Here we only care about the final "current" model state.
            # ---------------------------

            # Distancias para reportes (desde el estado actual del modelo)
            distancia_expo = sum(value(model.fc[s, b, t]) * value(model.LC[s, b]) for b in model.B for s in model.S for t in model.T)
            distancia_impo = sum(value(model.fe[s, b, t]) * value(model.LE[b])     for b in model.B for s in model.S for t in model.T)

            distancia_expo_por_seg = {
                s: sum(value(model.fc[s, b, t]) * value(model.LC[s, b]) for b in model.B for t in model.T)
                for s in model.S
            }
            distancia_impo_por_seg = {
                s: sum(value(model.fe[s, b, t]) * value(model.LE[b]) for b in model.B for t in model.T)
                for s in model.S
            }

            movimientos_dlvr_por_seg = {s: sum(value(model.fe[s, b, t]) for b in model.B for t in model.T) for s in model.S}
            movimientos_load_por_seg = {s: sum(value(model.fc[s, b, t]) for b in model.B for t in model.T) for s in model.S}

            distancia_load_total = sum(distancia_expo_por_seg.values())
            distancia_dlvr_total = sum(distancia_impo_por_seg.values())

            # Resumen semanal (del estado final)
            eps_val_final = float(value(model.eps_D)) if model.constr_epsD.active else None
            resumen_semanal_actual.append({
                'Semana': semana_actual,
                'Eps_D': eps_val_final,
                'Distancia Total (D)': float(value(model.D)),
                'Distancia LOAD': float(distancia_load_total),
                'Distancia DLVR': float(distancia_dlvr_total),
                'B(x) = Σ_t (p_t - q_t)': float(value(model.B_balance)),
                'Movimientos_DLVR': float(sum(movimientos_dlvr_por_seg.values())),
                'Movimientos_LOAD': float(sum(movimientos_load_por_seg.values()))
            })

            for s in model.S:
                resultados_segregacion_actual.append({
                    'Semana': semana_actual,
                    'Segregacion': segregacion_map[s],
                    'Distancia_Total': float(distancia_expo_por_seg[s] + distancia_impo_por_seg[s]),
                    'Distancia_DLVR': float(distancia_impo_por_seg[s]),
                    'Distancia_LOAD': float(distancia_expo_por_seg[s]),
                    'Movimientos_DLVR': float(movimientos_dlvr_por_seg[s]),
                    'Movimientos_LOAD': float(movimientos_load_por_seg[s])
                })

            for s in model.S:
                for b in model.B:
                    movimientos_dlvr = sum(value(model.fe[s, b, t]) for t in model.T)
                    movimientos_load = sum(value(model.fc[s, b, t]) for t in model.T)
                    if movimientos_dlvr > 0 or movimientos_load > 0:
                        detalle_movimientos_actual.append({
                            'Semana': semana_actual,
                            'Segregacion': segregacion_map[s],
                            'Bloque': b,
                            'Movimientos DLVR': float(movimientos_dlvr),
                            'Movimientos LOAD': float(movimientos_load)
                        })

            # Extract variables
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

            df_pareto = pd.DataFrame(pareto_rows) if pareto_rows else pd.DataFrame()
            
            if not df_pareto.empty:
                df_pareto.to_csv(pareto_csv_file_semana, index=False)


            # contenedores turno-bloque
            datos_turno_bloque = []
            for t in model.T:
                for b in model.B:
                    total_contenedores = sum(value(model.i[s, b, t]) for s in model.S)
                    datos_turno_bloque.append({'Turno': t, 'Bloque': b, 'Contenedores': total_contenedores})
            df_turno_bloque = pd.DataFrame(datos_turno_bloque)
            df_pivot_turno_bloque = df_turno_bloque.pivot(index='Turno', columns='Bloque', values='Contenedores').fillna(0)

            # Save Excel
            with pd.ExcelWriter(resultado_file_semana, engine='openpyxl') as writer:
                if not df_pareto.empty:
                    df_pareto.to_excel(writer, sheet_name="Pareto", index=False)
                df_fr.to_excel(writer, sheet_name="Recibir", index=False)
                df_fc.to_excel(writer, sheet_name="Cargar", index=False)
                df_fd.to_excel(writer, sheet_name="Descargar", index=False)
                df_fe.to_excel(writer, sheet_name="Entregar", index=False)
                df_f.to_excel(writer, sheet_name="Flujos", index=False)
                df_y.to_excel(writer, sheet_name="Asignado", index=False)
                df_k.to_excel(writer, sheet_name="Total bloques", index=False)
                df_i.to_excel(writer, sheet_name="Volumen bloques (TEUs)", index=False)
                df_v.to_excel(writer, sheet_name="Bahías por bloques", index=False)
                df_w.to_excel(writer, sheet_name="Workload bloques", index=False)
                df_pq.to_excel(writer, sheet_name="Carga máx-min", index=False)
                df_pivot_turno_bloque.to_excel(writer, sheet_name="Contenedores Turno-Bloque", index=True)

            print(f"Resultados principales para {semana_actual} guardados en {resultado_file_semana}")

        except Exception as e:
            print(f"Error procesando semana {semana_actual}: Error - {str(e)}")
            continue

        # Save distancias resumen
        df_resumen = pd.DataFrame(resumen_semanal_actual)
        df_seg = pd.DataFrame(resultados_segregacion_actual)
        df_det = pd.DataFrame(detalle_movimientos_actual)

        try:
            with pd.ExcelWriter(resultado_distancias_file_semana, engine='openpyxl') as writer:
                df_resumen.to_excel(writer, sheet_name='Resumen Semanal', index=False)
                df_seg.to_excel(writer, sheet_name='Resultados por Segregación', index=False)
                df_det.to_excel(writer, sheet_name='Detalle de Movimientos', index=False)
            print(f"Resumen de distancias para {semana_actual} guardado en {resultado_distancias_file_semana}")
        except Exception as e:
            print(f"Error al guardar resumen de distancias para {semana_actual}: {str(e)}")

        # Telemetry: if pareto enabled, write last feasible sweep point (or B_min)
        if pareto_enabled and pareto_rows:
            feasible = [r for r in pareto_rows if r.get("D_x") is not None and r.get("B_x") is not None]
            chosen = feasible[-1] if feasible else pareto_rows[0]
            meta = {
                "modelo": "magdalena",
                "semana": semana_actual,
                "participacion": str(PARTICIPACION_C),
                "fase": "final",
                "resultado_xlsx": resultado_file_semana,
                "resultado_distancias": resultado_distancias_file_semana,
                "D_x": float(chosen["D_x"]) if chosen.get("D_x") is not None else None,
                "B_x": float(chosen["B_x"]) if chosen.get("B_x") is not None else None,
                "eps_D": float(chosen["eps_D"]) if chosen.get("eps_D") is not None else None,
            }
        else:
            meta = {
                "modelo": "magdalena",
                "semana": semana_actual,
                "participacion": str(PARTICIPACION_C),
                "fase": "final",
                "resultado_xlsx": resultado_file_semana,
                "resultado_distancias": resultado_distancias_file_semana,
                "D_x": float(value(model.D)),
                "B_x": float(value(model.B_balance)),
                "eps_D": None,
            }

        # objective_value_safe will read "obj_B" or "obj_D"? we keep old naming fallback
        # We'll just log B_balance as objective for consistency
        obj_val = float(value(model.B_balance))
        row = telemetry_pack(model, meta=meta, solve_elapsed=0.0, res=None, objective=obj_val)
        append_metrics_row(metrics_csv, row)

    print("\nProceso completado para todas las semanas.")

    semanas_filtradas = [s for s in semanas_a_procesar if s not in semanas_infactibles]

    print("\nsemanas_a_procesar = [")
    for s in semanas_filtradas:
        print(f'    "{s}",')
    print("]")

    return semanas_filtradas, semanas_infactibles
