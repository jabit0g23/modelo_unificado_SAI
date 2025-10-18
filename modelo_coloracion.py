import math
from pyomo.environ import *
import logging, sys
from pyomo.opt import TerminationCondition
import pandas as pd
import os

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

def _has_incumbent(res):
    tc = res.solver.termination_condition
    st = res.solver.status

    # Si el solver adjuntó soluciones, úsalo como señal definitiva
    has_sol = hasattr(res, "solution") and res.solution and len(res.solution) > 0

    # Condiciones que NO tienen solución
    if tc in (TC.infeasible, TC.invalidProblem, TC.error, TC.solverFailure):
        return False

    # Acepta óptimo/feasible o límite de tiempo/recursos/interrupción con incumbente
    time_limited = tc in tuple(
        x for x in (
            getattr(TC, "maxTimeLimit", None),
            getattr(TC, "resourceInterrupt", None),
            getattr(TC, "userInterrupt", None),
        ) if x is not None
    )

    if tc in (TC.optimal, TC.feasible):
        return True

    if time_limited and has_sol:
        return True

    # Fallback razonable: si el solver quedó OK/Warning y hay solución adjunta
    return (st in (SolverStatus.ok, SolverStatus.warning)) and has_sol


def ejecutar_instancias_coloracion(
    semanas,
    participacion,
    resultados_dir,
    *,
    usar_cota_inferior=False,  # activa/desactiva cota inferior (equidad por turno)
    beta_alpha=0.8,            # beta para alpha[s] = beta / KS[s]
    usar_cota_superior=False,  # activa/desactiva cota superior dinámica (dispersión)
    gamma_val=0.2,             # gamma en (0,1]
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
    
        directorio_datos_semanal = os.path.join(BASE_DIR, "resultados_generados", "instancias_magdalena", semana_actual)
        
        # Asegurar que el directorio exista (principalmente para la salida, ya que la instancia debe existir)
        os.makedirs(directorio_datos_semanal, exist_ok=True)
    
        archivo_instancia = os.path.join(directorio_datos_semanal, f"Instancia_{semana_actual}_{PARTICIPACION_C}_K.xlsx")
        resultado_file_semana = os.path.join(resultados_magdalena_base_path, semana_actual, f"resultado_{semana_actual}_{PARTICIPACION_C}_K.xlsx")
        resultado_distancias_file_semana = os.path.join(resultados_magdalena_base_path, semana_actual, f"Distancias_Modelo_{semana_actual}_{PARTICIPACION_C}.xlsx")
    
        try:
            # Verificar si el archivo de instancia existe ANTES de intentar leerlo
            if not os.path.exists(archivo_instancia):
                print(f"ADVERTENCIA: Archivo de instancia no encontrado para la semana {semana_actual}: {archivo_instancia}. Saltando esta semana.")
                continue # Pasar a la siguiente semana
    
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
            
            # ► Preparación de parámetros para restricciones opcionales
            #   - TC[s,t] siempre que alguna cota esté activa
            if usar_cota_inferior or usar_cota_superior:
                TC_dict = {
                    (row['S'], row['T']): int(row['DR']) + int(row['DD'])
                    for _, row in df['D_params'].iterrows()
                }
                model.TC = Param(model.S, model.T, initialize=TC_dict, within=NonNegativeIntegers)
            
            #   - alpha[s] = beta / KS[s] si se usa cota inferior
            if usar_cota_inferior:
                ks_map = df['KS_s'].set_index('S')['KS'].to_dict()
                alpha_dict = {s: float(beta_alpha) / float(ks_map[s]) for s in df['S']['S']}
                model.alpha = Param(model.S, initialize=alpha_dict, within=Reals)
            
            #   - gamma[s] si se usa cota superior (flujo)
            #     Nota: forzamos gamma >= 1/KS[s] para que con a lo más KS[s] bloques
            #     se pueda repartir el 100% del flujo del turno (evita infactibilidad).
            if usar_cota_superior:
                ks_map = df['KS_s'].set_index('S')['KS'].to_dict()
                gamma_dict = {
                    s: max(float(gamma_val), 1.0 / float(ks_map[s]))
                    for s in df['S']['S']
                }
                model.gamma = Param(model.S, initialize=gamma_dict, within=Reals)

            
            model.LE = Param(model.B, initialize=df['LE_b'].set_index('B')['LE'].to_dict())
            model.TEU = Param(model.S, initialize=df['TEU_s'].set_index('S')['TEU'].to_dict())
            model.OS = Param(initialize=1, mutable=True)
            
            OI_dict = {row.B: 1.0/row.C for _, row in df['C_b'].iterrows()}
            model.OI = Param(model.B, within=PositiveReals, initialize=OI_dict)
            
            #model.OI = Param(initialize=0.0204081632653061)
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
                    return m.fr[s, b, t] + m.fd[s, b, t] >= math.ceil(rhs) * m.u[s, b]
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
    
            # Restricción (21)
            model.constraint_21 = ConstraintList()
            for t in model.T:
                for b in model.B:
                    model.constraint_21.add(
                        expr = sum(model.v[s, b, t] * model.TEU[s] * model.R[s] for s in model.S) <= model.VSR[b]
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
            
            solver = SolverFactory('gurobi')
            solver.options.update({
                'LogToConsole': 1,
                'LogFile': os.path.join(directorio_datos_semanal, f'gurobi_log_{semana_actual}.log'),
                'MIPGap': 1e-6,
                'FeasibilityTol': 1e-5,
                'OptimalityTol': 1e-8,
                'IntFeasTol': 1e-5,
                'TimeLimit': 250,
                'MIPFocus': 1,      # prioriza factibilidad
                'Heuristics': 0.5,  # más heurística
                'PumpPasses': 20,   # feasibility pump
            })
    
            res = solver.solve(model, tee=True, load_solutions=False)
            
            if res.solver.termination_condition == TerminationCondition.infeasible:
                logger.error("🚨 Infactible en %s: escribiendo LP + IIS…", semana_actual)
                results_dir_semana = os.path.join(resultados_magdalena_base_path, semana_actual)
                lp_path  = os.path.join(results_dir_semana, f"modelo_inf_{semana_actual}.lp")
                iis_path = os.path.join(results_dir_semana, f"modelo_inf_{semana_actual}.iis")
                model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})
                write_iis(model, iis_path, solver="gurobi")
                semanas_infactibles.append(semana_actual)
                continue
            
            if not _has_incumbent(res):
                logger.error("⛔ %s terminó sin incumbente. Guardo LP y sigo.", semana_actual)
                results_dir_semana = os.path.join(resultados_magdalena_base_path, semana_actual)
                lp_path = os.path.join(results_dir_semana, f"modelo_{semana_actual}_nolb.lp")
                model.write(lp_path, format="lp", io_options={'symbolic_solver_labels': True})
                continue
    
            model.solutions.load_from(res)
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
    
    print("\nProceso completado para todas las semanas.")
    
    semanas_filtradas = [s for s in semanas_a_procesar 
                         if s not in semanas_infactibles]
    
    # Imprimimos en el formato literal Python que pedías
    print("\nsemanas_a_procesar = [")
    for s in semanas_filtradas:
        print(f'    "{s}",')
    print("]")
    
    return semanas_filtradas, semanas_infactibles