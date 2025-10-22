#!/usr/bin/env python
# coding: utf-8

import os
import logging
import sys
import pandas as pd
from pyomo.opt import SolverStatus 
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, NonNegativeIntegers, Binary, NonNegativeReals, maximize,
    SolverFactory, TerminationCondition, value
)
from pyomo.contrib.iis import write_iis

logger = logging.getLogger("camila_maxmin")

def _swap_ext(path, new_ext):
    base, _ = os.path.splitext(path)
    return base + new_ext

def _has_solution(model):
    try:
        _ = value(model.obj)  # intenta evaluar el objetivo
        return True
    except:
        return False

def _dump_infeasible_artifacts(model, xlsx_path, logger=None):
    ilp_path = _swap_ext(xlsx_path, ".ilp")  # mismo nombre, extensión .ilp
    iis_path = _swap_ext(xlsx_path, ".iis")

    # Volcar LP (LP format; usas .ilp por convenio)
    model.write(ilp_path, format="lp", io_options={'symbolic_solver_labels': True})
    # IIS (diagnóstico)
    try:
        write_iis(model, iis_path, solver="gurobi")
    except Exception as e:
        if logger: logger.warning(f"No se pudo escribir IIS: {e}")

    # Borra un .xlsx viejo si existía
    try:
        if os.path.exists(xlsx_path):
            os.remove(xlsx_path)
    except Exception as e:
        if logger: logger.warning(f"No se pudo eliminar {xlsx_path}: {e}")

    if logger: logger.error(f"Modelo infactible/aborted sin incumbente. Artefactos: {ilp_path}, {iis_path}")


def ejecutar_instancias_gruas_maxmin(semanas, turnos, participacion, base_instancias, base_resultados):
    for semana in semanas:
        out_dir = os.path.join(base_resultados, f"resultados_turno_{semana}")
        os.makedirs(out_dir, exist_ok=True)
        
        for turno in turnos:
            logger.info(f"--- INICIANDO TURNO {turno} / SEMANA {semana} ---")
            datos = pd.read_excel(
                os.path.join(base_instancias,
                             f"instancias_turno_{semana}",
                             f"Instancia_{semana}_{participacion}_T{turno}.xlsx"),
                sheet_name=None
            )
            
            # -------------------------
            # Modelo
            # -------------------------
            m = ConcreteModel()
            
            # Conjuntos
            m.G   = Set(initialize=[r['G']   for r in datos['G'].to_dict('records')])
            m.B   = Set(initialize=[r['B']   for r in datos['B'].to_dict('records')])
            m.B_I = Set(initialize=[r['B_I'] for r in datos['B_I'].to_dict('records')])
            m.B_E = Set(initialize=[r['B_E'] for r in datos['B_E'].to_dict('records')])
            m.T   = Set(initialize=[r['T']   for r in datos['T'].to_dict('records')])
            m.S   = Set(initialize=[r['S']   for r in datos['S'].to_dict('records')])
            m.S_E = Set(initialize=[r['S_E'] for r in datos['S_E'].to_dict('records')])
            m.S_I = Set(initialize=[r['S_I'] for r in datos['S_I'].to_dict('records')])
            
            # Parámetros matriciales
            m.AEbs = Param(
                m.B, m.S,
                initialize={(r['B_E'], r['S_E']): r['AEbs'] for r in datos['AEbs'].to_dict('records')},
                default=0, mutable=True
            )
            m.AIbs = Param(
                m.B, m.S,
                initialize={(r['B_I'], r['S_I']): r['AIbs'] for r in datos['AIbs'].to_dict('records')},
                default=0, mutable=True
            )
            m.EIbs = Param(
                m.B, m.S,
                initialize={(r['B_I'], r['S_I']): r['EIbs'] for r in datos['EIbs'].to_dict('records')},
                default=0, mutable=True
            )
            m.Gs    = Param(
                m.S_E,
                initialize={r['S_E']: r['Gs'] for r in datos['Gs'].to_dict('records')},
                default=0, mutable=True
            )
            m.DMEst = Param(
                m.S_E, m.T,
                initialize={(r['S_E'], r['T']): r['DMEst'] for r in datos['DMEst'].to_dict('records')},
                default=0, mutable=True
            )
            m.DMIst = Param(
                m.S_I, m.T,
                initialize={(r['S_I'], r['T']): r['DMIst'] for r in datos['DMIst'].to_dict('records')},
                default=0, mutable=True
            )
            m.Cbs   = Param(
                m.B, m.S,
                initialize={(r['B'], r['S']): r['Cbs'] for r in datos['Cbs'].to_dict('records')},
                default=0, mutable=True
            )
            
            # Parámetros escalares
            m.mu   = Param(initialize=datos['mu' ].iloc[0,0], mutable=True)
            m.W    = Param(initialize=datos['W'  ].iloc[0,0], mutable=True)
            m.K    = Param(initialize=datos['K'  ].iloc[0,0], mutable=True)
            m.Rmax = Param(initialize=datos['Rmax'].iloc[0,0], mutable=True)
            
            # Exclusividad de bloques
            
            adyac_no_exc_b = {
                ('b1','b3'), ('b1','b1'),
                ('b2','b4'), ('b2','b2'),
                ('b3','b3'), ('b6','b3'),
                ('b4','b4'), ('b7','b4'),
                ('b5','b5'), ('b5','b8'),
                ('b6','b6'),
                ('b7','b7'),
                ('b8','b8'),
                ('b9','b9'),
            }
            
            adyac_no_exc_t = {
                ('t1','t1'), ('t2','t2'), ('t3','t3'), ('t4','t4'),
                ('t1','t2'), ('t3','t4'),
            }
            
            adyac_no_exc_h = {
                ('h1','h1'), ('h2','h2'), ('h3','h3'),('h4','h4'),('h5','h5'),
                ('h1','h2'), ('h2','h3'), ('h3','h4'), ('h4','h5'),
            }
            
            adyac_no_exc_i = {
                ('i1','i1'), ('i2','i2'),
                ('i1','i2'),
            }
            
            def _fam(x: str) -> str:
                return str(x)[0].lower() if x is not None else ''
            
            _fam_to_set = {
                'b': adyac_no_exc_b,
                't': adyac_no_exc_t,
                'h': adyac_no_exc_h,
                'i': adyac_no_exc_i,
            }
            
            def init_ex(m, b1, b2):
                if b1 == b2:
                    return 2
                f1, f2 = _fam(b1), _fam(b2)
                if f1 != f2:
                    return 1
                tabla = _fam_to_set.get(f1, set())
                if (b1, b2) in tabla or (b2, b1) in tabla:
                    return 2
                return 1
            
            m.ex = Param(m.B, m.B, initialize=init_ex, mutable=True)
            
            
            # Variables
            m.fc_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fd_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fr_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fe_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.ygbt      = Var(m.G, m.B, m.T, domain=Binary)
            m.alpha_gbt = Var(m.G, m.B, m.T, domain=Binary)
            m.Z_gb      = Var(m.G, m.B,      domain=Binary)
            m.min_diff_val = Var(domain=NonNegativeReals, name="min_diff_val")
            
            # ----------------------------------------------------------------
            # 1) “Forzar cero” fuera de dominios (bloques vs. flujos y segregaciones)
            # ----------------------------------------------------------------
            m.bloque_I = ConstraintList()
            for b in m.B:
                for t in m.T:
                    if b not in m.B_I:
                        for s in m.S_I:
                            m.bloque_I.add(m.fd_sbt[s,b,t] == 0)
                            m.bloque_I.add(m.fe_sbt[s,b,t] == 0)
                    if b not in m.B_E:
                        for s in m.S_E:
                            m.bloque_I.add(m.fc_sbt[s,b,t] == 0)
                            m.bloque_I.add(m.fr_sbt[s,b,t] == 0)
            
            m.seg_I = ConstraintList()
            for b in m.B:
                for t in m.T:
                    for s in m.S:
                        if s not in m.S_I:
                            m.seg_I.add(m.fd_sbt[s,b,t] == 0)
                            m.seg_I.add(m.fe_sbt[s,b,t] == 0)
                        if s not in m.S_E:
                            m.seg_I.add(m.fc_sbt[s,b,t] == 0)
                            m.seg_I.add(m.fr_sbt[s,b,t] == 0)
            
            # -------------------------
            # 2) Demanda por turno
            # -------------------------
            def dem_carga(m, s, t):
                if s in m.S_E:
                    return sum(m.fc_sbt[s,b,t] for b in m.B_E) == m.DMEst[s,t]
                return Constraint.Skip
            m.dem_carga = Constraint(m.S, m.T, rule=dem_carga)
            
            def dem_descarga(m, s, t):
                if s in m.S_I:
                    return sum(m.fd_sbt[s,b,t] for b in m.B_I) == m.DMIst[s,t]
                return Constraint.Skip
            m.dem_descarga = Constraint(m.S, m.T, rule=dem_descarga)
            
            # -------------------------
            # 3) Inventario final
            # -------------------------
            def dem_recibir(m, s):
                return sum(m.fr_sbt[s,b,t] for b in m.B_E for t in m.T) == m.Gs[s]
            m.dem_recibir = Constraint(m.S_E, rule=dem_recibir)
            """
            def plan_entregar(m, b, s):
                if s in m.S_I and b in m.B_I:
                    return sum(m.fe_sbt[s,b,t] for t in m.T) == m.EIbs[b,s]
                return Constraint.Skip
            m.plan_entregar  = Constraint(m.B, m.S, rule=plan_entregar)
            """
            
            def plan_entregar_total(m, s):
                if s in m.S_I:
                    total_turno = sum(m.EIbs[b, s] for b in m.B_I)
                    return sum(m.fe_sbt[s, b, t] for b in m.B_I for t in m.T) == total_turno
                return Constraint.Skip
            m.plan_entregar_total = Constraint(m.S, rule=plan_entregar_total)
            
            # ---------------------------------
            # 4) Restricciones de diferencia
            # ---------------------------------
            def diff_rule(m, b, t):
                carga     = sum(m.fc_sbt[s,b,t] + m.fr_sbt[s,b,t] for s in m.S_E)
                descarga  = sum(m.fd_sbt[s,b,t] + m.fe_sbt[s,b,t] for s in m.S_I)
                return m.mu * sum(m.ygbt[g,b,t] for g in m.G) - (carga + descarga) >= m.min_diff_val
            m.diff_constr = Constraint(m.B, m.T, rule=diff_rule)
            
            # -------------------------
            # 5) Objetivo
            # -------------------------
            m.obj = Objective(expr=m.min_diff_val, sense=maximize)
            
            # -------------------------
            # 6) Vincular Z y ygbt
            # -------------------------
            m.Z_y_up = ConstraintList()
            m.y_Z_up = ConstraintList()
            for g in m.G:
                for b in m.B:
                    m.Z_y_up.add(
                        m.Z_gb[g,b] <= sum(m.ygbt[g,b,t] for t in m.T)
                    )
                    m.y_Z_up.add(
                        sum(m.ygbt[g,b,t] for t in m.T) <= m.Z_gb[g,b] * len(m.T)
                    )
            
            # ---------------------------------
            # 7) Exclusividad entre bloques (Z)
            # ---------------------------------
            m.excl = ConstraintList()
            for g in m.G:
                for b1 in m.B:
                    for b2 in m.B:
                        if b1 != b2:
                            m.excl.add(
                                m.Z_gb[g,b1] + m.Z_gb[g,b2] <= m.ex[b1,b2]
                            )
            
            # -------------------------
            # 8) Capacidad por turno
            # -------------------------
            def cap_bloque(m, b, t):
                carg = sum(m.fc_sbt[s,b,t] + m.fr_sbt[s,b,t] for s in m.S_E)
                desc = sum(m.fd_sbt[s,b,t] + m.fe_sbt[s,b,t] for s in m.S_I)
                return carg + desc <= m.mu * sum(m.ygbt[g,b,t] for g in m.G)
            m.capacidad = Constraint(m.B, m.T, rule=cap_bloque)
            
            # ---------------------------------
            # 9) Inventario dinámico (min/max)
            # ---------------------------------
            def inv_min(m, b, s, t):
                inv = m.AEbs[b,s] + m.AIbs[b,s]
                for i in range(1, t+1):
                    if s in m.S_I:
                        inv += m.fd_sbt[s,b,i] - m.fe_sbt[s,b,i]
                    if s in m.S_E:
                        inv += m.fr_sbt[s,b,i] - m.fc_sbt[s,b,i]
                return inv >= 0
            m.inv_min = Constraint(m.B, m.S, m.T, rule=inv_min)
            
            def inv_max(m, b, s, t):
                inv = m.AEbs[b,s] + m.AIbs[b,s]
                for i in range(1, t+1):
                    if s in m.S_I:
                        inv += m.fd_sbt[s,b,i] - m.fe_sbt[s,b,i]
                    if s in m.S_E:
                        inv += m.fr_sbt[s,b,i] - m.fc_sbt[s,b,i]
                return inv <= m.Cbs[b,s]
            m.inv_max = Constraint(m.B, m.S, m.T, rule=inv_max)
            
            # -------------------------
            # 10) Exclusividad de grúas
            # -------------------------
            def one_block(m, g, t):
                return sum(m.ygbt[g,b,t] for b in m.B) <= 1
            m.one_block = Constraint(m.G, m.T, rule=one_block)
            
            def max_cranes(m, t):
                return sum(m.ygbt[g,b,t] for g in m.G for b in m.B) <= m.Rmax
            m.max_cranes = Constraint(m.T, rule=max_cranes)
            
            def max_collision(m, b, t):
                return sum(m.ygbt[g,b,t] for g in m.G) <= m.W
            m.max_collision = Constraint(m.B, m.T, rule=max_collision)
            
            # -------------------------------------------------
            # 11) Duración mínima: normal y tramo final
            # -------------------------------------------------
            def lb_constraint(m, g, b, t):
                K_int = int(value(m.K))
                if t <= max(m.T) - K_int + 1:
                    return K_int*m.alpha_gbt[g,b,t] <= sum(
                        m.ygbt[g,b,r] for r in m.T if r>=t and r< t+K_int
                    )
                return Constraint.Skip
            m.lb_constraint = Constraint(m.G, m.B, m.T, rule=lb_constraint)
            
            def lb1_constraint(m, g, b, t):
                K_int = int(value(m.K))
                if t > max(m.T) - K_int + 1:
                    return (max(m.T)-t+1)*m.alpha_gbt[g,b,t] <= sum(
                        m.ygbt[g,b,r] for r in m.T if r>=t
                    )
                return Constraint.Skip
            m.lb1_constraint = Constraint(m.G, m.B, m.T, rule=lb1_constraint)
            
            # -------------------------------------------------
            # 12) Upper‐bound (activaciones)
            # -------------------------------------------------
            def ub_constraint(m, g, b, t):
                if t > min(m.T):
                    return m.ygbt[g,b,t] <= m.ygbt[g,b,t-1] + m.alpha_gbt[g,b,t]
                return Constraint.Skip
            m.ub_constraint = Constraint(m.G, m.B, m.T, rule=ub_constraint)
            
            def ub1_constraint(m, g, b):
                t0 = min(m.T)
                return m.ygbt[g,b,t0] <= m.alpha_gbt[g,b,t0]
            m.ub1_constraint = Constraint(m.G, m.B, rule=ub1_constraint)
            
            # -------------------------------------------------
            # 13) No‐solapamiento de alphas
            # -------------------------------------------------
            m.alpha_nosolapa = ConstraintList()
            K_int = int(value(m.K))
            for g in m.G:
                for b in m.B:
                    for t in m.T:
                        for r in m.T:
                            if t < r < t + K_int:
                                m.alpha_nosolapa.add(
                                    m.alpha_gbt[g,b,t] <= 1 - m.alpha_gbt[g,b,r]
                                )
            
            # -------------------------
            # Solver
            # -------------------------
            solver = SolverFactory("gurobi")
            base = {
                #"TimeLimit":   1000,             # tu límite
                "Threads":     max(1, os.cpu_count()-1),  # usa n-1 hilos
                "Presolve":    2,                # presolve agresivo
                "Cuts":        2,                # cortes moderados-agresivos
                "Symmetry":    2,                # manejo agresivo de simetrías
                "Heuristics":  0.10,             # heurísticas moderadas
                "Seed":        42,               # reproducibilidad
                "Method":      2,                # dual simplex para LPs raíz (suele ser rápido)
                "LogToConsole": 1,
                # "LogFile":   os.path.join(out_dir, f"gurobi_{turno}.log"),
            }
            # Si tienes 4+ hilos, deja que pruebe estrategias en paralelo:
            if base["Threads"] >= 4:
                base["ConcurrentMIP"] = 1
            
            solver.options.update(base)
            
            # Ruta destino (xlsx / ilp / iis comparten la misma raíz)
            resultado_xlsx = os.path.join(out_dir, f"resultados_{semana}_{participacion}_T{turno}.xlsx")
            
            # PRIMER SOLVE: chequear factibilidad sin cargar solución
            res = solver.solve(m, tee=False, load_solutions=False)
            term = res.solver.termination_condition
            status = res.solver.status
            
            # Caso INFACTIBLE o INF-UNBD -> .ilp + .iis (NO Excel) y siguiente turno
            if term in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
                _dump_infeasible_artifacts(m, resultado_xlsx, logger)
                logger.info("Turno %s completado.", turno)
                continue
            
            # SEGUNDO SOLVE: cargar solución si existe
            res2 = solver.solve(m, tee=False, load_solutions=True)
            term2 = res2.solver.termination_condition
            status2 = res2.solver.status
            
            # Si TimeLimit/aborted y NO hay solución cargada -> .ilp + .iis (NO Excel)
            if (term2 in (TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations) or status2 in (SolverStatus.aborted, SolverStatus.unknown)) and not _has_solution(m):
                _dump_infeasible_artifacts(m, resultado_xlsx, logger)
                logger.info("Turno %s completado.", turno)
                continue
            
            # En cualquier otro caso (óptimo o TL con incumbente) -> exportar Excel
            df = []
            for v in m.component_objects(Var, active=True):
                for idx in v:
                    val = v[idx].value
                    if val:  # graba sólo valores no nulos
                        df.append({'var': v.name, 'idx': idx, 'val': val})
            
            pd.DataFrame(df).to_excel(resultado_xlsx, index=False)
            logger.info("Turno %s completado.", turno)