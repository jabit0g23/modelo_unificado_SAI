#!/usr/bin/env python
# coding: utf-8

import os
import logging
import sys
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, NonNegativeIntegers, Binary, NonNegativeReals, minimize,
    SolverFactory, TerminationCondition, value
)
from pyomo.contrib.iis import write_iis

logger = logging.getLogger("camila_minmax")

def ejecutar_instancias_gruas_minmax(semanas, turnos, participacion, base_instancias, base_resultados):
    for semana in semanas:
        out_dir = os.path.join(base_resultados, f"resultados_turno_{semana}")
        os.makedirs(out_dir, exist_ok=True)

        for turno in turnos:
            logger.info(f"--- [MINMAX] INICIANDO TURNO {turno} / SEMANA {semana} ---")
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
            m.mu    = Param(initialize=datos['mu' ].iloc[0,0], mutable=True)
            m.W     = Param(initialize=datos['W'  ].iloc[0,0], mutable=True)
            m.K     = Param(initialize=datos['K'  ].iloc[0,0], mutable=True)
            m.Rmax  = Param(initialize=datos['Rmax'].iloc[0,0], mutable=True)

            # Exclusividad de bloques
            adyac_no_exc = {
                ('b1','b3'),('b1','b1'),('b2','b4'),('b2','b2'),
                ('b3','b3'),('b6','b3'),('b4','b4'),('b5','b5'),
                ('b6','b6'),('b7','b7'),('b8','b8'),('b9','b9'),
                ('b7','b4'),('b5','b8')
            }
            def init_ex(m, b1, b2):
                return 2 if (b1,b2) in adyac_no_exc or (b2,b1) in adyac_no_exc or b1==b2 else 1
            m.ex = Param(m.B, m.B, initialize=init_ex, mutable=True)

            # Variables
            m.fc_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fd_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fr_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fe_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.ygbt      = Var(m.G, m.B, m.T, domain=Binary)
            m.alpha_gbt = Var(m.G, m.B, m.T, domain=Binary)
            m.Z_gb      = Var(m.G, m.B,      domain=Binary)
            m.max_diff  = Var(domain=NonNegativeReals, name="max_diff")

            # ----------------------------------------------------------------
            # 1) “Forzar cero” fuera de dominios
            # ----------------------------------------------------------------
            m.bloque_I = ConstraintList()
            m.seg_I    = ConstraintList()
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
            m.dem_carga    = Constraint(m.S, m.T, rule=dem_carga)

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

            def inv_entregar(m, b, s):
                if s in m.S_I and b in m.B_I:
                    return sum(m.fe_sbt[s,b,t] for t in m.T) == m.AIbs[b,s]
                return Constraint.Skip
            m.inv_entregar = Constraint(m.B, m.S, rule=inv_entregar)

            # ---------------------------------
            # 4) Restricciones de diferencia
            # ---------------------------------
            def diff_rule(m, b, t):
                carga     = sum(m.fc_sbt[s,b,t] + m.fr_sbt[s,b,t] for s in m.S_E)
                descarga  = sum(m.fd_sbt[s,b,t] + m.fe_sbt[s,b,t] for s in m.S_I)
                # aquí cambia >= min_diff_val por <= max_diff
                return m.mu * sum(m.ygbt[g,b,t] for g in m.G) - (carga + descarga) <= m.max_diff
            m.diff_constr = Constraint(m.B, m.T, rule=diff_rule)

            # -------------------------
            # 5) Objetivo
            # -------------------------
            # minimizar el máximo desequilibrio
            m.obj = Objective(expr=m.max_diff, sense=minimize)

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
            # 12) Upper-bound (activaciones)
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
            # 13) No-solapamiento de alphas
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
            solver = SolverFactory('gurobi')
            solver.options.update({
                'LogToConsole': 1,
                'LogFile':      os.path.join(out_dir, f'gurobi_{turno}.log'),
                'TimeLimit':    15
            })

            res = solver.solve(m, tee=False, load_solutions=False)
            if res.solver.termination_condition in (TerminationCondition.infeasible,):
                logger.error("Infactible, escribiendo IIS…")
                write_iis(m, os.path.join(out_dir, f"IIS_{semana}_{turno}.ilp"), solver="gurobi")

            solver.solve(m, tee=False)

            # guardar
            df = []
            for v in m.component_objects(Var, active=True):
                for idx in v:
                    val = v[idx].value
                    if val:
                        df.append({'var': v.name, 'idx': idx, 'val': val})
            pd.DataFrame(df).to_excel(
                os.path.join(out_dir, f"resultados_{semana}_{participacion}_T{turno}.xlsx"),
                index=False
            )
            logger.info(f"--- [MINMAX] Turno {turno} completado. ---")
