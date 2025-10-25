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
            
            t2 = str(turno).zfill(2)
            
            logger.info(f"--- INICIANDO TURNO {turno} / SEMANA {semana} ---")
            datos = pd.read_excel(
                os.path.join(
                    base_instancias,
                    f"instancias_turno_{semana}",
                    f"Instancia_{semana}_{participacion}_T{t2}.xlsx"
                ),
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

            _BC = datos.get('BC',  pd.DataFrame({'BC': []}))
            _BT = datos.get('BT',  pd.DataFrame({'BT': []}))
            _BH = datos.get('BH',  pd.DataFrame({'BH': []}))
            _BI = datos.get('BI',  pd.DataFrame({'BI': []}))
            m.BC = Set(initialize=[r['BC'] for r in _BC.to_dict('records')])   # Costanera
            m.BT = Set(initialize=[r['BT'] for r in _BT.to_dict('records')])   # Tebas
            m.BH = Set(initialize=[r['BH'] for r in _BH.to_dict('records')])   # O'Higgins
            m.BI = Set(initialize=[r['BI'] for r in _BI.to_dict('records')])   # Imo

            _GRT = datos.get('GRT', pd.DataFrame({'GRT': []}))
            _GRS = datos.get('GRS', pd.DataFrame({'GRS': []}))
            m.GRT = Set(initialize=[r['GRT'] for r in _GRT.to_dict('records')])  # RTG
            m.GRS = Set(initialize=[r['GRS'] for r in _GRS.to_dict('records')])  # RS

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

            # ---------- Parámetros escalares legacy (no usados ya en capacidad) ----------
            # Se mantienen por compatibilidad, pero la capacidad y FO usan Prod por tipo.
            m.mu = Param(initialize=datos['mu'].iloc[0,0], mutable=True)   # legacy
            m.W  = Param(initialize=datos['W' ].iloc[0,0], mutable=True)   # legacy (NO usado)
            m.K_legacy = Param(initialize=datos['K' ].iloc[0,0], mutable=True)   # legacy (NO usado directo)

            # ---------- Límite por bloque W_b ----------
            df_Wb = datos.get('W_b', pd.DataFrame({'B': [], 'W_b': []}))
            if df_Wb.empty:
                _Wb_map = {b: 3 for b in list(m.B)}
            else:
                _Wb_map = {r['B']: int(r['W_b']) for r in df_Wb.to_dict('records')}
            m.Wb = Param(m.B, initialize=lambda m,b: _Wb_map.get(b, 3), mutable=True)

            # ---------- K por grúa ----------
            df_Kg = datos.get('K_g', pd.DataFrame({'G': [], 'K': []}))
            if df_Kg.empty:
                _Kg_map = {g: (2 if g in set(m.GRT) else 1) for g in list(m.G)}
            else:
                _Kg_map = {r['G']: int(r['K']) for r in df_Kg.to_dict('records')}
            m.Kg = Param(m.G, initialize=lambda m,g: _Kg_map.get(g, 2), mutable=True)

            # ---------- Disponibilidad por tipo (usamos tamaño del set) ----------
            m.RmaxRTG = Param(initialize=len(list(m.GRT)), mutable=True)
            m.RmaxRS  = Param(initialize=len(list(m.GRS)), mutable=True)

            # ---------- Productividades por tipo ----------
            df_PROD = datos.get('PROD', pd.DataFrame({'Tipo': [], 'Prod': []}))
            _prod_map = {str(r['Tipo']).strip().upper(): int(r['Prod']) for r in df_PROD.to_dict('records')}
            m.ProdRTG = Param(initialize=_prod_map.get("RTG", 300), mutable=True)
            m.ProdRS  = Param(initialize=_prod_map.get("RS",  300), mutable=True)

            # ---------- Compatibilidades por tipo (simultaneidad) ----------
            df_CBR = datos.get('CBR', None)
            df_CBS = datos.get('CBS', None)
            if df_CBR is None or len(df_CBR) == 0:
                CBR_init = {(b1, b2): 1 for b1 in list(m.B) for b2 in list(m.B)}
            else:
                CBR_init = {(r['b1'], r['b2']): int(r['CBR']) for r in df_CBR.to_dict('records')}
            if df_CBS is None or len(df_CBS) == 0:
                CBS_init = {(b1, b2): 1 for b1 in list(m.B) for b2 in list(m.B)}
            else:
                CBS_init = {(r['b1'], r['b2']): int(r['CBS']) for r in df_CBS.to_dict('records')}
            m.CBR = Param(m.B, m.B, initialize=lambda m,b1,b2: CBR_init.get((b1,b2), 1), mutable=True)
            m.CBS = Param(m.B, m.B, initialize=lambda m,b1,b2: CBS_init.get((b1,b2), 1), mutable=True)

            # ---------- Exclusividad a horizonte ----------
            # * Base (EX): neutral (todo 2) => mantiene la restricción original sin afectar.
            # * RTG (EX_RTG): pares legacy permitidos en Costanera; 1 = veto, 2 = permitido.
            df_EX_base = datos.get('EX', pd.DataFrame({'b1': [], 'b2': [], 'EX': []}))
            if df_EX_base.empty:
                EX_init = {(b1, b2): 2 for b1 in list(m.B) for b2 in list(m.B)}
            else:
                EX_init = {(r['b1'], r['b2']): int(r['EX']) for r in df_EX_base.to_dict('records')}
            m.EX = Param(m.B, m.B, initialize=lambda m,b1,b2: EX_init.get((b1,b2), 2), mutable=True)

            df_EX_rtg = datos.get('EX_RTG', pd.DataFrame({'b1': [], 'b2': [], 'EX': []}))
            if df_EX_rtg.empty:
                EXRTG_init = {(b1, b2): 2 for b1 in list(m.B) for b2 in list(m.B)}
            else:
                EXRTG_init = {(r['b1'], r['b2']): int(r['EX']) for r in df_EX_rtg.to_dict('records')}
            m.EX_RTG = Param(m.B, m.B, initialize=lambda m,b1,b2: EXRTG_init.get((b1,b2), 2), mutable=True)

            # ----------------- (LEGACY ELIMINADO) -----------------
            # Las tablas locales adyac_no_exc_* y m.ex legacy ya no se usan.
            # ------------------------------------------------------

            # Variables de flujos
            m.fc_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fd_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fr_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
            m.fe_sbt    = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)

            # Variables de asignación y permanencia
            m.ygbt      = Var(m.G, m.B, m.T, domain=Binary)
            m.alpha_gbt = Var(m.G, m.B, m.T, domain=Binary)
            m.Z_gb      = Var(m.G, m.B,      domain=Binary)

            # Agregados por tipo y activadores por bloque–tiempo
            m.nRTG = Var(m.B, m.T, domain=NonNegativeIntegers)  # # RTG en (b,t)
            m.nRS  = Var(m.B, m.T, domain=NonNegativeIntegers)  # # RS  en (b,t)
            m.aRTG = Var(m.B, m.T, domain=Binary)               # activación RTG en (b,t)
            m.aRS  = Var(m.B, m.T, domain=Binary)               # activación RS  en (b,t)

            # Min–max (maximizamos el mínimo slack)
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
            # 2) Demandas por turno
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
            # 3) Totalizar recibir/entregar
            # -------------------------
            def dem_recibir(m, s):
                return sum(m.fr_sbt[s,b,t] for b in m.B_E for t in m.T) == m.Gs[s]
            m.dem_recibir = Constraint(m.S_E, rule=dem_recibir)

            def plan_entregar_total(m, s):
                if s in m.S_I:
                    total_turno = sum(m.EIbs[b, s] for b in m.B_I)
                    return sum(m.fe_sbt[s, b, t] for b in m.B_I for t in m.T) == total_turno
                return Constraint.Skip
            m.plan_entregar_total = Constraint(m.S, rule=plan_entregar_total)

            # ---------------------------------
            # 4) FO (Max–Min slack) por tipo de grúa
            # ---------------------------------
            def diff_rule(m, b, t):
                carga     = sum(m.fc_sbt[s,b,t] + m.fr_sbt[s,b,t] for s in m.S_E)
                descarga  = sum(m.fd_sbt[s,b,t] + m.fe_sbt[s,b,t] for s in m.S_I)
                capacidad = m.ProdRTG * m.nRTG[b,t] + m.ProdRS * m.nRS[b,t]
                return capacidad - (carga + descarga) >= m.min_diff_val
            m.diff_constr = Constraint(m.B, m.T, rule=diff_rule)

            m.obj = Objective(expr=m.min_diff_val, sense=maximize)

            # -------------------------
            # 5) Vincular Z y ygbt
            # -------------------------
            m.Z_y_up = ConstraintList()
            m.y_Z_up = ConstraintList()
            for g in m.G:
                for b in m.B:
                    m.Z_y_up.add(m.Z_gb[g,b] <= sum(m.ygbt[g,b,t] for t in m.T))
                    m.y_Z_up.add(sum(m.ygbt[g,b,t] for t in m.T) <= m.Z_gb[g,b] * len(m.T))

            # ---------------------------------
            # 6) Exclusividad entre bloques (Z)
            # ---------------------------------
            # Base (neutral) — se mantiene la 21 original sin que restrinja.
            m.excl_base = ConstraintList()
            for g in m.G:
                for b1 in m.B:
                    for b2 in m.B:
                        if b1 != b2:
                            m.excl_base.add(m.Z_gb[g,b1] + m.Z_gb[g,b2] <= m.EX[b1,b2])

            # Específica RTG (21.1): pares permitidos legacy (EX_RTG)
            m.excl_rtg = ConstraintList()
            for g in m.GRT:
                for b1 in m.B:
                    for b2 in m.B:
                        if b1 != b2:
                            m.excl_rtg.add(m.Z_gb[g,b1] + m.Z_gb[g,b2] <= m.EX_RTG[b1,b2])

            # -------------------------
            # 7) Capacidad por tipo
            # -------------------------
            # Conteo exacto por tipo: nRTG/nRS = suma de asignaciones del tipo en (b,t)
            m.count_rtg = Constraint(m.B, m.T,
                                     rule=lambda m,b,t: m.nRTG[b,t] == sum(m.ygbt[g,b,t] for g in m.GRT))
            m.count_rs  = Constraint(m.B, m.T,
                                     rule=lambda m,b,t: m.nRS[b,t]  == sum(m.ygbt[g,b,t] for g in m.GRS))

            # Activación vs cantidad (si hay ≥1, activa)
            BIG_RTG = max(1, int(value(m.RmaxRTG)))
            BIG_RS  = max(1, int(value(m.RmaxRS)))
            m.link_rtg = Constraint(m.B, m.T, rule=lambda m,b,t: m.nRTG[b,t] <= BIG_RTG * m.aRTG[b,t])
            m.link_rs  = Constraint(m.B, m.T, rule=lambda m,b,t: m.nRS[b,t]  <= BIG_RS  * m.aRS[b,t])

            # Disponibilidad total por tipo por periodo
            m.total_rtg = Constraint(m.T, rule=lambda m,t: sum(m.nRTG[b,t] for b in m.B) <= m.RmaxRTG)
            m.total_rs  = Constraint(m.T, rule=lambda m,t: sum(m.nRS[b,t]  for b in m.B) <= m.RmaxRS)

            # Compatibilidades por tipo (CBR/CBS); sólo se activan si hay 0s
            m.compat_rtg = ConstraintList()
            m.compat_rs  = ConstraintList()
            for t in m.T:
                for b1 in m.B:
                    for b2 in m.B:
                        if b1 < b2:
                            if int(value(m.CBR[b1,b2])) == 0:
                                m.compat_rtg.add(m.aRTG[b1,t] + m.aRTG[b2,t] <= 1)
                            if int(value(m.CBS[b1,b2])) == 0:
                                m.compat_rs.add(m.aRS[b1,t] + m.aRS[b2,t] <= 1)

            # RTG solo en Costanera (BC)
            m.rtg_solo_costanera = ConstraintList()
            for b in m.B:
                if b not in set(m.BC):
                    for t in m.T:
                        m.rtg_solo_costanera.add(m.nRTG[b,t] == 0)

            # -------------------------
            # 8) Inventario dinámico (min/max)
            # -------------------------
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
            # 9) Exclusividad de grúas por periodo
            # -------------------------
            m.one_block = Constraint(m.G, m.T,
                                     rule=lambda m,g,t: sum(m.ygbt[g,b,t] for b in m.B) <= 1)

            # (LEGACY) Máx grúas totales por periodo — eliminado; usamos límites por tipo
            # def max_cranes(m, t):
            #     return sum(m.ygbt[g,b,t] for g in m.G for b in m.B) <= m.Rmax
            # m.max_cranes = Constraint(m.T, rule=max_cranes)

            # -------------------------
            # 10) Colisiones por bloque (W_b=3 y máx 2 RTG)
            # -------------------------
            m.max_by_block = Constraint(m.B, m.T,
                                        rule=lambda m,b,t: sum(m.ygbt[g,b,t] for g in m.G) <= m.Wb[b])
            m.max_rtg_block = Constraint(m.B, m.T,
                                         rule=lambda m,b,t: sum(m.ygbt[g,b,t] for g in m.GRT) <= 2)

            # -------------------------
            # 11) Permanencia mínima por grúa (K_g)
            # -------------------------
            def lb_constraint(m, g, b, t):
                Kg = int(value(m.Kg[g]))
                T_max = max(m.T)
                if t <= T_max - Kg + 1:
                    return Kg * m.alpha_gbt[g,b,t] <= sum(m.ygbt[g,b,r] for r in m.T if r >= t and r < t + Kg)
                return Constraint.Skip
            m.lb_constraint = Constraint(m.G, m.B, m.T, rule=lb_constraint)

            def lb1_constraint(m, g, b, t):
                Kg = int(value(m.Kg[g]))
                T_max = max(m.T)
                if t > T_max - Kg + 1:
                    return (T_max - t + 1) * m.alpha_gbt[g,b,t] <= sum(m.ygbt[g,b,r] for r in m.T if r >= t)
                return Constraint.Skip
            m.lb1_constraint = Constraint(m.G, m.B, m.T, rule=lb1_constraint)

            # Evolución: sólo “enciende” con alpha (como antes)
            def ub_constraint(m, g, b, t):
                if t > min(m.T):
                    return m.ygbt[g,b,t] <= m.ygbt[g,b,t-1] + m.alpha_gbt[g,b,t]
                return Constraint.Skip
            m.ub_constraint = Constraint(m.G, m.B, m.T, rule=ub_constraint)

            def ub1_constraint(m, g, b):
                t0 = min(m.T)
                return m.ygbt[g,b,t0] <= m.alpha_gbt[g,b,t0]
            m.ub1_constraint = Constraint(m.G, m.B, rule=ub1_constraint)

            # No-solapamiento de inicios dentro de ventana K_g
            m.alpha_nosolapa = ConstraintList()
            for g in m.G:
                Kg = int(value(m.Kg[g]))
                for b in m.B:
                    for t in m.T:
                        for r in m.T:
                            if t < r < t + Kg:
                                m.alpha_nosolapa.add(m.alpha_gbt[g,b,t] <= 1 - m.alpha_gbt[g,b,r])

            # -------------------------
            # Solver
            # -------------------------
            solver = SolverFactory("gurobi")
            base = {
                "TimeLimit":   100,
                "Threads":     max(1, os.cpu_count()-1),
                "Presolve":    2,
                "Cuts":        2,
                "Symmetry":    2,
                "Heuristics":  0.10,
                "Seed":        42,
                "Method":      2,
                "LogToConsole": 0,
            }
            if base["Threads"] >= 4:
                base["ConcurrentMIP"] = 1
            solver.options.update(base)

            # Ruta destino (xlsx / ilp / iis comparten la misma raíz)
            resultado_xlsx = os.path.join(out_dir, f"resultados_{semana}_{participacion}_T{t2}.xlsx")

            # PRIMER SOLVE: factibilidad sin cargar solución
            res = solver.solve(m, tee=False, load_solutions=False)
            term = res.solver.termination_condition
            status = res.solver.status

            if term in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
                _dump_infeasible_artifacts(m, resultado_xlsx, logger)
                logger.info("Turno %s completado.", turno)
                continue

            # SEGUNDO SOLVE: cargar solución si existe
            res2 = solver.solve(m, tee=False, load_solutions=True)
            term2 = res2.solver.termination_condition
            status2 = res2.solver.status

            if (term2 in (TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations) or
                status2 in (SolverStatus.aborted, SolverStatus.unknown)) and not _has_solution(m):
                _dump_infeasible_artifacts(m, resultado_xlsx, logger)
                logger.info("Turno %s completado.", turno)
                continue

            # Exportar Excel (solo valores no nulos)
            df = []
            for v in m.component_objects(Var, active=True):
                for idx in v:
                    val = v[idx].value
                    if val:
                        df.append({'var': v.name, 'idx': idx, 'val': val})

            pd.DataFrame(df).to_excel(resultado_xlsx, index=False)
            logger.info("Turno %s completado.", turno)
