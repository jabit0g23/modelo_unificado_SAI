"""
Construcción del modelo de grúas (maxmin).

Toma un dict de DataFrames (hojas del Excel por turno) y devuelve un
ConcreteModel de Pyomo con sets, parámetros, variables, restricciones y FO.
"""

import pandas as pd
from pyomo.environ import (
    Binary, ConcreteModel, Constraint, ConstraintList,
    NonNegativeIntegers, NonNegativeReals,
    Objective, Param, Set, Var, maximize, value,
)

from .config import (
    LIMITE_COMBINADO_BT,
    MAX_RTG_POR_BLOQUE,
    PROD_DEFAULT,
    W_B_DEFAULT,
)


def _set_from(df, col):
    return [r[col] for r in df.to_dict("records")]


def _param_map(df, keys, val_col):
    return {tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]: r[val_col]
            for r in df.to_dict("records")}


def construir_modelo(datos: dict) -> ConcreteModel:
    """Construye el ConcreteModel del modelo de grúas maxmin a partir de las hojas."""
    m = ConcreteModel()

    # ── Conjuntos ────────────────────────────────────────────────
    m.G   = Set(initialize=_set_from(datos["G"],   "G"))
    m.B   = Set(initialize=_set_from(datos["B"],   "B"))
    m.B_I = Set(initialize=_set_from(datos["B_I"], "B_I"))
    m.B_E = Set(initialize=_set_from(datos["B_E"], "B_E"))
    m.T   = Set(initialize=_set_from(datos["T"],   "T"))
    m.S   = Set(initialize=_set_from(datos["S"],   "S"))
    m.S_E = Set(initialize=_set_from(datos["S_E"], "S_E"))
    m.S_I = Set(initialize=_set_from(datos["S_I"], "S_I"))

    m.BC = Set(initialize=_set_from(datos.get("BC", pd.DataFrame({"BC": []})), "BC"))
    m.BT = Set(initialize=_set_from(datos.get("BT", pd.DataFrame({"BT": []})), "BT"))
    m.BH = Set(initialize=_set_from(datos.get("BH", pd.DataFrame({"BH": []})), "BH"))
    m.BI = Set(initialize=_set_from(datos.get("BI", pd.DataFrame({"BI": []})), "BI"))

    m.GRT = Set(initialize=_set_from(datos.get("GRT", pd.DataFrame({"GRT": []})), "GRT"))
    m.GRS = Set(initialize=_set_from(datos.get("GRS", pd.DataFrame({"GRS": []})), "GRS"))

    # ── Parámetros matriciales ──────────────────────────────────
    m.AEbs = Param(m.B, m.S,
                   initialize={(r["B_E"], r["S_E"]): r["AEbs"] for r in datos["AEbs"].to_dict("records")},
                   default=0, mutable=True)
    m.AIbs = Param(m.B, m.S,
                   initialize={(r["B_I"], r["S_I"]): r["AIbs"] for r in datos["AIbs"].to_dict("records")},
                   default=0, mutable=True)
    m.EIbs = Param(m.B, m.S,
                   initialize={(r["B_I"], r["S_I"]): r["EIbs"] for r in datos["EIbs"].to_dict("records")},
                   default=0, mutable=True)
    m.Gs   = Param(m.S_E,
                   initialize={r["S_E"]: r["Gs"] for r in datos["Gs"].to_dict("records")},
                   default=0, mutable=True)
    m.DMEst = Param(m.S_E, m.T,
                    initialize={(r["S_E"], r["T"]): r["DMEst"] for r in datos["DMEst"].to_dict("records")},
                    default=0, mutable=True)
    m.DMIst = Param(m.S_I, m.T,
                    initialize={(r["S_I"], r["T"]): r["DMIst"] for r in datos["DMIst"].to_dict("records")},
                    default=0, mutable=True)
    m.Cbs   = Param(m.B, m.S,
                    initialize={(r["B"], r["S"]): r["Cbs"] for r in datos["Cbs"].to_dict("records")},
                    default=0, mutable=True)

    # ── W_b (capacidad simultánea de grúas por bloque) ─────────
    df_Wb = datos.get("W_b", pd.DataFrame({"B": [], "W_b": []}))
    Wb_map = ({r["B"]: int(r["W_b"]) for r in df_Wb.to_dict("records")}
              if not df_Wb.empty else {b: W_B_DEFAULT for b in list(m.B)})
    m.Wb = Param(m.B, initialize=lambda m, b: Wb_map.get(b, W_B_DEFAULT), mutable=True)

    # ── K por grúa (permanencia mínima consecutiva por bloque) ─
    df_Kg = datos.get("K_g", pd.DataFrame({"G": [], "K": []}))
    if df_Kg.empty:
        Kg_map = {g: (2 if g in set(m.GRT) else 1) for g in list(m.G)}
    else:
        Kg_map = {r["G"]: int(r["K"]) for r in df_Kg.to_dict("records")}
    m.Kg = Param(m.G, initialize=lambda m, g: Kg_map.get(g, 2), mutable=True)

    # ── Disponibilidad total por tipo (tamaño del set) ─────────
    m.RmaxRTG = Param(initialize=len(list(m.GRT)), mutable=True)
    m.RmaxRS  = Param(initialize=len(list(m.GRS)), mutable=True)

    # ── Productividades por tipo (mov/turno) ───────────────────
    df_PROD = datos.get("PROD", pd.DataFrame({"Tipo": [], "Prod": []}))
    prod_map = {str(r["Tipo"]).strip().upper(): int(r["Prod"]) for r in df_PROD.to_dict("records")}
    m.ProdRTG = Param(initialize=prod_map.get("RTG", PROD_DEFAULT), mutable=True)
    m.ProdRS  = Param(initialize=prod_map.get("RS",  PROD_DEFAULT), mutable=True)

    # ── Compatibilidades simultáneas por tipo ──────────────────
    def _compat_init(df, col):
        if df is None or len(df) == 0:
            return {(b1, b2): 1 for b1 in list(m.B) for b2 in list(m.B)}
        return {(r["b1"], r["b2"]): int(r[col]) for r in df.to_dict("records")}

    CBR_init = _compat_init(datos.get("CBR"), "CBR")
    CBS_init = _compat_init(datos.get("CBS"), "CBS")
    m.CBR = Param(m.B, m.B, initialize=lambda m, b1, b2: CBR_init.get((b1, b2), 1), mutable=True)
    m.CBS = Param(m.B, m.B, initialize=lambda m, b1, b2: CBS_init.get((b1, b2), 1), mutable=True)

    # ── Exclusividad a horizonte (EX neutral + EX_RTG legacy) ──
    def _ex_init(df):
        if df.empty:
            return {(b1, b2): 2 for b1 in list(m.B) for b2 in list(m.B)}
        return {(r["b1"], r["b2"]): int(r["EX"]) for r in df.to_dict("records")}

    EX_init    = _ex_init(datos.get("EX",     pd.DataFrame({"b1": [], "b2": [], "EX": []})))
    EXRTG_init = _ex_init(datos.get("EX_RTG", pd.DataFrame({"b1": [], "b2": [], "EX": []})))
    m.EX     = Param(m.B, m.B, initialize=lambda m, b1, b2: EX_init.get((b1, b2), 2), mutable=True)
    m.EX_RTG = Param(m.B, m.B, initialize=lambda m, b1, b2: EXRTG_init.get((b1, b2), 2), mutable=True)

    # ── Variables ──────────────────────────────────────────────
    m.fc_sbt = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
    m.fd_sbt = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
    m.fr_sbt = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)
    m.fe_sbt = Var(m.S, m.B, m.T, domain=NonNegativeIntegers)

    m.ygbt      = Var(m.G, m.B, m.T, domain=Binary)
    m.alpha_gbt = Var(m.G, m.B, m.T, domain=Binary)
    m.Z_gb      = Var(m.G, m.B,      domain=Binary)

    m.nRTG = Var(m.B, m.T, domain=NonNegativeIntegers)
    m.nRS  = Var(m.B, m.T, domain=NonNegativeIntegers)
    m.aRTG = Var(m.B, m.T, domain=Binary)
    m.aRS  = Var(m.B, m.T, domain=Binary)

    m.min_diff_val = Var(domain=NonNegativeReals, name="min_diff_val")

    # ── Ceros fuera de dominio (bloques × segregación) ─────────
    m.bloque_I = ConstraintList()
    for b in m.B:
        for t in m.T:
            if b not in m.B_I:
                for s in m.S_I:
                    m.bloque_I.add(m.fd_sbt[s, b, t] == 0)
                    m.bloque_I.add(m.fe_sbt[s, b, t] == 0)
            if b not in m.B_E:
                for s in m.S_E:
                    m.bloque_I.add(m.fc_sbt[s, b, t] == 0)
                    m.bloque_I.add(m.fr_sbt[s, b, t] == 0)

    m.seg_I = ConstraintList()
    for b in m.B:
        for t in m.T:
            for s in m.S:
                if s not in m.S_I:
                    m.seg_I.add(m.fd_sbt[s, b, t] == 0)
                    m.seg_I.add(m.fe_sbt[s, b, t] == 0)
                if s not in m.S_E:
                    m.seg_I.add(m.fc_sbt[s, b, t] == 0)
                    m.seg_I.add(m.fr_sbt[s, b, t] == 0)

    # ── Demandas por turno ─────────────────────────────────────
    def dem_carga(m, s, t):
        if s in m.S_E:
            return sum(m.fc_sbt[s, b, t] for b in m.B_E) == m.DMEst[s, t]
        return Constraint.Skip
    m.dem_carga = Constraint(m.S, m.T, rule=dem_carga)

    def dem_descarga(m, s, t):
        if s in m.S_I:
            return sum(m.fd_sbt[s, b, t] for b in m.B_I) == m.DMIst[s, t]
        return Constraint.Skip
    m.dem_descarga = Constraint(m.S, m.T, rule=dem_descarga)

    # ── Totales recibir/entregar del turno ─────────────────────
    m.dem_recibir = Constraint(m.S_E,
        rule=lambda m, s: sum(m.fr_sbt[s, b, t] for b in m.B_E for t in m.T) == m.Gs[s])

    def plan_entregar_total(m, s):
        if s in m.S_I:
            total_turno = sum(m.EIbs[b, s] for b in m.B_I)
            return sum(m.fe_sbt[s, b, t] for b in m.B_I for t in m.T) == total_turno
        return Constraint.Skip
    m.plan_entregar_total = Constraint(m.S, rule=plan_entregar_total)

    # ── FO: Max-Min slack de capacidad vs demanda por (b,t) ────
    def diff_rule(m, b, t):
        carga     = sum(m.fc_sbt[s, b, t] + m.fr_sbt[s, b, t] for s in m.S_E)
        descarga  = sum(m.fd_sbt[s, b, t] + m.fe_sbt[s, b, t] for s in m.S_I)
        capacidad = m.ProdRTG * m.nRTG[b, t] + m.ProdRS * m.nRS[b, t]
        return capacidad - (carga + descarga) >= m.min_diff_val
    m.diff_constr = Constraint(m.B, m.T, rule=diff_rule)
    m.obj = Objective(expr=m.min_diff_val, sense=maximize)

    # ── Vínculo Z ↔ ygbt ───────────────────────────────────────
    m.Z_y_up = ConstraintList()
    m.y_Z_up = ConstraintList()
    for g in m.G:
        for b in m.B:
            m.Z_y_up.add(m.Z_gb[g, b] <= sum(m.ygbt[g, b, t] for t in m.T))
            m.y_Z_up.add(sum(m.ygbt[g, b, t] for t in m.T) <= m.Z_gb[g, b] * len(m.T))

    # ── Exclusividad entre bloques (Z): base neutral + RTG ─────
    m.excl_base = ConstraintList()
    for g in m.G:
        for b1 in m.B:
            for b2 in m.B:
                if b1 != b2:
                    m.excl_base.add(m.Z_gb[g, b1] + m.Z_gb[g, b2] <= m.EX[b1, b2])

    m.excl_rtg = ConstraintList()
    for g in m.GRT:
        for b1 in m.B:
            for b2 in m.B:
                if b1 != b2:
                    m.excl_rtg.add(m.Z_gb[g, b1] + m.Z_gb[g, b2] <= m.EX_RTG[b1, b2])

    # ── Capacidad por tipo ─────────────────────────────────────
    m.count_rtg = Constraint(m.B, m.T,
        rule=lambda m, b, t: m.nRTG[b, t] == sum(m.ygbt[g, b, t] for g in m.GRT))
    m.count_rs  = Constraint(m.B, m.T,
        rule=lambda m, b, t: m.nRS[b, t]  == sum(m.ygbt[g, b, t] for g in m.GRS))

    BIG_RTG = max(1, int(value(m.RmaxRTG)))
    BIG_RS  = max(1, int(value(m.RmaxRS)))
    m.link_rtg = Constraint(m.B, m.T, rule=lambda m, b, t: m.nRTG[b, t] <= BIG_RTG * m.aRTG[b, t])
    m.link_rs  = Constraint(m.B, m.T, rule=lambda m, b, t: m.nRS[b, t]  <= BIG_RS  * m.aRS[b, t])

    m.total_rtg = Constraint(m.T, rule=lambda m, t: sum(m.nRTG[b, t] for b in m.B) <= m.RmaxRTG)
    m.total_rs  = Constraint(m.T, rule=lambda m, t: sum(m.nRS[b, t]  for b in m.B) <= m.RmaxRS)

    # Compatibilidades (CBR/CBS): sólo activas si traen ceros
    m.compat_rtg = ConstraintList()
    m.compat_rs  = ConstraintList()
    for t in m.T:
        for b1 in m.B:
            for b2 in m.B:
                if b1 < b2:
                    if int(value(m.CBR[b1, b2])) == 0:
                        m.compat_rtg.add(m.aRTG[b1, t] + m.aRTG[b2, t] <= 1)
                    if int(value(m.CBS[b1, b2])) == 0:
                        m.compat_rs.add(m.aRS[b1, t] + m.aRS[b2, t] <= 1)

    # RTG sólo pueden operar en Costanera
    m.rtg_solo_costanera = ConstraintList()
    BC_set = set(m.BC)
    for b in m.B:
        if b not in BC_set:
            for t in m.T:
                m.rtg_solo_costanera.add(m.nRTG[b, t] == 0)

    # ── Inventario dinámico ───────────────────────────────────
    def _inv_expr(m, b, s, t):
        inv = m.AEbs[b, s] + m.AIbs[b, s]
        for i in range(1, t + 1):
            if s in m.S_I:
                inv += m.fd_sbt[s, b, i] - m.fe_sbt[s, b, i]
            if s in m.S_E:
                inv += m.fr_sbt[s, b, i] - m.fc_sbt[s, b, i]
        return inv

    m.inv_min = Constraint(m.B, m.S, m.T, rule=lambda m, b, s, t: _inv_expr(m, b, s, t) >= 0)
    m.inv_max = Constraint(m.B, m.S, m.T, rule=lambda m, b, s, t: _inv_expr(m, b, s, t) <= m.Cbs[b, s])

    # ── Exclusividad de grúa por periodo ─────────────────────
    m.one_block = Constraint(m.G, m.T,
        rule=lambda m, g, t: sum(m.ygbt[g, b, t] for b in m.B) <= 1)

    # ── Colisiones por bloque ────────────────────────────────
    def limit_combinado(m, b, t):
        rtg = sum(m.ygbt[g, b, t] for g in m.GRT)
        rs  = sum(m.ygbt[g, b, t] for g in m.GRS)
        return 2 * rtg + rs <= LIMITE_COMBINADO_BT
    m.limit_13_1 = Constraint(m.B, m.T, rule=limit_combinado)

    m.max_rtg_block = Constraint(m.B, m.T,
        rule=lambda m, b, t: sum(m.ygbt[g, b, t] for g in m.GRT) <= MAX_RTG_POR_BLOQUE)

    # ── Permanencia mínima K_g por grúa ──────────────────────
    def lb_constraint(m, g, b, t):
        Kg = int(value(m.Kg[g]))
        T_max = max(m.T)
        if t <= T_max - Kg + 1:
            return Kg * m.alpha_gbt[g, b, t] <= sum(
                m.ygbt[g, b, r] for r in m.T if t <= r < t + Kg
            )
        return Constraint.Skip
    m.lb_constraint = Constraint(m.G, m.B, m.T, rule=lb_constraint)

    def lb1_constraint(m, g, b, t):
        Kg = int(value(m.Kg[g]))
        T_max = max(m.T)
        if t > T_max - Kg + 1:
            return (T_max - t + 1) * m.alpha_gbt[g, b, t] <= sum(
                m.ygbt[g, b, r] for r in m.T if r >= t
            )
        return Constraint.Skip
    m.lb1_constraint = Constraint(m.G, m.B, m.T, rule=lb1_constraint)

    def ub_constraint(m, g, b, t):
        if t > min(m.T):
            return m.ygbt[g, b, t] <= m.ygbt[g, b, t - 1] + m.alpha_gbt[g, b, t]
        return Constraint.Skip
    m.ub_constraint = Constraint(m.G, m.B, m.T, rule=ub_constraint)

    m.ub1_constraint = Constraint(m.G, m.B,
        rule=lambda m, g, b: m.ygbt[g, b, min(m.T)] <= m.alpha_gbt[g, b, min(m.T)])

    # Inicios no solapados dentro de la ventana K_g
    m.alpha_nosolapa = ConstraintList()
    for g in m.G:
        Kg = int(value(m.Kg[g]))
        for b in m.B:
            for t in m.T:
                for r in m.T:
                    if t < r < t + Kg:
                        m.alpha_nosolapa.add(m.alpha_gbt[g, b, t] <= 1 - m.alpha_gbt[g, b, r])

    return m
