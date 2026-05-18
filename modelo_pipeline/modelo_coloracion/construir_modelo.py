# Construcción del modelo Pyomo de coloración (Magdalena).
# Recibe el dict de DataFrames leído desde Instancia_*.xlsx y devuelve (model, ctx).

import math
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, NonNegativeIntegers, Binary, NonNegativeReals, minimize, value,
)

from . import config as _cfg


def construir_modelo(df: dict) -> tuple:
    """
    Construye el modelo Pyomo a partir del dict de DataFrames `df`.

    Retorna:
        model  : ConcreteModel de Pyomo listo para resolver.
        ctx    : dict con metadata: segregacion_map, es_pila, S40/S20R/S40R,
                 bloque_id_map, seg_id_map, dispersion_mode.
    """
    model = ConcreteModel()

    model.B = Set(initialize=df["B"].iloc[:, 0].tolist())
    model.S = Set(initialize=df["S"].iloc[:, 0].tolist())
    model.T = Set(initialize=df["T"].iloc[:, 0].tolist())

    yard_map = _yard_map(list(model.B))
    model.YARDS = Set(initialize=list(yard_map.keys()))

    segregacion_map = dict(zip(df['S']['S'], df['S']['Segregacion']))
    bloques_orden   = df['B'].iloc[:, 0].tolist()
    segs_orden      = df['S'].iloc[:, 0].tolist()
    bloque_id_map   = {b: i + 1 for i, b in enumerate(bloques_orden)}
    seg_id_map      = {s: i + 1 for i, s in enumerate(segs_orden)}

    _agregar_parametros(model, df, yard_map)

    if 'MODE' in df:
        es_pila = str(df['MODE'].iloc[0, 0]).strip().lower() == 'pila'
    else:
        C_mediana = float(pd.Series(df['C_b']['C']).astype(float).median())
        es_pila   = C_mediana <= 6.0

    S40  = [s for s in model.S if int(value(model.TEU[s])) == 2]
    S20R = [s for s in model.S if int(value(model.TEU[s])) == 1 and int(value(model.R[s])) == 1]
    S40R = [s for s in model.S if int(value(model.TEU[s])) == 2 and int(value(model.R[s])) == 1]

    _agregar_variables(model, es_pila)
    _agregar_restricciones(model, es_pila, S40, S20R, S40R, yard_map)
    _agregar_objetivo(model)

    ctx = {
        "segregacion_map": segregacion_map,
        "es_pila":         es_pila,
        "S40":             S40,
        "S20R":            S20R,
        "S40R":            S40R,
        "bloque_id_map":   bloque_id_map,
        "seg_id_map":      seg_id_map,
    }

    return model, ctx


# --- helpers ---

def _yard_map(blocks):
    return {
        'C':  [b for b in blocks if b.startswith('C')],
        'H':  [b for b in blocks if b.startswith('H')],
        'TI': [b for b in blocks if b.startswith('T') or b.startswith('I')],
    }


# --- parámetros ---

def _agregar_parametros(model, df, yard_map):
    from .config import VALOR_BASE_R, VALOR_BASE_M

    model.C   = Param(model.B, initialize=df['C_b'].set_index('B')['C'].to_dict())
    model.VS  = Param(model.B, initialize=df['VS_b'].set_index('B')['VS'].to_dict())
    model.VSR = Param(model.B, initialize=df['VSR_b'].set_index('B')['VSR'].to_dict())

    model.KI = Param(model.S, initialize=df['KI_s'].set_index('S')['KI'].to_dict())

    I0_dict = {(row['S'], row['B']): row['I0'] for _, row in df['I0_sb'].iterrows()}
    model.I0 = Param(model.S, model.B, initialize=I0_dict, within=NonNegativeIntegers)

    D_params = df['D_params']
    DR_dict  = {(row['S'], row['T']): row['DR'] for _, row in D_params.iterrows()}
    DC_dict  = {(row['S'], row['T']): row['DC'] for _, row in D_params.iterrows()}
    DD_dict  = {(row['S'], row['T']): row['DD'] for _, row in D_params.iterrows()}
    DE_dict  = {(row['S'], row['T']): row['DE'] for _, row in D_params.iterrows()}

    model.DR = Param(model.S, model.T, initialize=DR_dict, within=NonNegativeIntegers)
    model.DC = Param(model.S, model.T, initialize=DC_dict, within=NonNegativeIntegers)
    model.DD = Param(model.S, model.T, initialize=DD_dict, within=NonNegativeIntegers)
    model.DE = Param(model.S, model.T, initialize=DE_dict, within=NonNegativeIntegers)

    lc_dict  = {(row['S'], row['B']): row['LC'] for _, row in df['LC_sb'].iterrows()}
    model.LC = Param(model.S, model.B, initialize=lc_dict, within=NonNegativeIntegers)

    model.VP   = Param(model.B, initialize=df['VP_b'].set_index('B')['VP'].to_dict())
    model.ROWS = Param(model.B, initialize=df['ROWS_b'].set_index('B')['ROWS'].to_dict())
    model.E    = Param(model.B, initialize=df['E_b'].set_index('B')['E'].to_dict())
    model.LE   = Param(model.B, initialize=df['LE_b'].set_index('B')['LE'].to_dict())
    model.TEU  = Param(model.S, initialize=df['TEU_s'].set_index('S')['TEU'].to_dict())
    model.OS   = Param(initialize=1, mutable=True)
    model.R    = Param(model.S, initialize=df['R_s'].set_index('S')['R'].to_dict())

    OI_dict  = {row.B: 1.0 / float(row.C) for _, row in df['C_b'].iterrows()}
    model.OI = Param(model.B, within=NonNegativeReals, initialize=OI_dict)

    model.r = Param(model.YARDS, initialize=VALOR_BASE_R, mutable=True)
    model.M = Param(model.YARDS, initialize=VALOR_BASE_M, mutable=True)



# --- variables ---

def _agregar_variables(model, es_pila: bool):
    model.fr = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.fc = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.fd = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.fe = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)

    model.y  = Var(model.S, model.B, model.T, domain=Binary,              initialize=0)
    model.u  = Var(model.S, model.B,          domain=Binary,              initialize=0)
    model.v  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.k  = Var(model.S,                   domain=NonNegativeIntegers, initialize=0)

    model.i  = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.w  = Var(model.B,          model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.p  = Var(model.YARDS, model.T, domain=NonNegativeIntegers, initialize=0.0)
    model.q  = Var(model.YARDS, model.T, domain=NonNegativeIntegers, initialize=0.0)

    if es_pila:
        model.g20 = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0)
        model.g40 = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0)


# --- restricciones ---

def _agregar_restricciones(model, es_pila: bool, S40, S20R, S40R, yard_map):

    # dispersión de flujos (eq. 17 y 17b)
    T_sorted = sorted(model.T)
    cum_entrada = {}
    cum_salida  = {}
    for s in model.S:
        acc_in, acc_out = 0, 0
        for t in T_sorted:
            acc_in  += value(model.DR[s, t]) + value(model.DD[s, t])
            acc_out += value(model.DC[s, t]) + value(model.DE[s, t])
            cum_entrada[s, t] = acc_in
            cum_salida[s, t]  = acc_out

    model.constraint_disp_entrada = ConstraintList()
    model.constraint_disp_salida  = ConstraintList()
    for s in model.S:
        ki = max(1, value(model.KI[s]))
        for b in model.B:
            i0_sb = value(model.I0[s, b])
            for idx, t in enumerate(T_sorted):
                periodos = T_sorted[:idx + 1]
                rhs_in = math.ceil(_cfg.THETA_DISPERSION * cum_entrada[s, t] / ki)
                if rhs_in > 0:
                    model.constraint_disp_entrada.add(
                        sum(model.fr[s, b, tau] + model.fd[s, b, tau] for tau in periodos) <= rhs_in
                    )
                rhs_out = math.ceil(_cfg.THETA_DISPERSION * cum_salida[s, t] / ki) + i0_sb
                if rhs_out > 0:
                    model.constraint_disp_salida.add(
                        sum(model.fc[s, b, tau] + model.fe[s, b, tau] for tau in periodos) <= rhs_out
                    )

    if es_pila:
        model.constraint_pairs_40 = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_pairs_40.add(
                    expr=sum(model.v[s, b, t] for s in S40) <= model.VP[b]
                )

    # balance de inventario
    model.constraint_2 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                if t == 1:
                    model.constraint_2.add(
                        expr=model.i[s, b, t]
                        == model.I0[s, b] + model.fr[s, b, t] + model.fd[s, b, t]
                        - model.fc[s, b, t] - model.fe[s, b, t]
                    )
                else:
                    model.constraint_2.add(
                        expr=model.i[s, b, t]
                        == model.i[s, b, t - 1] + model.fr[s, b, t] + model.fd[s, b, t]
                        - model.fc[s, b, t] - model.fe[s, b, t]
                    )

    # cotas de ocupación por bahía
    model.constraint_3 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                model.constraint_3.add(
                    expr=model.i[s, b, t] <= model.v[s, b, t] * model.OS * model.C[b]
                )

    model.constraint_4 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                model.constraint_4.add(
                    expr=(model.v[s, b, t] - 1) * model.C[b] * model.OS
                    + model.C[b] * model.OI[b]
                    <= model.i[s, b, t]
                )

    # balance de flujos por demanda
    model.constraint_5 = ConstraintList()
    model.constraint_6 = ConstraintList()
    model.constraint_7 = ConstraintList()
    model.constraint_8 = ConstraintList()
    for t in model.T:
        for s in model.S:
            model.constraint_5.add(expr=sum(model.fr[s, b, t] for b in model.B) == model.DR[s, t])
            model.constraint_6.add(expr=sum(model.fc[s, b, t] for b in model.B) == model.DC[s, t])
            model.constraint_7.add(expr=sum(model.fd[s, b, t] for b in model.B) == model.DD[s, t])
            model.constraint_8.add(expr=sum(model.fe[s, b, t] for b in model.B) == model.DE[s, t])

    # activación de y (ec. 17-19)
    _activacion_y(model)

    # vínculo u ↔ y
    model.constraint_11 = ConstraintList()
    for b in model.B:
        for s in model.S:
            model.constraint_11.add(
                expr=model.u[s, b] <= sum(model.y[s, b, t] for t in model.T)
            )

    model.constraint_12 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                model.constraint_12.add(expr=model.u[s, b] >= model.y[s, b, t])

    # capacidad TEU por bloque
    model.constraint_13 = ConstraintList()
    for t in model.T:
        for b in model.B:
            model.constraint_13.add(
                expr=sum(model.v[s, b, t] * model.TEU[s] for s in model.S) <= model.VS[b]
            )

    # total de bloques por segregación
    model.constraint_14 = ConstraintList()
    for s in model.S:
        model.constraint_14.add(
            expr=model.k[s] == sum(model.u[s, b] for b in model.B)
        )

    # cota inferior KI por segregación (ec. 23)
    model.constraint_16 = ConstraintList()
    for s in model.S:
        entradas_totales  = sum(model.DR[s, t] + model.DD[s, t] for t in model.T)
        inventario_inicial = sum(model.I0[s, b] for b in model.B)

        if entradas_totales == 0 and inventario_inicial == 0:
            model.constraint_16.add(model.k[s] == 0)
        else:
            model.constraint_16.add(model.k[s] >= model.KI[s])
            n_bloques_I0 = sum(1 for b in model.B if value(model.I0[s, b]) > 0)
            cota_sup     = max(int(value(model.KI[s])), n_bloques_I0) + _cfg.ALPHA_K
            model.constraint_16.add(model.k[s] <= cota_sup)

    # workload y desbalance p/q
    block_to_yard = {b: j for j, blocks_j in yard_map.items() for b in blocks_j}
    model.constraint_17 = ConstraintList()
    model.constraint_18 = ConstraintList()
    model.constraint_19 = ConstraintList()
    for t in model.T:
        for b in model.B:
            model.constraint_17.add(
                model.w[b, t]
                == sum(
                    model.fr[s, b, t] + model.fc[s, b, t]
                    + model.fd[s, b, t] + model.fe[s, b, t]
                    for s in model.S
                )
            )
            j = block_to_yard[b]
            model.constraint_18.add(model.p[j, t] >= model.w[b, t])
            model.constraint_19.add(model.q[j, t] <= model.w[b, t])

    model.constraint_20 = ConstraintList()
    for t in model.T:
        for j in yard_map:
            model.constraint_20.add(expr=model.p[j, t] - model.q[j, t] <= model.r[j])

    # tope absoluto de movimientos por bloque por turno (por patio)
    model.constraint_w_max = ConstraintList()
    for t in model.T:
        for b in model.B:
            j = block_to_yard[b]
            model.constraint_w_max.add(expr=model.w[b, t] <= model.M[j])

    # reefer (pila) / VSR (bahía)
    if es_pila:
        model.constraint_reefer_plugs = ConstraintList()
        model.constraint_reefer_20    = ConstraintList()
        model.constraint_reefer_40    = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_reefer_plugs.add(
                    expr=model.g20[b, t] + model.g40[b, t] <= model.E[b]
                )
                model.constraint_reefer_20.add(
                    expr=sum(model.v[s, b, t] for s in S20R) <= model.ROWS[b] * model.g20[b, t]
                )
                model.constraint_reefer_40.add(
                    expr=sum(model.v[s, b, t] for s in S40R) <= model.ROWS[b] * model.g40[b, t]
                )
    else:
        model.constraint_21 = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_21.add(
                    expr=sum(model.v[s, b, t] * model.R[s] for s in model.S) <= model.VSR[b]
                )


# --- activación de y_{sbt} (ec. 17, 18, 19) ---

def _activacion_y(model):
    """
    17: fr+fc+fd+fe ≤ (DR+DC+DD+DE)·y  — fuerza flujos a cero si y=0
    18: i ≤ (I0_total_s + neto_acum[t])·y  — Big-M sobre inventario del terminal
    19: i + flujos ≥ y  — fuerza y=0 si no hay actividad
    """
    model.constraint_9_flujo = ConstraintList()
    model.constraint_9_inv   = ConstraintList()
    model.constraint_10      = ConstraintList()

    T_sorted = sorted(model.T)

    for s in model.S:
        # Big-M: inventario inicial total de s en el terminal
        I0_total_s = sum(value(model.I0[s, b_]) for b_ in model.B)

        # flujo neto acumulado hasta cada periodo
        neto_acum = {}
        neto = 0.0
        for t in T_sorted:
            neto += (value(model.DR[s, t]) - value(model.DC[s, t])
                     + value(model.DD[s, t]) - value(model.DE[s, t]))
            neto_acum[t] = neto

        for b in model.B:
            for t in T_sorted:
                model.constraint_9_flujo.add(
                    model.fr[s, b, t] + model.fc[s, b, t]
                    + model.fd[s, b, t] + model.fe[s, b, t]
                    <= (model.DR[s, t] + model.DC[s, t]
                        + model.DD[s, t] + model.DE[s, t]) * model.y[s, b, t]
                )

                model.constraint_9_inv.add(
                    model.i[s, b, t]
                    <= (I0_total_s + neto_acum[t]) * model.y[s, b, t]
                )

                model.constraint_10.add(
                    model.i[s, b, t]
                    + model.fr[s, b, t] + model.fc[s, b, t]
                    + model.fd[s, b, t] + model.fe[s, b, t]
                    >= model.y[s, b, t]
                )


# --- objetivo ---

def _agregar_objetivo(model):
    """Minimiza distancia total de carga (LOAD) + entrega (DLVR)."""
    def objective_rule(m):
        return (
            sum(m.fc[s, b, t] * m.LC[s, b] for b in m.B for s in m.S for t in m.T)
            + sum(m.fe[s, b, t] * m.LE[b]   for b in m.B for s in m.S for t in m.T)
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)
