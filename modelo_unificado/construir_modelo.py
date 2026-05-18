# Construcción del modelo unificado Pyomo (Magdalena + Camila, horario).
# Lee Instancia_{semana}_{p}_K.xlsx, trunca T al horizonte y devuelve (model, ctx).
# THETA_DISPERSION se lee del módulo config (inyectado desde main.py).

import logging
import math
import time
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, ConstraintList,
    Objective, Expression, NonNegativeIntegers, Binary, NonNegativeReals,
    minimize, value,
)

from . import config as _cfg
from .config import VALOR_BASE_R, horas_horizonte, LIMITE_COMBINADO_BT, MAX_RTG_POR_BLOQUE

logger = logging.getLogger("unificado")


def construir_modelo(df: dict) -> tuple:
    t_start = time.perf_counter()
    horas = horas_horizonte()
    T_horizonte = [t for t in df["T"].iloc[:, 0].tolist() if int(t) <= horas]

    model = ConcreteModel()

    model.B = Set(initialize=df["B"].iloc[:, 0].tolist())
    model.S = Set(initialize=df["S"].iloc[:, 0].tolist())
    model.T = Set(initialize=T_horizonte)

    segregacion_map = dict(zip(df['S']['S'], df['S']['Segregacion']))
    bloques_orden   = df['B'].iloc[:, 0].tolist()
    segs_orden      = df['S'].iloc[:, 0].tolist()
    bloque_id_map   = {b: i + 1 for i, b in enumerate(bloques_orden)}
    seg_id_map      = {s: i + 1 for i, s in enumerate(segs_orden)}

    nB = len(bloques_orden)
    nS = len(segs_orden)
    nT = len(T_horizonte)
    logger.info("Conjuntos: |B|=%d  |S|=%d  |T|=%d  (horizonte %d h)", nB, nS, nT, horas)

    # S_E/S_I: expo vs impo por prefijo de nombre
    S_E = [s for s, name in segregacion_map.items() if str(name).startswith("expo")]
    S_I = [s for s, name in segregacion_map.items() if str(name).startswith("impo")]
    model.S_E = Set(initialize=S_E)
    model.S_I = Set(initialize=S_I)
    logger.info("Segregaciones: expo=%d  impo=%d  total=%d", len(S_E), len(S_I), nS)

    _agregar_sets_camila(model, df)
    logger.info("Grúas: |G|=%d  RTG=%d  RS=%d  BC=%d",
                len(list(model.G)), len(list(model.GRT)),
                len(list(model.GRS)), len(list(model.BC)))

    logger.info("Cargando parámetros Magdalena…")
    _agregar_parametros_magdalena(model, df, horas)

    if 'MODE' in df:
        es_pila = str(df['MODE'].iloc[0, 0]).strip().lower() == 'pila'
        logger.info("Modo detectado en hoja MODE: %s", "pila" if es_pila else "bahia")
    else:
        C_mediana = float(pd.Series(df['C_b']['C']).astype(float).median())
        es_pila = C_mediana <= 6.0
        logger.info("Modo inferido por C_mediana=%.1f: %s", C_mediana, "pila" if es_pila else "bahia")

    S40  = [s for s in model.S if int(value(model.TEU[s])) == 2]
    S20R = [s for s in model.S if int(value(model.TEU[s])) == 1 and int(value(model.R[s])) == 1]
    S40R = [s for s in model.S if int(value(model.TEU[s])) == 2 and int(value(model.R[s])) == 1]
    logger.info("Subsets TEU/Reefer: S40=%d  S20R=%d  S40R=%d", len(S40), len(S20R), len(S40R))

    logger.info("Dispersión: θ=%.2f", _cfg.THETA_DISPERSION)
    _agregar_parametros_dispersion(model, df, horas)

    logger.info("Cargando parámetros Camila…")
    _agregar_parametros_camila(model, df)

    logger.info("Creando variables…")
    _agregar_variables(model, es_pila)

    logger.info("Añadiendo restricciones Magdalena…")
    _restricciones_magdalena(model, es_pila, S40, S20R, S40R)

    logger.info("Añadiendo restricciones Camila + acople…")
    _restricciones_camila(model)

    _definir_objetivos(model)

    elapsed = time.perf_counter() - t_start
    logger.info("Modelo construido en %.1f s", elapsed)

    ctx = {
        "segregacion_map": segregacion_map,
        "es_pila":         es_pila,
        "S40":             S40,
        "S20R":            S20R,
        "S40R":            S40R,
        "bloque_id_map":   bloque_id_map,
        "seg_id_map":      seg_id_map,
        "horas":           horas,
    }
    return model, ctx


# --- parámetros ---

def _agregar_parametros_magdalena(model, df, horas):
    model.C   = Param(model.B, initialize=df['C_b'].set_index('B')['C'].to_dict())
    model.VS  = Param(model.B, initialize=df['VS_b'].set_index('B')['VS'].to_dict())
    model.VSR = Param(model.B, initialize=df['VSR_b'].set_index('B')['VSR'].to_dict())

    # KI calibrado para la semana completa; escalamos al horizonte para evitar
    # infactibilidad cuando horas < 168 (muy poco flujo para cubrir el mínimo semanal).
    ki_raw = df['KI_s'].set_index('S')['KI'].to_dict()
    if horas < 168:
        ki_dict = {s: max(1, int(math.floor(v * horas / 168))) for s, v in ki_raw.items()}
        logger.info("KI escalado al horizonte (%d h): max=%d  media=%.1f",
                    horas, max(ki_dict.values()), sum(ki_dict.values()) / len(ki_dict))
    else:
        ki_dict = {s: max(1, int(v)) for s, v in ki_raw.items()}
    model.KI = Param(model.S, initialize=ki_dict)

    I0_dict = {(row['S'], row['B']): row['I0'] for _, row in df['I0_sb'].iterrows()}
    model.I0 = Param(model.S, model.B, initialize=I0_dict, within=NonNegativeIntegers)

    # D_params filtrado al horizonte
    D_params = df['D_params']
    D_params = D_params[D_params['T'].astype(int) <= horas]
    DR_dict = {(r['S'], r['T']): r['DR'] for _, r in D_params.iterrows()}
    DC_dict = {(r['S'], r['T']): r['DC'] for _, r in D_params.iterrows()}
    DD_dict = {(r['S'], r['T']): r['DD'] for _, r in D_params.iterrows()}
    DE_dict = {(r['S'], r['T']): r['DE'] for _, r in D_params.iterrows()}

    model.DR = Param(model.S, model.T, initialize=DR_dict, within=NonNegativeIntegers)
    model.DC = Param(model.S, model.T, initialize=DC_dict, within=NonNegativeIntegers)
    model.DD = Param(model.S, model.T, initialize=DD_dict, within=NonNegativeIntegers)
    model.DE = Param(model.S, model.T, initialize=DE_dict, within=NonNegativeIntegers)

    lc_dict  = {(r['S'], r['B']): r['LC'] for _, r in df['LC_sb'].iterrows()}
    model.LC = Param(model.S, model.B, initialize=lc_dict, within=NonNegativeIntegers)
    model.LE = Param(model.B, initialize=df['LE_b'].set_index('B')['LE'].to_dict())

    # VP, ROWS, E (pila). Si no vienen, usamos defaults seguros.
    VP_map   = (df['VP_b'].set_index('B')['VP'].to_dict()     if 'VP_b'   in df else {b: int(value(model.VS[b])) // 2 for b in model.B})
    ROWS_map = (df['ROWS_b'].set_index('B')['ROWS'].to_dict() if 'ROWS_b' in df else {b: 6 for b in model.B})
    E_map    = (df['E_b'].set_index('B')['E'].to_dict()       if 'E_b'    in df else {b: 0 for b in model.B})
    model.VP   = Param(model.B, initialize=VP_map)
    model.ROWS = Param(model.B, initialize=ROWS_map)
    model.E    = Param(model.B, initialize=E_map)

    model.TEU = Param(model.S, initialize=df['TEU_s'].set_index('S')['TEU'].to_dict())
    model.OS  = Param(initialize=1, mutable=True)
    model.R   = Param(model.S, initialize=df['R_s'].set_index('S')['R'].to_dict())

    OI_dict = {row.B: 1.0 / float(row.C) for _, row in df['C_b'].iterrows()}
    model.OI = Param(model.B, within=NonNegativeReals, initialize=OI_dict)

    # cota laxa p[t]-q[t] ≤ r (solo estabiliza el LP, no entra a la FO)
    model.r = Param(initialize=VALOR_BASE_R, mutable=True)


def _agregar_parametros_dispersion(model, df, horas):
    S_list = [s for s in df["S"].iloc[:, 0].tolist()]
    T_list = sorted([int(t) for t in model.T])
    KI_map = df['KI_s'].set_index('S')['KI'].to_dict()
    D_rows = df['D_params']
    D_rows = D_rows[D_rows['T'].astype(int) <= horas]
    DR_dict = {(r['S'], r['T']): r['DR'] for _, r in D_rows.iterrows()}
    DC_dict = {(r['S'], r['T']): r['DC'] for _, r in D_rows.iterrows()}
    DD_dict = {(r['S'], r['T']): r['DD'] for _, r in D_rows.iterrows()}
    DE_dict = {(r['S'], r['T']): r['DE'] for _, r in D_rows.iterrows()}

    CapBlockI_dict = {}
    for s in S_list:
        ki = max(1, int(KI_map[s]))
        running = 0
        for t in T_list:
            running += (int(DR_dict[(s, t)]) - int(DC_dict[(s, t)])
                        + int(DD_dict[(s, t)]) - int(DE_dict[(s, t)]))
            CapBlockI_dict[(s, t)] = int(math.ceil(_cfg.THETA_DISPERSION * running / ki)) if running > 0 else running
    model.CAPBLOCKI = Param(model.S, model.T, initialize=CapBlockI_dict)


def _agregar_sets_camila(model, df):
    def _safe(name, col):
        return df[name][col].tolist() if name in df else []

    model.G   = Set(initialize=_safe("G",   "G"))
    model.GRT = Set(initialize=_safe("GRT", "GRT"))
    model.GRS = Set(initialize=_safe("GRS", "GRS"))
    model.BC  = Set(initialize=_safe("BC",  "BC"))
    model.BH  = Set(initialize=_safe("BH",  "BH"))
    model.BT  = Set(initialize=_safe("BT",  "BT"))
    model.BI  = Set(initialize=_safe("BI",  "BI"))
    model.B_E = Set(initialize=_safe("B_E", "B_E"))
    model.B_I = Set(initialize=_safe("B_I", "B_I"))


def _agregar_parametros_camila(model, df):
    df_PROD = df.get("PROD", pd.DataFrame({"Tipo": [], "Prod": []}))
    prod_map = {str(r["Tipo"]).strip().upper(): float(r["Prod"]) for _, r in df_PROD.iterrows()}
    model.mu_RTG = Param(initialize=prod_map.get("RTG", 30.0), mutable=True)
    model.mu_RS  = Param(initialize=prod_map.get("RS",  20.0), mutable=True)

    df_Wb = df.get("W_b", pd.DataFrame({"B": [], "W_b": []}))
    Wb_map = ({r["B"]: int(r["W_b"]) for _, r in df_Wb.iterrows()}
              if not df_Wb.empty else {b: 6 for b in model.B})
    model.Wb = Param(model.B, initialize=lambda m, b: Wb_map.get(b, 6), mutable=True)

    df_Kg = df.get("K_g", pd.DataFrame({"G": [], "K": []}))
    if df_Kg.empty:
        Kg_map = {g: (2 if g in set(model.GRT) else 1) for g in list(model.G)}
    else:
        Kg_map = {r["G"]: int(r["K"]) for _, r in df_Kg.iterrows()}
    model.Kg = Param(model.G, initialize=lambda m, g: Kg_map.get(g, 2), mutable=True)

    model.RmaxRTG = Param(initialize=len(list(model.GRT)), mutable=True)
    model.RmaxRS  = Param(initialize=len(list(model.GRS)), mutable=True)

    # C_bs: capacidad máxima de contenedores por (b, s)
    df_Cbs = df.get("C_bs", pd.DataFrame({"B": [], "S": [], "Cbs": []}))
    Cbs_dict = {(r["B"], r["S"]): int(r["Cbs"]) for _, r in df_Cbs.iterrows()}
    model.Cbs = Param(model.B, model.S, initialize=Cbs_dict, default=9999, mutable=True)

    # Compatibilidad / exclusividad entre bloques
    def _compat_init(df_src, col, default):
        if df_src is None or df_src.empty:
            return {(b1, b2): default for b1 in list(model.B) for b2 in list(model.B)}
        return {(r["b1"], r["b2"]): int(r[col]) for _, r in df_src.iterrows()}

    CBR_init = _compat_init(df.get("CBR"),    "CBR", 1)
    CBS_init = _compat_init(df.get("CBS"),    "CBS", 1)
    EX_init  = _compat_init(df.get("EX"),     "EX",  2)
    EXR_init = _compat_init(df.get("EX_RTG"), "EX",  2)
    model.CBR    = Param(model.B, model.B, initialize=lambda m, b1, b2: CBR_init.get((b1, b2), 1), mutable=True)
    model.CBS    = Param(model.B, model.B, initialize=lambda m, b1, b2: CBS_init.get((b1, b2), 1), mutable=True)
    model.EX     = Param(model.B, model.B, initialize=lambda m, b1, b2: EX_init.get((b1, b2), 2),  mutable=True)
    model.EX_RTG = Param(model.B, model.B, initialize=lambda m, b1, b2: EXR_init.get((b1, b2), 2), mutable=True)

    # ε Pareto (activado/desactivado por el resolver)
    model.eps_D = Param(initialize=0.0, mutable=True)


# --- variables ---

def _agregar_variables(model, es_pila: bool):
    # Magdalena
    model.fr = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.fc = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.fd = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.fe = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)

    model.y = Var(model.S, model.B, model.T, domain=Binary, initialize=0)
    model.u = Var(model.S, model.B,          domain=Binary, initialize=0)
    model.v = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.k = Var(model.S,                   domain=NonNegativeIntegers, initialize=0)

    model.i = Var(model.S, model.B, model.T, domain=NonNegativeIntegers, initialize=0)
    model.w = Var(model.B,          model.T, domain=NonNegativeIntegers, initialize=0)
    model.p = Var(                   model.T, domain=NonNegativeIntegers, initialize=0)
    model.q = Var(                   model.T, domain=NonNegativeIntegers, initialize=0)

    if es_pila:
        model.g20 = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0)
        model.g40 = Var(model.B, model.T, domain=NonNegativeIntegers, initialize=0)

    # Camila
    model.ygbt      = Var(model.G, model.B, model.T, domain=Binary, initialize=0)
    model.alpha_gbt = Var(model.G, model.B, model.T, domain=Binary, initialize=0)
    model.Z_gb      = Var(model.G, model.B,          domain=Binary, initialize=0)
    model.nRTG      = Var(model.B, model.T,          domain=NonNegativeIntegers, initialize=0)
    model.nRS       = Var(model.B, model.T,          domain=NonNegativeIntegers, initialize=0)
    model.aRTG      = Var(model.B, model.T,          domain=Binary, initialize=0)
    model.aRS       = Var(model.B, model.T,          domain=Binary, initialize=0)


# --- restricciones Magdalena ---

def _restricciones_magdalena(model, es_pila: bool, S40, S20R, S40R):
    # dispersión
    def rule_inv(m, s, b, t):
        if m.CAPBLOCKI[s, t] <= 0:
            return Constraint.Skip
        return m.i[s, b, t] - m.I0[s, b] <= m.CAPBLOCKI[s, t]
    model.constraint_dispersion_inventario = Constraint(model.S, model.B, model.T, rule=rule_inv)

    # Pares de 40' (pila)
    if es_pila:
        model.constraint_pairs_40 = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_pairs_40.add(sum(model.v[s, b, t] for s in S40) <= model.VP[b])

    # Balance de inventario (2)
    model.constraint_2 = ConstraintList()
    T_sorted = sorted(model.T)
    for t in T_sorted:
        for b in model.B:
            for s in model.S:
                if t == T_sorted[0]:
                    model.constraint_2.add(
                        model.i[s, b, t] == model.I0[s, b] + model.fr[s, b, t] + model.fd[s, b, t]
                        - model.fc[s, b, t] - model.fe[s, b, t]
                    )
                else:
                    model.constraint_2.add(
                        model.i[s, b, t] == model.i[s, b, t - 1] + model.fr[s, b, t] + model.fd[s, b, t]
                        - model.fc[s, b, t] - model.fe[s, b, t]
                    )

    # (3) (4) Cotas de ocupación por bahía
    model.constraint_3 = ConstraintList()
    model.constraint_4 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                model.constraint_3.add(model.i[s, b, t] <= model.v[s, b, t] * model.OS * model.C[b])
                model.constraint_4.add(
                    (model.v[s, b, t] - 1) * model.C[b] * model.OS + model.C[b] * model.OI[b]
                    <= model.i[s, b, t]
                )

    # (5-8) Balance de flujos por demanda
    model.constraint_5 = ConstraintList()
    model.constraint_6 = ConstraintList()
    model.constraint_7 = ConstraintList()
    model.constraint_8 = ConstraintList()
    for t in model.T:
        for s in model.S:
            model.constraint_5.add(sum(model.fr[s, b, t] for b in model.B) == model.DR[s, t])
            model.constraint_6.add(sum(model.fc[s, b, t] for b in model.B) == model.DC[s, t])
            model.constraint_7.add(sum(model.fd[s, b, t] for b in model.B) == model.DD[s, t])
            model.constraint_8.add(sum(model.fe[s, b, t] for b in model.B) == model.DE[s, t])

    # (9-10) Activación de y unificada (3 ecuaciones, match pipeline)
    _activacion_y(model)

    # (11) (12) vínculo u ↔ y
    model.constraint_11 = ConstraintList()
    for b in model.B:
        for s in model.S:
            model.constraint_11.add(model.u[s, b] <= sum(model.y[s, b, t] for t in model.T))
    model.constraint_12 = ConstraintList()
    for t in model.T:
        for b in model.B:
            for s in model.S:
                model.constraint_12.add(model.u[s, b] >= model.y[s, b, t])

    # (13) Capacidad de bahías por bloque
    model.constraint_13 = ConstraintList()
    for t in model.T:
        for b in model.B:
            model.constraint_13.add(sum(model.v[s, b, t] * model.TEU[s] for s in model.S) <= model.VS[b])

    # (14) Total bloques por segregación
    model.constraint_14 = ConstraintList()
    for s in model.S:
        model.constraint_14.add(model.k[s] == sum(model.u[s, b] for b in model.B))

    # (16) Cota inferior KI, respetando segregaciones muertas
    model.constraint_16 = ConstraintList()
    for s in model.S:
        entradas_totales   = sum(model.DR[s, t] + model.DD[s, t] for t in model.T)
        inventario_inicial = sum(model.I0[s, b] for b in model.B)
        if entradas_totales == 0 and inventario_inicial == 0:
            model.constraint_16.add(model.k[s] == 0)
        else:
            model.constraint_16.add(model.k[s] >= model.KI[s])

    # (17-20) Workload y desbalance p/q
    model.constraint_17 = ConstraintList()
    model.constraint_18 = ConstraintList()
    model.constraint_19 = ConstraintList()
    for t in model.T:
        for b in model.B:
            model.constraint_17.add(
                model.w[b, t] == sum(
                    model.fr[s, b, t] + model.fc[s, b, t]
                    + model.fd[s, b, t] + model.fe[s, b, t]
                    for s in model.S
                )
            )
            model.constraint_18.add(model.p[t] >= model.w[b, t])
            model.constraint_19.add(model.q[t] <= model.w[b, t])

    model.constraint_20 = ConstraintList()
    for t in model.T:
        model.constraint_20.add(model.p[t] - model.q[t] <= model.r)

    # Reefer (pila) / VSR (bahía)
    if es_pila:
        model.constraint_reefer_plugs = ConstraintList()
        model.constraint_reefer_20    = ConstraintList()
        model.constraint_reefer_40    = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_reefer_plugs.add(model.g20[b, t] + model.g40[b, t] <= model.E[b])
                model.constraint_reefer_20.add(sum(model.v[s, b, t] for s in S20R) <= model.ROWS[b] * model.g20[b, t])
                model.constraint_reefer_40.add(sum(model.v[s, b, t] for s in S40R) <= model.ROWS[b] * model.g40[b, t])
    else:
        model.constraint_21 = ConstraintList()
        for t in model.T:
            for b in model.B:
                model.constraint_21.add(sum(model.v[s, b, t] * model.R[s] for s in model.S) <= model.VSR[b])


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


# --- restricciones Camila (+ acople con Magdalena) ---

def _restricciones_camila(model):
    # Bloquear flujos fuera del dominio S_E↔B_E / S_I↔B_I
    model.bloque_exportacion = ConstraintList()
    model.bloque_importacion = ConstraintList()
    B_E = set(model.B_E); B_I = set(model.B_I)
    if B_E:
        for s in model.S_E:
            for b in model.B:
                if b not in B_E:
                    for t in model.T:
                        model.bloque_exportacion.add(model.fr[s, b, t] == 0)
                        model.bloque_exportacion.add(model.fc[s, b, t] == 0)
    if B_I:
        for s in model.S_I:
            for b in model.B:
                if b not in B_I:
                    for t in model.T:
                        model.bloque_importacion.add(model.fd[s, b, t] == 0)
                        model.bloque_importacion.add(model.fe[s, b, t] == 0)

    # (ACOPLE) Workload total (fr+fc+fd+fe) ≤ capacidad agregada de grúas por (b,t)
    def workload_rule(m, b, t):
        return m.w[b, t] <= m.mu_RTG * m.nRTG[b, t] + m.mu_RS * m.nRS[b, t]
    model.acople_workload = Constraint(model.B, model.T, rule=workload_rule)

    # Capacidad por segregación-bloque (límite de inventario)
    def cap_seg_rule(m, b, s, t):
        return m.i[s, b, t] <= m.Cbs[b, s]
    model.cap_seg = Constraint(model.B, model.S, model.T, rule=cap_seg_rule)

    # Conteo de grúas por tipo por (b,t)
    model.count_rtg = Constraint(model.B, model.T,
        rule=lambda m, b, t: m.nRTG[b, t] == sum(m.ygbt[g, b, t] for g in m.GRT))
    model.count_rs = Constraint(model.B, model.T,
        rule=lambda m, b, t: m.nRS[b, t] == sum(m.ygbt[g, b, t] for g in m.GRS))

    # Exclusividad grúa (1 bloque por periodo)
    model.one_block = Constraint(model.G, model.T,
        rule=lambda m, g, t: sum(m.ygbt[g, b, t] for b in m.B) <= 1)

    # Disponibilidad total por tipo
    model.total_rtg = Constraint(model.T, rule=lambda m, t: sum(m.nRTG[b, t] for b in m.B) <= m.RmaxRTG)
    model.total_rs  = Constraint(model.T, rule=lambda m, t: sum(m.nRS[b, t]  for b in m.B) <= m.RmaxRS)

    # Colisiones: 2·RTG + RS ≤ LIMITE
    model.limit_combinado = Constraint(model.B, model.T,
        rule=lambda m, b, t: 2 * m.nRTG[b, t] + m.nRS[b, t] <= LIMITE_COMBINADO_BT)

    # Máx RTG por bloque
    model.max_rtg_block = Constraint(model.B, model.T,
        rule=lambda m, b, t: m.nRTG[b, t] <= MAX_RTG_POR_BLOQUE)

    # RTG sólo en Costanera
    BC_set = set(model.BC)
    model.rtg_solo_costanera = ConstraintList()
    for b in model.B:
        if b not in BC_set:
            for t in model.T:
                model.rtg_solo_costanera.add(model.nRTG[b, t] == 0)

    # Vínculo activación Big-M (link_rtg/link_rs)
    BIG_RTG = max(1, int(value(model.RmaxRTG)))
    BIG_RS  = max(1, int(value(model.RmaxRS)))
    model.link_rtg = Constraint(model.B, model.T, rule=lambda m, b, t: m.nRTG[b, t] <= BIG_RTG * m.aRTG[b, t])
    model.link_rs  = Constraint(model.B, model.T, rule=lambda m, b, t: m.nRS[b, t]  <= BIG_RS  * m.aRS[b, t])

    # Compatibilidad CBR/CBS a nivel (b,t)
    model.compat_rtg = ConstraintList()
    model.compat_rs  = ConstraintList()
    for t in model.T:
        for b1 in model.B:
            for b2 in model.B:
                if b1 < b2:
                    if int(value(model.CBR[b1, b2])) == 0:
                        model.compat_rtg.add(model.aRTG[b1, t] + model.aRTG[b2, t] <= 1)
                    if int(value(model.CBS[b1, b2])) == 0:
                        model.compat_rs.add(model.aRS[b1, t] + model.aRS[b2, t] <= 1)

    # Vínculo Z ↔ y (global)
    model.Z_y_up = ConstraintList()
    model.y_Z_up = ConstraintList()
    T_len = len(model.T)
    for g in model.G:
        for b in model.B:
            model.Z_y_up.add(model.Z_gb[g, b] <= sum(model.ygbt[g, b, t] for t in model.T))
            model.y_Z_up.add(sum(model.ygbt[g, b, t] for t in model.T) <= model.Z_gb[g, b] * T_len)

    # Exclusividad entre bloques: neutral (EX) + RTG-específica (EX_RTG)
    model.excl_base = ConstraintList()
    for g in model.G:
        for b1 in model.B:
            for b2 in model.B:
                if b1 != b2:
                    model.excl_base.add(model.Z_gb[g, b1] + model.Z_gb[g, b2] <= model.EX[b1, b2])
    model.excl_rtg = ConstraintList()
    for g in model.GRT:
        for b1 in model.B:
            for b2 in model.B:
                if b1 != b2:
                    model.excl_rtg.add(model.Z_gb[g, b1] + model.Z_gb[g, b2] <= model.EX_RTG[b1, b2])

    # Permanencia mínima K_g (lb/lb1/ub/ub1 + alpha_nosolapa)
    T_sorted = sorted(model.T)
    T_max    = T_sorted[-1]
    T_min    = T_sorted[0]

    def lb_rule(m, g, b, t):
        Kg = int(value(m.Kg[g]))
        if t <= T_max - Kg + 1:
            return Kg * m.alpha_gbt[g, b, t] <= sum(
                m.ygbt[g, b, r] for r in m.T if t <= r < t + Kg
            )
        return Constraint.Skip
    model.lb_constraint = Constraint(model.G, model.B, model.T, rule=lb_rule)

    def lb1_rule(m, g, b, t):
        Kg = int(value(m.Kg[g]))
        if t > T_max - Kg + 1:
            return (T_max - t + 1) * m.alpha_gbt[g, b, t] <= sum(
                m.ygbt[g, b, r] for r in m.T if r >= t
            )
        return Constraint.Skip
    model.lb1_constraint = Constraint(model.G, model.B, model.T, rule=lb1_rule)

    def ub_rule(m, g, b, t):
        if t > T_min:
            return m.ygbt[g, b, t] <= m.ygbt[g, b, t - 1] + m.alpha_gbt[g, b, t]
        return Constraint.Skip
    model.ub_constraint = Constraint(model.G, model.B, model.T, rule=ub_rule)

    model.ub1_constraint = Constraint(model.G, model.B,
        rule=lambda m, g, b: m.ygbt[g, b, T_min] <= m.alpha_gbt[g, b, T_min])

    model.alpha_nosolapa = ConstraintList()
    for g in model.G:
        Kg = int(value(model.Kg[g]))
        for b in model.B:
            for t in model.T:
                for r in model.T:
                    if t < r < t + Kg:
                        model.alpha_nosolapa.add(model.alpha_gbt[g, b, t] <= 1 - model.alpha_gbt[g, b, r])


# --- objetivos (ε-constraint) ---

def _definir_objetivos(model):
    model.D = Expression(rule=lambda m: (
        sum(m.fc[s, b, t] * m.LC[s, b] for b in m.B for s in m.S for t in m.T)
        + sum(m.fe[s, b, t] * m.LE[b] for b in m.B for s in m.S for t in m.T)
    ))
    model.B_balance = Expression(rule=lambda m: sum(m.p[t] - m.q[t] for t in m.T))

    model.constr_epsD = Constraint(expr=model.D <= model.eps_D)
    model.constr_epsD.deactivate()

    model.obj_D = Objective(expr=model.D,         sense=minimize)
    model.obj_B = Objective(expr=model.B_balance, sense=minimize)
    model.obj_D.deactivate()
    model.obj_B.deactivate()
