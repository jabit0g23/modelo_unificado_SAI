# Configuración del modelo unificado (Magdalena + Camila, granularidad horaria).
# THETA_DISPERSION y ALPHA_K son inyectados desde main.py antes de cada corrida.

import os

# --- horizonte temporal ---
# Sobreescrito por main.py:  uconf.DIAS_HORIZONTE = DIAS_HORIZONTE
# Valores válidos: 1 (24h) / 2 (48h) / 7 (168h).
DIAS_HORIZONTE: int = 1


def horas_horizonte() -> int:
    return int(DIAS_HORIZONTE) * 24


# --- dispersión (sobreescrito por main.py) ---
THETA_DISPERSION = 1.2

# --- cotas de k_s (sobreescrito por main.py) ---
ALPHA_K = 2


# --- ε-constraint Pareto ---
# Configuración chica para pruebas creíbles sin disparar demasiado el tiempo:
# 2 anclas (D_min, B_min) + 3 puntos del barrido ε = 5 solves por semana.
PARETO_ENABLED = False
PARETO_POINTS  = 3
PARETO_PAD     = 0.05


# --- construcción base de C_bs en la instancia ---
# "global":   C_bs[b,s] = VS_b * C_b  (versión histórica, muy laxa)
# "i0_bays":  solo habilita bloques con I0 positivo y da un buffer fijo
#             de bahías/pilas por bloque activo. Los bloques inactivos
#             solo reciben apertura local, no universal.
CBS_MODE = "global"
CBS_EXTRA_SLOTS = 1
CBS_INACTIVE_SLOTS = 0
CBS_INACTIVE_RADIUS = 2
CBS_ZERO_I0_SEED_BLOCKS = 4


# --- grúas ---
LIMITE_COMBINADO_BT = 6
MAX_RTG_POR_BLOQUE  = 2


# --- solver Gurobi ---
# TimeLimit se calcula dinámicamente según horas_horizonte():
#   1d (24h)  → 120 s/h × 24  =   2880 s
#   2d (48h)  → 120 s/h × 48  =   5760 s
#   7d (168h) → 120 s/h × 168 = 20160 s
TIMELIMIT_PER_HOUR = 120


def solver_timelimit() -> int:
    return max(600, TIMELIMIT_PER_HOUR * horas_horizonte())


def solver_options_for_horizon() -> dict:
    """Opciones de Gurobi adaptadas al horizonte. Para 7d activa barrier + RINS."""
    horas = horas_horizonte()
    timelimit = solver_timelimit()

    opts = {
        "LogToConsole":   0,
        "MIPGap":         1e-3,
        "FeasibilityTol": 1e-5,
        "OptimalityTol":  1e-8,
        "IntFeasTol":     1e-5,
        "MIPFocus":       1,
        "Heuristics":     0.3,
        "PumpPasses":     20,
        "Threads":        max(1, (os.cpu_count() or 2) - 1),
    }
    if horas >= 48:
        opts["Method"]        = 3
        opts["NoRelHeurTime"] = timelimit // 4
        opts["RINS"]          = 50
    return opts
