# Configuración del modelo unificado (Magdalena + Camila, granularidad horaria).
# THETA_DISPERSION es inyectado desde main.py antes de cada corrida.

import os

# --- horizonte temporal ---
# Sobreescrito por main.py:  uconf.DIAS_HORIZONTE = DIAS_HORIZONTE
# Valores válidos: 1 (24h) / 2 (48h) / 7 (168h).
DIAS_HORIZONTE: int = 7


def horas_horizonte() -> int:
    return int(DIAS_HORIZONTE) * 24


# --- dispersión (sobreescrito por main.py) ---
THETA_DISPERSION = 1.2


# --- cota laxa p[t]-q[t] ≤ R (no es la FO, solo estabiliza el LP) ---
VALOR_BASE_R = 10_000


# --- ε-constraint Pareto ---
PARETO_ENABLED = True
PARETO_POINTS  = 12
PARETO_PAD     = 0.05


# --- grúas ---
LIMITE_COMBINADO_BT = 6
MAX_RTG_POR_BLOQUE  = 2


# --- solver Gurobi ---
# TimeLimit se calcula dinámicamente según horas_horizonte():
#   1d (24h)  → 100 s/h × 24  =   2400 s
#   2d (48h)  → 100 s/h × 48  =   4800 s
#   7d (168h) → 100 s/h × 168 = 16800 s
TIMELIMIT_PER_HOUR = 100


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
