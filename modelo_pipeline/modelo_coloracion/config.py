# --- dispersión (sobreescrito por main.py) ---
THETA_DISPERSION = 1.4

# ALPHA_K, VALOR_BASE_R y VALOR_BASE_M se inyectan desde main.py.

# --- sweep por patio: producto cartesiano de los candidatos ---
ACTIVAR_R_SWEEP = False
R_SWEEP_VALORES = {
    'C':  [50, 100, 150, 300],
    'H':  [50, 100, 150, 300],
    'TI': [50, 100, 150, 300],
}

# --- opciones Gurobi para sweep (corridas rápidas exploratorias) ---
SOLVER_OPTIONS_SWEEP = {
    'LogToConsole': 0,
    'MIPGap':        1e-3,
    'FeasibilityTol': 1e-5,
    'OptimalityTol':  1e-8,
    'IntFeasTol':     1e-5,
    'TimeLimit':      600,
    'MIPFocus':       0.2,
    'Heuristics':     0.2,
    'PumpPasses':     20,
}

# --- tiempo máximo IIS al diagnosticar infactibilidades ---
IIS_TIME_LIMIT = 10000  # segundos

# --- opciones Gurobi para corrida final ---
SOLVER_OPTIONS_FINAL = {
    'LogToConsole': 0,
    'MIPGap':        1e-3, 
    'FeasibilityTol': 1e-3,
    'OptimalityTol':  1e-8,
    'IntFeasTol':     1e-5,
    'TimeLimit':      3600,
    'MIPFocus':       0.2, 
    'Heuristics':     0.2,
    'PumpPasses':     20,
}
