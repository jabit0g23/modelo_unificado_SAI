import os

# Valor por defecto si una hoja W_b viene vacía en la instancia.
W_B_DEFAULT = 3

# Productividad por defecto si la hoja PROD no trae un tipo (mov/turno).
PROD_DEFAULT = 300

# Máximo de grúas RTG simultáneas por bloque (restricción de choque).
MAX_RTG_POR_BLOQUE = 2

# Límite combinado (2·RTG + RS) por (bloque, hora); refleja límite físico del bloque.
LIMITE_COMBINADO_BT = 6

# Tiempo máximo del IIS al diagnosticar infactibilidades (segundos).
IIS_TIME_LIMIT = 100


def solver_options():
    """
    Opciones de Gurobi para el modelo de grúas. Se calcula Threads en base a
    la máquina al llamarse (no al importar) para que el override sea fácil.
    """
    opts = {
        "TimeLimit":    100,
        "Threads":      max(1, (os.cpu_count() or 2) - 1),
        "Presolve":     2,
        "Cuts":         2,
        "Symmetry":     2,
        "Heuristics":   0.10,
        "Seed":         42,
        "Method":       2,
        "LogToConsole": 0,
    }
    if opts["Threads"] >= 4:
        opts["ConcurrentMIP"] = 1
    return opts
