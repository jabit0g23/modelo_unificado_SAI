"""
Definiciones estáticas del entorno: zonas del puerto, flota de grúas,
compatibilidades y constantes configurables de la instancia.

Las constantes de flota (N_RTG, N_RS, MU_RTG, MU_RS) viven acá porque se
consumen para construir las hojas de la instancia — el modelo de grúas las
lee desde el Excel, no las vuelve a definir.
"""

import pandas as pd
from .lectores import build_compat_df


# ────────────────────────────────────────────────────────────
# Flota y productividades (valores de la operación real)
# ────────────────────────────────────────────────────────────
N_RTG = 14
N_RS  = 11
MU_RTG = 30
MU_RS  = 15

# K por grúa (cantidad mínima de turnos seguidos en el mismo bloque)
K_RTG = 2
K_RS  = 1

# W_b por bloque (capacidad simultánea de grúas por bloque)
W_B_DEFAULT = 3


# ────────────────────────────────────────────────────────────
# Bloques por zona del patio
# ────────────────────────────────────────────────────────────
B_C = [f"b{i}" for i in range(1, 10)]   # Costanera  b1..b9
B_H = [f"h{i}" for i in range(1, 6)]    # O'Higgins  h1..h5
B_T = [f"t{i}" for i in range(1, 5)]    # Tebas      t1..t4
B_I = [f"i{i}" for i in range(1, 3)]    # Imo        i1..i2

BLOQUES = sorted(B_C + B_H + B_T + B_I)


# ────────────────────────────────────────────────────────────
# Adyacencias RTG en Costanera (pares permitidos legacy).
# Se usa para construir EX_RTG: 2 = par permitido, 1 = veto a horizonte.
# ────────────────────────────────────────────────────────────
ADYAC_RTG_B = {
    ('b1', 'b1'), ('b1', 'b3'),
    ('b2', 'b2'), ('b2', 'b4'),
    ('b3', 'b3'), ('b6', 'b3'),
    ('b4', 'b4'), ('b7', 'b4'),
    ('b5', 'b5'), ('b5', 'b8'),
    ('b6', 'b6'),
    ('b7', 'b7'),
    ('b8', 'b8'),
    ('b9', 'b9'),
}


def build_ex_rtg(allowed_pairs=ADYAC_RTG_B):
    """
    Construye EX_RTG[b1,b2] en {1,2}:
        2 si ambos bloques son de Costanera y (b1,b2) es par legacy permitido
          (o su simétrico), o en la diagonal (b,b) de Costanera.
        1 en cualquier otro caso (veto a horizonte).
    """
    allowed_sym = allowed_pairs | {(b2, b1) for (b1, b2) in allowed_pairs}
    rows = []
    for b1 in BLOQUES:
        for b2 in BLOQUES:
            if (b1 in B_C) and (b2 in B_C) and ((b1, b2) in allowed_sym):
                ex = 2
            elif (b1 == b2) and (b1 in B_C):
                ex = 2
            else:
                ex = 1
            rows.append({"b1": b1, "b2": b2, "EX": ex})
    return pd.DataFrame(rows)


def build_static_sheets():
    """
    Construye todas las hojas estáticas (flota, zonas, compatibilidades, etc.)
    que se replican igual en todos los Excel de turno.

    Retorna un dict {nombre_hoja → DataFrame} listo para pasarle al ExcelWriter.
    """
    G_RTG = [f"rtg{i}" for i in range(1, N_RTG + 1)]
    G_RS  = [f"rs{i}"  for i in range(1, N_RS + 1)]
    G_ALL = G_RTG + G_RS

    sheets = {
        "G":        pd.DataFrame({"G": G_ALL}),
        "GRT":      pd.DataFrame({"GRT": G_RTG}),
        "GRS":      pd.DataFrame({"GRS": G_RS}),
        "B":        pd.DataFrame({"B": BLOQUES}),
        "B_E":      pd.DataFrame({"B_E": BLOQUES}),
        "B_I":      pd.DataFrame({"B_I": BLOQUES}),
        "BC":       pd.DataFrame({"BC": B_C}),
        "BT":       pd.DataFrame({"BT": B_T}),
        "BH":       pd.DataFrame({"BH": B_H}),
        "BI":       pd.DataFrame({"BI": B_I}),
        "T":        pd.DataFrame({"T": list(range(1, 9))}),
        "mu":       pd.DataFrame({"mu": [MU_RTG]}),           # referencia legacy
        "W":        pd.DataFrame({"W": [W_B_DEFAULT]}),        # referencia legacy
        "K":        pd.DataFrame({"K": [K_RTG]}),              # referencia legacy
        "W_b":      pd.DataFrame({"B": BLOQUES, "W_b": [W_B_DEFAULT] * len(BLOQUES)}),
        "Rmax_rtg": pd.DataFrame({"Rmax": [N_RTG]}),
        "Rmax_rs":  pd.DataFrame({"Rmax": [N_RS]}),
        # Compatibilidades para simultaneidad (1 = permitido para todos)
        "CBR":      build_compat_df(BLOQUES, "CBR"),
        "CBS":      build_compat_df(BLOQUES, "CBS"),
        # Exclusividad a horizonte para RTG (1 veto, 2 permitido)
        "EX_RTG":   build_ex_rtg(),
        "K_g":      pd.DataFrame({"G": G_ALL, "K": [K_RTG] * len(G_RTG) + [K_RS] * len(G_RS)}),
        "PROD":     pd.DataFrame([
            {"Tipo": "RTG", "Prod": MU_RTG},
            {"Tipo": "RS",  "Prod": MU_RS},
        ]),
    }
    return sheets, BLOQUES
