"""
Exportación de resultados del modelo de grúas: vuelca las variables con valor
no nulo a un Excel plano (var, idx, val).
"""

import pandas as pd
from pyomo.environ import Var


def exportar_resultados(model, resultado_xlsx):
    """Escribe un Excel con las variables del modelo cuyo valor es no nulo."""
    filas = []
    for v in model.component_objects(Var, active=True):
        for idx in v:
            val = v[idx].value
            if val:
                filas.append({"var": v.name, "idx": idx, "val": val})
    pd.DataFrame(filas).to_excel(resultado_xlsx, index=False)
