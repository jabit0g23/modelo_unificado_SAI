import pandas as pd
import os

RESULTADOS = "/home/javier/GA/resultados_generados_bahia_criterio_ii"
semana = "2022-01-03"
participacion = 68

instancia_path = os.path.join(
    RESULTADOS,
    "instancias_magdalena",
    semana,
    f"Instancia_{semana}_{participacion}_K.xlsx"
)

datos = pd.read_excel(instancia_path, sheet_name=None)
df_D = datos["D_params"]

df_tot = df_D.groupby("T")[["DR", "DC", "DD", "DE"]].sum()
df_tot["demanda"] = df_tot[["DR", "DC", "DD", "DE"]].sum(axis=1)

# capacidad máxima global con todos los RTG y RS encendidos
max_mov = 14*30 + 11*20
df_tot["cap_max"] = max_mov

violados = df_tot[df_tot["demanda"] > df_tot["cap_max"]]
print(violados)
