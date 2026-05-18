"""
Cálculos por turno para pasar de los resultados semanales de coloración
a los parámetros que consume el modelo de grúas.

Cada función recibe el estado del turno (DataFrames ya normalizados) y
devuelve un DataFrame listo para escribir como hoja del Excel.
"""

import math
import numpy as np
import pandas as pd


def compute_Gs(df_recibir, df_S_E, turno):
    """Contenedores de export a recibir en el turno (RECV desde gate)."""
    rec = df_recibir.query("Periodo == @turno and S_low in @df_S_E.S_low")
    calc = (
        rec.groupby("S_low", as_index=False)["Recibir"]
           .sum()
           .rename(columns={"S_low": "S_E", "Recibir": "Gs"})
    )
    return (
        pd.DataFrame({"S_E": df_S_E["S_low"].unique()})
          .merge(calc, on="S_E", how="left")
          .fillna({"Gs": 0})
          .astype({"Gs": int})
    )


def _inventario_previo(turno, df_i0, df_volumen, teu_factor_by_S_low, grid_bs):
    """
    Inventario al inicio del turno en contenedores:
      - Turno 1: I0_sb del excel de instancia.
      - Turno >1: Volumen (TEU) del periodo previo / factor TEU.
    """
    if turno == 1:
        i0 = df_i0.groupby(["Bloque", "S_low"], as_index=False)["I0"].sum()
        return (
            grid_bs.merge(i0, on=["Bloque", "S_low"], how="left")
                   .fillna({"I0": 0})
                   .rename(columns={"I0": "cont_prev"})
        )

    prev = df_volumen[df_volumen["Periodo"] == turno - 1].copy()
    if prev.empty:
        prev = df_volumen[df_volumen["Periodo"] == turno].copy()

    prev["teu_factor"] = prev["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
    prev["cont_prev"]  = (prev["Volumen"] / prev["teu_factor"]).round().fillna(0).astype(int)

    inv_prev = (prev.groupby(["B", "S_low"], as_index=False)["cont_prev"].sum()
                     .rename(columns={"B": "Bloque"}))

    return (grid_bs.merge(inv_prev, on=["Bloque", "S_low"], how="left")
                    .fillna({"cont_prev": 0}))


def compute_AEbs_AIbs(turno, df_i0, df_volumen, df_S_E, df_S_I, teu_factor_by_S_low, grid_bs):
    """Inventario inicial por bloque y segregación, separado en export/import."""
    inv = _inventario_previo(turno, df_i0, df_volumen, teu_factor_by_S_low, grid_bs)

    AEbs = (inv[inv["S_low"].isin(df_S_E["S_low"].unique())]
              .rename(columns={"Bloque": "B_E", "S_low": "S_E", "cont_prev": "AEbs"})
              .sort_values(["B_E", "S_E"]))

    AIbs = (inv[inv["S_low"].isin(df_S_I["S_low"].unique())]
              .rename(columns={"Bloque": "B_I", "S_low": "S_I", "cont_prev": "AIbs"})
              .sort_values(["B_I", "S_I"]))

    return AEbs, AIbs


def compute_EIbs(df_entregar, grid_bs, turno):
    """Plan de entrega (import) del turno, por bloque y segregación."""
    if "Entregar" in df_entregar.columns:
        eibs = (df_entregar.query("Periodo == @turno")
                           .groupby(["B", "S_low"], as_index=False)["Entregar"].sum())
    else:
        eibs = pd.DataFrame(columns=["B", "S_low", "Entregar"])

    return (grid_bs.merge(eibs.rename(columns={"B": "Bloque"}),
                          on=["Bloque", "S_low"], how="left")
                    .fillna({"Entregar": 0})
                    .astype({"Entregar": int})
                    .rename(columns={"Bloque": "B_I", "S_low": "S_I", "Entregar": "EIbs"})
                    .sort_values(["B_I", "S_I"]))


def compute_DMEst(dpar, df_S_E, h_ini, h_fin):
    """Demanda de carga (LOAD) de export dentro del turno, por hora relativa 1..8."""
    exp = dpar.query("@h_ini <= T <= @h_fin and S_low in @df_S_E.S_low").copy()
    if not exp.empty:
        exp["T_rel"] = exp["T"] - h_ini + 1
        calc = exp[["S_low", "T_rel", "DC"]].rename(
            columns={"S_low": "S_E", "T_rel": "T", "DC": "DMEst"})
    else:
        calc = pd.DataFrame(columns=["S_E", "T", "DMEst"])

    grid = pd.MultiIndex.from_product(
        [df_S_E.S_low.unique(), range(1, 9)], names=["S_E", "T"]
    ).to_frame(index=False)
    return (grid.merge(calc, on=["S_E", "T"], how="left")
                .fillna({"DMEst": 0})
                .astype({"DMEst": int}))


def compute_DMIst(dpar, df_S_I, h_ini, h_fin):
    """Demanda de descarga (DSCH) de import dentro del turno, por hora relativa 1..8."""
    imp = dpar.query("@h_ini <= T <= @h_fin and S_low in @df_S_I.S_low").copy()
    if not imp.empty:
        imp["T_rel"] = imp["T"] - h_ini + 1
        calc = imp[["S_low", "T_rel", "DD"]].rename(
            columns={"S_low": "S_I", "T_rel": "T", "DD": "DMIst"})
    else:
        calc = pd.DataFrame(columns=["S_I", "T", "DMIst"])

    grid = pd.MultiIndex.from_product(
        [df_S_I.S_low.unique(), range(1, 9)], names=["S_I", "T"]
    ).to_frame(index=False)
    return (grid.merge(calc, on=["S_I", "T"], how="left")
                .fillna({"DMIst": 0})
                .astype({"DMIst": int}))


def compute_Cbs(
    turno, df_bahias, df_i0, all_s_low, bloques,
    cap_unit_map, OS, teu_factor_by_S_low,
):
    """
    Capacidad de contenedores por (bloque, segregación) asignable en el turno.

    La lógica: tomar el máximo entre bahías ocupadas en t y t-1 (el peor caso),
    multiplicar por C[b] * OS para tener TEUs, y dividir por el factor TEU de
    la segregación. En el primer turno se fuerza además una cota mínima a
    partir de I0 para garantizar que el inventario inicial tenga capacidad.
    """
    actual = df_bahias[df_bahias["Periodo"] == turno].copy()
    prev   = df_bahias[df_bahias["Periodo"] == turno - 1].copy()
    if prev.empty:
        prev = actual.copy()

    actual = actual.rename(columns={"Bahías ocupadas": "v"})
    prev   = prev.rename(columns={"Bahías ocupadas": "v_prev"})

    bayas = (pd.merge(actual[["S_low", "B", "v"]],
                      prev[["S_low", "B", "v_prev"]],
                      on=["S_low", "B"], how="outer").fillna(0))

    bayas["max_v"]      = bayas[["v", "v_prev"]].max(axis=1)
    bayas["cap_unit"]   = bayas["B"].map(cap_unit_map).fillna(0).astype(int)
    bayas["cap_teu"]    = bayas["max_v"] * bayas["cap_unit"] * OS
    bayas["teu_factor"] = bayas["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
    bayas["Cbs_calc"]   = np.ceil(bayas["cap_teu"] / bayas["teu_factor"]).astype(int)

    cbs_calc = bayas.groupby(["B", "S_low"])["Cbs_calc"].sum().reset_index()

    grid = pd.MultiIndex.from_product(
        [bloques, all_s_low], names=["B", "S_low"]
    ).to_frame(index=False)
    Cbs_tmp = grid.merge(cbs_calc, on=["B", "S_low"], how="left").fillna({"Cbs_calc": 0})

    if turno == 1:
        # Cota mínima desde I0 para que el inventario inicial siempre quepa
        i0_bs = (df_i0.groupby(["B", "S_low"], as_index=False)["I0"].sum()
                       .rename(columns={"I0": "I0_cont"}))
        i0_bs["teu_factor"] = i0_bs["S_low"].map(lambda s: teu_factor_by_S_low.get(s, 1))
        i0_bs["cap_unit"]   = i0_bs["B"].map(cap_unit_map).fillna(0).astype(int)
        i0_bs["b0"]   = np.ceil((i0_bs["I0_cont"] * i0_bs["teu_factor"]) /
                                (i0_bs["cap_unit"] * OS + 1e-9)).astype(int)
        i0_bs["Cbs0"] = np.ceil((i0_bs["b0"] * i0_bs["cap_unit"] * OS) /
                                (i0_bs["teu_factor"] + 1e-9)).astype(int)

        Cbs_tmp = (Cbs_tmp.merge(i0_bs[["B", "S_low", "Cbs0"]], on=["B", "S_low"], how="left")
                           .fillna({"Cbs0": 0}))
        Cbs_tmp["Cbs_final"] = Cbs_tmp[["Cbs_calc", "Cbs0"]].max(axis=1)
    else:
        Cbs_tmp["Cbs_final"] = Cbs_tmp["Cbs_calc"]

    return (Cbs_tmp[["B", "S_low", "Cbs_final"]]
              .rename(columns={"S_low": "S", "Cbs_final": "Cbs"})
              .astype({"Cbs": int})
              .sort_values(["B", "S"]))
