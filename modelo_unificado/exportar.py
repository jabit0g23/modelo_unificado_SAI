"""Escritura de resultados del modelo unificado a xlsx/csv."""

import os
import pandas as pd
from pyomo.environ import value


def exportar_resultados(
    model, ctx: dict, *, semana: str,
    resultado_xlsx: str, pareto_csv: str | None, pareto_rows: list,
):
    os.makedirs(os.path.dirname(resultado_xlsx), exist_ok=True)

    segregacion_map = ctx["segregacion_map"]
    bloque_id_map   = ctx["bloque_id_map"]
    seg_id_map      = ctx["seg_id_map"]

    # Pareto CSV
    if pareto_csv and pareto_rows:
        os.makedirs(os.path.dirname(pareto_csv), exist_ok=True)
        pd.DataFrame(pareto_rows).to_csv(pareto_csv, index=False)

    # Flujos
    df_fr = pd.DataFrame(
        [(s, b, t, model.fr[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Recibir"])
    df_fc = pd.DataFrame(
        [(s, b, t, model.fc[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Cargar"])
    df_fd = pd.DataFrame(
        [(s, b, t, model.fd[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Descargar"])
    df_fe = pd.DataFrame(
        [(s, b, t, model.fe[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Entregar"])

    df_y = pd.DataFrame(
        [(s, b, t, model.y[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Asignado"])

    df_i = pd.DataFrame(
        [(s, b, t, model.i[s, b, t].value) for s in model.S for b in model.B for t in model.T],
        columns=["Segregación", "Bloque", "Periodo", "Inventario"])

    df_w = pd.DataFrame(
        [(b, t, model.w[b, t].value, bloque_id_map[b]) for b in model.B for t in model.T],
        columns=["Bloque", "Periodo", "Workload", "BloqueID"])

    df_pq = pd.DataFrame(
        [(t, model.p[t].value, model.q[t].value) for t in model.T],
        columns=["Periodo", "CargaMax", "CargaMin"])

    # Grúas
    df_y_g = pd.DataFrame(
        [(g, b, t, model.ygbt[g, b, t].value)
         for g in model.G for b in model.B for t in model.T],
        columns=["Grúa", "Bloque", "Periodo", "Asignado"])

    df_nrtg = pd.DataFrame(
        [(b, t, model.nRTG[b, t].value, model.nRS[b, t].value)
         for b in model.B for t in model.T],
        columns=["Bloque", "Periodo", "nRTG", "nRS"])

    df_Z = pd.DataFrame(
        [(g, b, model.Z_gb[g, b].value) for g in model.G for b in model.B],
        columns=["Grúa", "Bloque", "Z"])

    # Resumen
    D_val = float(value(model.D))
    B_val = float(value(model.B_balance))
    df_resumen = pd.DataFrame([{
        "Semana": semana,
        "Horas": len(list(model.T)), "Segregaciones": len(list(model.S)),
        "Bloques": len(list(model.B)), "Grúas": len(list(model.G)),
        "D (distancia)": D_val, "B (desbalance)": B_val,
    }])

    with pd.ExcelWriter(resultado_xlsx, engine="openpyxl") as w:
        df_resumen.to_excel(w, sheet_name="Resumen", index=False)
        if pareto_rows:
            pd.DataFrame(pareto_rows).to_excel(w, sheet_name="Pareto", index=False)
        df_fr.to_excel(w, sheet_name="Recibir",      index=False)
        df_fc.to_excel(w, sheet_name="Cargar",       index=False)
        df_fd.to_excel(w, sheet_name="Descargar",    index=False)
        df_fe.to_excel(w, sheet_name="Entregar",     index=False)
        df_y.to_excel(w,  sheet_name="Asignado",     index=False)
        df_i.to_excel(w,  sheet_name="Inventario",   index=False)
        df_w.to_excel(w,  sheet_name="Workload",     index=False)
        df_pq.to_excel(w, sheet_name="Carga máx-min", index=False)
        df_y_g.to_excel(w, sheet_name="Grúas_asign", index=False)
        df_nrtg.to_excel(w, sheet_name="nRTG_nRS",   index=False)
        df_Z.to_excel(w,  sheet_name="Z_global",     index=False)

    print(f"[{semana}] Resultados guardados en {resultado_xlsx}  (D={D_val:.2f}, B={B_val:.2f})")
    return {"D": D_val, "B": B_val}
