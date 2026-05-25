import os
import pandas as pd

MODELO = "pipeline"    # "pipeline" / "unificado"

# --- config compartida ---
ANIO = 2022
SEMANAS = ["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24", "2022-01-31"]

USAR_RANGO   = False
ISO_WEEK_INI = 1
ISO_WEEK_FIN = 29

CAP_MODE       = "bahia"        # "pila" / "bahia"
DIAS_HORIZONTE = 7              # 1 (24h) / 2 (48h) / 7 (168h)
USAR_KI_FLUJO  = True
    
THETA_DISPERSION = 1.4

# --- parámetros del modelo de coloración (inyectados a modelo_coloracion/config.py) ---
ALPHA_K      = 2                                        # holgura 
VALOR_BASE_R = {'C': 240, 'H': 160, 'TI': 160}          # desbalance máx p_jt - q_jt por patio
VALOR_BASE_M = {'C': 432, 'H': 216, 'TI': 216}          # tope absoluto w_bt por patio (movs/turno) original es 480, 240, 240

# R menos = {'C': 120, 'H': 80, 'TI': 80}
# R base = {'C': 240, 'H': 160, 'TI': 160}  
# R más = {'C': 480, 'H': 240, 'TI': 240}

# 120% original = {'C': 576, 'H': 288, 'TI': 288}
# 100% original = {'C': 480, 'H': 240, 'TI': 240}
# 90% original = {'C': 432, 'H': 216, 'TI': 216}
# 85% original = {'C': 408, 'H': 204, 'TI': 204}
# 80% original = {'C': 384, 'H': 192, 'TI': 192}
# 75% original = {'C': 360, 'H': 180, 'TI': 180}
# 70% original = {'C': 336, 'H': 168, 'TI': 168}

# --- config pipeline ---
AUX_KI              = 140
UMBRAL_AGRUPACION   = 20
OBJETIVO_GRUAS  = "maxmin"
DIAGNOSTICO_INF = True

N_TURNOS = DIAS_HORIZONTE * 3
TURNOS   = list(range(1, N_TURNOS + 1))

EJECUTAR_PIPELINE = {
    "instancias_coloracion": True,
    "modelo_coloracion":     True,
    "guardar_csv":           False,
    "instancias_gruas":      False,
    "modelo_gruas":          False,
}

# --- config unificado ---
EJECUTAR_UNIFICADO = {
    "instancias": True,
    "modelo":     True,
}


# --- rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS    = os.path.join(BASE_DIR, "datos")

if MODELO == "unificado":
    RESULTADOS = os.path.join(
        BASE_DIR, "resultados",
        f"unificado_{CAP_MODE}_criterio_iii_{DIAS_HORIZONTE}d",
    )
else:
    RESULTADOS = os.path.join(
        BASE_DIR, "resultados",
        f"pipeline_{CAP_MODE}_criterio_iii_{DIAS_HORIZONTE}d_theta{THETA_DISPERSION}_alfa{ALPHA_K}_umbral{UMBRAL_AGRUPACION}_test",
    )

def _run_pipeline(semanas, resultados):
    # Inyectar parámetros de dispersión antes de que se carguen los módulos del modelo
    from modelo_pipeline.modelo_coloracion import config as _col_cfg
    _col_cfg.THETA_DISPERSION = THETA_DISPERSION
    _col_cfg.ALPHA_K          = ALPHA_K
    _col_cfg.VALOR_BASE_R     = VALOR_BASE_R
    _col_cfg.VALOR_BASE_M     = VALOR_BASE_M

    from modelo_pipeline.instancias_coloracion import generar_instancias_coloracion
    from modelo_pipeline.modelo_coloracion     import ejecutar_instancias_coloracion
    from modelo_pipeline.instancias_gruas      import generar_instancias_gruas
    from modelo_pipeline.modelo_gruas          import ejecutar_instancias_gruas

    # 1) Instancias de coloración
    if EJECUTAR_PIPELINE["instancias_coloracion"]:
        print("\n[1/5] Generando instancias de coloración…")
        anio_inst = int(semanas[0][:4])
        generar_instancias_coloracion(
            semanas, anio_inst,
            resultados, DATOS,
            cap_mode=CAP_MODE, aux_ki=AUX_KI, n_turnos=N_TURNOS,
            umbral_agrupacion=UMBRAL_AGRUPACION,
        )
    else:
        print("\n[1/5] Saltando generación de instancias de coloración.")

    # 2) Modelo de coloración
    if EJECUTAR_PIPELINE["modelo_coloracion"]:
        print("\n[2/5] Ejecutando modelo de coloración…")
        semanas_ok, semanas_inf = ejecutar_instancias_coloracion(
            semanas, resultados, usar_ki_flujo=USAR_KI_FLUJO,
            diagnostico_inf=DIAGNOSTICO_INF,
        )
        print(f"OK = {len(semanas_ok)}  |  Infactibles = {len(semanas_inf)}")
    else:
        print("\n[2/5] Saltando modelo de coloración.")
        semanas_ok, semanas_inf = semanas, []

    # 3) CSV semanas
    if EJECUTAR_PIPELINE["guardar_csv"]:
        print("\n[3/5] Guardando CSV de semanas…")
        pd.DataFrame({"semana": semanas_ok}).to_csv(
            os.path.join(resultados, "semanas_filtradas.csv"), index=False)
        pd.DataFrame({"semana": semanas_inf}).to_csv(
            os.path.join(resultados, "semanas_infactibles.csv"), index=False)
        print(f"CSV guardados en {resultados}")
    else:
        print("\n[3/5] Saltando guardado de CSV.")

    # 4) Instancias de grúas
    if EJECUTAR_PIPELINE["instancias_gruas"]:
        print("\n[4/5] Generando instancias por turno…")
        generar_instancias_gruas(semanas_ok, resultados, n_turnos=N_TURNOS, usar_ki_flujo=USAR_KI_FLUJO)
    else:
        print("\n[4/5] Saltando generación de instancias de grúas.")

    # 5) Modelo de grúas
    if EJECUTAR_PIPELINE["modelo_gruas"]:
        print("\n[5/5] Ejecutando modelo de grúas…")
        turnos_str = [f"{t:02d}" for t in TURNOS]
        if OBJETIVO_GRUAS.strip().lower() == "maxmin":
            base_inst = os.path.join(resultados, "instancias_gruas")
            base_res  = os.path.join(resultados, "resultados_gruas")
            os.makedirs(base_res, exist_ok=True)
            ejecutar_instancias_gruas(
                semanas_ok, turnos_str, base_inst, base_res,
                diagnostico_inf=DIAGNOSTICO_INF,
            )
        else:
            raise ValueError("OBJETIVO_GRUAS debe ser 'maxmin'.")
    else:
        print("\n[5/5] Saltando modelo de grúas.")


def _run_unificado(semanas, resultados):
    from modelo_unificado import generar_instancias_unificado, ejecutar_instancias_unificado
    from modelo_unificado import config as uconf

    uconf.DIAS_HORIZONTE   = int(DIAS_HORIZONTE)
    uconf.THETA_DISPERSION = THETA_DISPERSION
    uconf.ALPHA_K          = ALPHA_K
    print(f"Horizonte: {DIAS_HORIZONTE} día(s) = {uconf.horas_horizonte()} h")

    # 1) Instancias unificadas
    if EJECUTAR_UNIFICADO["instancias"]:
        print("\n[1/2] Generando instancias unificadas…")
        generar_instancias_unificado(
            semanas, ANIO,
            resultados, DATOS, cap_mode=CAP_MODE,
            aux_ki=AUX_KI, umbral_agrupacion=UMBRAL_AGRUPACION,
        )
    else:
        print("\n[1/2] Saltando generación de instancias.")

    # 2) Modelo unificado
    if EJECUTAR_UNIFICADO["modelo"]:
        print("\n[2/2] Resolviendo modelo unificado…")
        ok, inf = ejecutar_instancias_unificado(
            semanas, resultados, usar_ki_flujo=USAR_KI_FLUJO,
        )
        print(f"OK = {len(ok)}  |  Infactibles = {len(inf)}")
    else:
        print("\n[2/2] Saltando ejecución del modelo.")

def main():
    from modelo_pipeline.utils.semanas import resolver_semanas

    os.makedirs(RESULTADOS, exist_ok=True)

    semanas = resolver_semanas(SEMANAS, ANIO, USAR_RANGO, ISO_WEEK_INI, ISO_WEEK_FIN)
    if not semanas:
        raise SystemExit("No hay semanas para procesar. Revisa la configuración.")

    preview = ", ".join(semanas[:3]) + (" ..." if len(semanas) > 3 else "")
    print(f"Modelo: {MODELO.upper()}")
    print(f"Semanas ({len(semanas)}): {preview}")
    print(f"CAP_MODE={CAP_MODE}  θ={THETA_DISPERSION}")
    if MODELO == "pipeline":
        print(f"Horizonte: {DIAS_HORIZONTE} día(s) = {N_TURNOS} turnos ({N_TURNOS * 8} h)")
    print(f"Resultados: {RESULTADOS}")

    if MODELO == "pipeline":
        _run_pipeline(semanas, RESULTADOS)
    elif MODELO == "unificado":
        _run_unificado(semanas, RESULTADOS)
    else:
        raise ValueError(f"MODELO debe ser 'pipeline' o 'unificado', no '{MODELO}'.")

    print("\nFlujo finalizado.")


if __name__ == "__main__":
    main()
