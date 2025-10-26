import os
from datetime import date
import pandas as pd

from instancias_coloracion import generar_instancias_coloracion
from modelo_coloracion import ejecutar_instancias_coloracion
from instancias_gruas import generar_instancias_gruas
from modelo_gruas_maxmin import ejecutar_instancias_gruas_maxmin
from modelo_gruas_minmax import ejecutar_instancias_gruas_minmax  # opcional

# ───────── CONFIGURACIÓN EDITABLE ─────────

#["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24", "2022-01-31", "2022-02-07", "2022-02-14", "2022-02-21", "2022-02-28", "2022-03-07", "2022-03-14", "2022-03-21", "2022-04-04", "2022-04-11", "2022-04-18", "2022-04-25", "2022-05-02", "2022-05-09", "2022-05-16", "2022-05-23", "2022-05-30", "2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27", "2022-07-04", "2022-07-11", "2022-07-18", "2022-07-25", "2022-08-01", "2022-08-08", "2022-08-15", "2022-08-22", "2022-08-29", "2022-09-05", "2022-09-12", "2022-09-19", "2022-09-26", "2022-10-03", "2022-10-10", "2022-10-17", "2022-10-24", "2022-10-31", "2022-11-07", "2022-11-14", "2022-11-21", "2022-11-28", "2022-12-05", "2022-12-12", "2022-12-19", "2022-12-26"]  


# (A) Semanas
ANIO = 2022
SEMANAS =  ["2022-02-21", "2022-03-14", "2022-05-23", "2022-06-06", "2022-06-13", "2022-07-04", "2022-08-08", "2022-09-05", "2022-12-12", "2022-12-26"]  # lista semanas del año

USAR_RANGO = False
ISO_WEEK_INI = 41
ISO_WEEK_FIN = 52

# (B) Parámetros generales
PARTICIPACION = 68
CRITERIO = "criterio_ii" # "criterio_ii" / "criterio_iii"
OBJETIVO_GRUAS = "maxmin"  # "maxmin" o "minmax"
TURNOS = list(range(1, 22))  # 1..21
CAP_MODE = "bahia"   # cambiar entre "pila" y "bahia"

# (C) Switches de pasos
EJECUTAR = {
    "instancias_coloracion": False,
    "modelo_coloracion":     False,
    "guardar_csv":           False,
    "instancias_gruas":      True,
    "modelo_gruas":          True,
}

# (D) ►► Switches/valores de RESTRICCIONES (Magdalena) ◄◄
RESTRICCIONES_MAGDALENA = {
    "usar_cota_inferior": True,
    "beta_alpha":         0.8,
    "usar_cota_superior": False,
    "gamma":              0.2,
}

# (E) Rutas
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ESTATICOS  = os.path.join(BASE_DIR, "archivos_estaticos")
RESULTADOS = os.path.join(BASE_DIR, f"resultados_generados_{CAP_MODE}_{CRITERIO}_k08")
BASE_INST  = os.path.join(RESULTADOS, "instancias_camila")
BASE_RES   = os.path.join(RESULTADOS, "resultados_camila")

# ───────── Helpers de semanas ─────────

def generar_semanas_iso(year: int):
    last_week = date(year, 12, 28).isocalendar()[1]
    return [date.fromisocalendar(year, w, 1).isoformat() for w in range(1, last_week + 1)]

def generar_semanas_rango(year: int, w_ini: int, w_fin: int):
    if w_fin < w_ini:
        raise ValueError("ISO_WEEK_FIN no puede ser menor que ISO_WEEK_INI")
    return [date.fromisocalendar(year, w, 1).isoformat() for w in range(w_ini, w_fin + 1)]

def resolver_semanas():
    if SEMANAS:
        print(f"Usando lista fija de {len(SEMANAS)} semana(s).")
        return SEMANAS
    if USAR_RANGO:
        semanas = generar_semanas_rango(ANIO, ISO_WEEK_INI, ISO_WEEK_FIN)
        print(f"Generadas {len(semanas)} semanas por rango ISO [{ISO_WEEK_INI}..{ISO_WEEK_FIN}] del {ANIO}.")
        return semanas
    semanas = generar_semanas_iso(ANIO)
    print(f"Generadas {len(semanas)} semanas ISO del año {ANIO}.")
    return semanas

# ───────── Orquestación ─────────

def ensure_dirs():
    os.makedirs(RESULTADOS, exist_ok=True)
    os.makedirs(BASE_INST, exist_ok=True)
    os.makedirs(BASE_RES, exist_ok=True)

def main():
    ensure_dirs()
    semanas = resolver_semanas()
    if not semanas:
        raise SystemExit("No hay semanas para procesar. Revisa la configuración.")

    preview = ", ".join(semanas[:3]) + (" ..." if len(semanas) > 3 else "")
    print(f"Semanas seleccionadas ({len(semanas)}): {preview}")
    print(f"Modo de capacidad seleccionado: CAP_MODE = '{CAP_MODE}'")

    # 1) Instancias de coloración (Magdalena)
    if EJECUTAR["instancias_coloracion"]:
        print("\n[1/5] Generando instancias de coloración (Magdalena)…")
        anio_para_instancias = int(semanas[0][:4])
        generar_instancias_coloracion(
            semanas, CRITERIO, anio_para_instancias,
            PARTICIPACION, RESULTADOS, ESTATICOS,
            cap_mode=CAP_MODE,  # ← pasamos el modo aquí
        )
    else:
        print("\n[1/5] Saltando generación de instancias de coloración.")

    # 2) Ejecutar modelo de coloración (Magdalena)
    if EJECUTAR["modelo_coloracion"]:
        print("\n[2/5] Ejecutando modelo de coloración (Magdalena)…")
        semanas_filtradas, semanas_infactibles = ejecutar_instancias_coloracion(
            semanas,
            PARTICIPACION,
            RESULTADOS,
            usar_cota_inferior=RESTRICCIONES_MAGDALENA["usar_cota_inferior"],
            beta_alpha=RESTRICCIONES_MAGDALENA["beta_alpha"],
            usar_cota_superior=RESTRICCIONES_MAGDALENA["usar_cota_superior"],
            gamma_val=RESTRICCIONES_MAGDALENA["gamma"],
        )
        print(f"Procesamiento OK = {len(semanas_filtradas)}")
        print(f"Semanas infactibles = {len(semanas_infactibles)}")
    else:
        print("\n[2/5] Saltando modelo de coloración.")
        semanas_filtradas, semanas_infactibles = semanas, []

    # 3) Guardar listados a CSV
    if EJECUTAR["guardar_csv"]:
        print("\n[3/5] Guardando CSV de semanas…")
        pd.DataFrame({"semana": semanas_filtradas}).to_csv(
            os.path.join(RESULTADOS, "semanas_filtradas.csv"), index=False
        )
        pd.DataFrame({"semana": semanas_infactibles}).to_csv(
            os.path.join(RESULTADOS, "semanas_infactibles.csv"), index=False
        )
        print(f"CSV guardados en {RESULTADOS}:\n - semanas_filtradas.csv\n - semanas_infactibles.csv")
    else:
        print("\n[3/5] Saltando guardado de CSV.")

    # 4) Instancias de grúas (Camila)
    if EJECUTAR["instancias_gruas"]:
        print("\n[4/5] Generando instancias por turno (Camila)…")
        generar_instancias_gruas(semanas_filtradas, PARTICIPACION, RESULTADOS)
    else:
        print("\n[4/5] Saltando generación de instancias de grúas.")

    # 5) Ejecutar modelo de grúas (Camila)
    if EJECUTAR["modelo_gruas"]:
        print("\n[5/5] Ejecutando modelo de grúas (Camila)…")
        turnos_str = [f"{t:02d}" for t in TURNOS]
        objetivo = OBJETIVO_GRUAS.strip().lower()
        if objetivo == "maxmin":
            ejecutar_instancias_gruas_maxmin(
                semanas_filtradas, turnos_str, PARTICIPACION, BASE_INST, BASE_RES
            )
        elif objetivo == "minmax":
            ejecutar_instancias_gruas_minmax(
                semanas_filtradas, turnos_str, PARTICIPACION, BASE_INST, BASE_RES
            )
        else:
            raise ValueError("OBJETIVO_GRUAS debe ser 'maxmin' o 'minmax'.")
    else:
        print("\n[5/5] Saltando modelo de grúas.")

    print("\n✅ Flujo finalizado.")

if __name__ == "__main__":
    main()
