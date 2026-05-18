"""
Orquestador para generar instancias unificadas (horarias) por semana.

Reutiliza los scripts legacy (`analisis_flujos`, `evolucion_turnos`,
`construir_instancia`) pero escribe bajo `instancias_unificadas/{semana}/`
para no colisionar con el pipeline.
"""

import os
from datetime import date

from .scripts.analisis_flujos import run_analysis_flujos
from .scripts.evolucion_turnos import criterioII_a_evolucion
from .scripts.construir_instancia import generar_instancias as _generar_instancia_semana


def _resolver_carpeta_estaticos(semana: str, anio: int, criterio: str,
                                estaticos_dir: str, todas_semanas: list[str]) -> str | None:
    base = os.path.join(estaticos_dir, str(anio), criterio)
    iso_week = date.fromisoformat(semana).isocalendar()[1]
    try:
        ordinal = todas_semanas.index(semana) + 1
    except ValueError:
        ordinal = None
    cand = [
        os.path.join(base, f"Semana {iso_week} - {semana}"),
        os.path.join(base, f"Semana {iso_week:02d} - {semana}"),
        os.path.join(base, f"Semana {iso_week}"),
        os.path.join(base, f"Semana {iso_week:02d}"),
    ]
    if ordinal is not None:
        cand += [
            os.path.join(base, f"Semana {ordinal} - {semana}"),
            os.path.join(base, f"Semana {ordinal:02d} - {semana}"),
            os.path.join(base, f"Semana {ordinal}"),
            os.path.join(base, f"Semana {ordinal:02d}"),
        ]
    cand.append(base)
    for p in cand:
        if os.path.isdir(p):
            return p
    return None


def generar_instancias_unificado(
    semanas: list[str], anio: int,
    resultados_dir: str, estaticos_dir: str, *, cap_mode: str = "bahia",
):
    """
    Genera archivos `Instancia_{semana}_{p}[_K].xlsx` con granularidad horaria
    bajo `{resultados_dir}/instancias_unificadas/{semana}/`.

    Los scripts legacy (analisis_flujos/evolucion/construir_instancia) siguen
    escribiendo bajo `instancias_magdalena/{semana}/`; tras generarlos movemos
    los archivos finales al directorio propio del modelo unificado.
    """
    out_base = os.path.join(resultados_dir, "instancias_unificadas")
    os.makedirs(out_base, exist_ok=True)
    for sem in semanas:
        os.makedirs(os.path.join(out_base, sem), exist_ok=True)

    # Los scripts legacy leen/escriben bajo "instancias_magdalena". Reutilizamos
    # ese staging y luego movemos los xlsx finales.
    magdalena_base = os.path.join(resultados_dir, "instancias_magdalena")
    os.makedirs(magdalena_base, exist_ok=True)

    for sem in semanas:
        print(f"\n(Unificado) ==== Semana {sem}  [cap_mode={cap_mode}] ====")
        os.makedirs(os.path.join(magdalena_base, sem), exist_ok=True)

        # 1) Análisis de flujos → analisis_flujos_w{sem}_0.xlsx
        run_analysis_flujos(sem, resultados_dir=resultados_dir,
                            estaticos_flujos_dir=estaticos_dir,
                            debug=False)

        # 2) Evolución por turnos → evolucion_turnos_w{sem}.xlsx
        evo_path = os.path.join(magdalena_base, sem, f"evolucion_turnos_w{sem}.xlsx")
        if not os.path.isfile(evo_path):
            carpeta_est = _resolver_carpeta_estaticos(sem, anio, "criterio_iii", estaticos_dir, semanas)
            if carpeta_est is None:
                print(f"ADVERTENCIA: no encontré estáticos para {sem}.")
            else:
                try:
                    criterioII_a_evolucion(sem, carpeta_est, evo_path)
                except Exception as e:
                    print(f"ADVERTENCIA: falla evolución {sem}: {e}")

        # 3) Construcción de la instancia horaria
        try:
            _generar_instancia_semana(sem, resultados_dir=resultados_dir,
                                      estaticos_dir=estaticos_dir,
                                      cap_mode=cap_mode)
        except Exception as e:
            print(f"ADVERTENCIA: falla construir_instancia {sem}: {e}")
            continue

        # 4) Mover resultados al directorio del unificado
        src_dir = os.path.join(magdalena_base, sem)
        dst_dir = os.path.join(out_base, sem)
        for nom in os.listdir(src_dir):
            if nom.startswith(f"Instancia_{sem}"):
                os.replace(os.path.join(src_dir, nom), os.path.join(dst_dir, nom))

    print("\n(Unificado) ==== Instancias generadas ====")
