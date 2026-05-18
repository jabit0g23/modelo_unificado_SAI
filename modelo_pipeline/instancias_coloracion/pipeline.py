"""
Pipeline semanal para generar las instancias del modelo de coloración.

Por cada semana ejecuta: análisis de flujos → evolución por turnos → armado del
archivo Excel Instancia_*.xlsx que luego consume el modelo.
"""

import os
from datetime import date

from .flujos import run_analysis_flujos
from .evolucion import criterioII_a_evolucion
from .construir import generar_instancias


def _resolver_carpeta_estaticos(
    semana: str,
    anio: int,
    criterio: str,
    estaticos_dir: str,
    todas_semanas: list[str],
) -> str | None:
    """
    Devuelve la carpeta donde están los CSV CriterioII-* para `semana`.

    Prueba varios nombres porque el dataset histórico mezcla convenciones:
    por semana ISO real, con/ sin cero a la izquierda, o por índice ordinal
    dentro de la lista (naming legacy).
    """
    base = os.path.join(estaticos_dir, str(anio), criterio)
    iso_week = date.fromisoformat(semana).isocalendar()[1]

    try:
        ordinal = todas_semanas.index(semana) + 1
    except ValueError:
        ordinal = None

    candidatos = [
        os.path.join(base, f"Semana {iso_week} - {semana}"),
        os.path.join(base, f"Semana {iso_week:02d} - {semana}"),
        os.path.join(base, f"Semana {iso_week}"),
        os.path.join(base, f"Semana {iso_week:02d}"),
    ]
    if ordinal is not None:
        candidatos += [
            os.path.join(base, f"Semana {ordinal} - {semana}"),
            os.path.join(base, f"Semana {ordinal:02d} - {semana}"),
            os.path.join(base, f"Semana {ordinal}"),
            os.path.join(base, f"Semana {ordinal:02d}"),
        ]
    candidatos.append(base)

    for p in candidatos:
        if os.path.isdir(p):
            return p
    return None


def _process_semana(
    semana, anio,
    resultados_dir, estaticos_dir, todas_semanas,
    *, cap_mode: str, aux_ki, n_turnos: int = 21, umbral_agrupacion: int = 0,
):
    print(f"\n===== PROCESANDO SEMANA: {semana}  [cap_mode={cap_mode}] =====")

    # Genera analisis_flujos_w{semana}_0.xlsx
    run_analysis_flujos(
        semana,
        resultados_dir=resultados_dir,
        estaticos_flujos_dir=estaticos_dir,
        debug=False,
    )

    # Genera evolucion_turnos_w{semana}.xlsx si aún no existe
    out_sem_dir = os.path.join(resultados_dir, "instancias_coloracion", semana)
    os.makedirs(out_sem_dir, exist_ok=True)
    output_evo = os.path.join(out_sem_dir, f"evolucion_turnos_w{semana}.xlsx")

    if not os.path.isfile(output_evo):
        carpeta_estaticos = _resolver_carpeta_estaticos(
            semana, anio, "criterio_iii", estaticos_dir, todas_semanas
        )
        if carpeta_estaticos is None:
            print(
                f"ADVERTENCIA: No encontré carpeta de estáticos para {semana}. "
                f"Busqué bajo: {os.path.join(estaticos_dir, str(anio), 'criterio_iii')}. "
                f"Saltando evolución por turnos."
            )
        else:
            print(f"Generando evolución por turnos desde: {carpeta_estaticos}")
            try:
                criterioII_a_evolucion(semana, carpeta_estaticos, output_evo)
            except Exception as e:
                print(f"ADVERTENCIA: Falló la evolución para {semana}: {e}. Continúo.")

    # Genera el archivo Instancia_*.xlsx consumido por el modelo
    try:
        generar_instancias(
            semana,
            resultados_dir=resultados_dir,
            estaticos_dir=estaticos_dir,
            cap_mode=cap_mode,
            aux_ki=aux_ki,
            n_turnos=n_turnos,
            umbral_agrupacion=umbral_agrupacion,
        )
    except Exception as e:
        print(f"ADVERTENCIA: Falló la generación de instancia para {semana}: {e}.")


def generar_instancias_coloracion(
    semanas, anio,
    resultados_dir, estaticos_dir,
    *, cap_mode=None, aux_ki=None, n_turnos: int = 21, umbral_agrupacion: int = 0,
):
    """
    Args:
        cap_mode: 'bahia' o 'pila' — unidad sobre la que se contabiliza capacidad.
        aux_ki: umbral base (contenedores) para el cálculo por tramos de KI.
        n_turnos: turnos a modelar (3=1día, 6=2días, 21=semana completa).
    """
    inst_base = os.path.join(resultados_dir, "instancias_coloracion")
    os.makedirs(inst_base, exist_ok=True)

    for sem in semanas:
        os.makedirs(os.path.join(inst_base, sem), exist_ok=True)

    for sem in semanas:
        _process_semana(
            sem, anio,
            resultados_dir, estaticos_dir, semanas,
            cap_mode=cap_mode, aux_ki=aux_ki, n_turnos=n_turnos,
            umbral_agrupacion=umbral_agrupacion,
        )

    print("\n===== GENERACIÓN DE INSTANCIAS DE COLORACIÓN COMPLETADA =====")
