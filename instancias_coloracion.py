import os
from datetime import date

from codigos.analisis_flujos import run_analysis_flujos
from codigos.evolucion_turnos import criterioII_a_evolucion
from codigos.instancias import generar_instancias


def _resolver_carpeta_estaticos(semana: str, anio: int, criterio: str, estaticos_dir: str, todas_semanas: list[str]) -> str | None:
    """
    Devuelve la carpeta donde están los CSV CriterioII-* para 'semana'.
    Prioriza la semana ISO (correcta) y contempla múltiples variantes de nombre.
    Último fallback: la carpeta base del año/criterio (buscará por patrón dentro).
    """
    base = os.path.join(estaticos_dir, str(anio), criterio)
    # Semana ISO real del año
    iso_week = date.fromisoformat(semana).isocalendar()[1]

    # Número 'ordinal' dentro de tu lista (posible naming legacy)
    try:
        idx = todas_semanas.index(semana)
        ordinal = idx + 1  # base 1
    except ValueError:
        ordinal = None

    candidatos = [
        # Preferimos el nombre correcto por semana ISO
        os.path.join(base, f"Semana {iso_week} - {semana}"),
        os.path.join(base, f"Semana {iso_week:02d} - {semana}"),
        os.path.join(base, f"Semana {iso_week}"),
        os.path.join(base, f"Semana {iso_week:02d}"),
    ]
    # También probamos legacy por ordinal (por si existen así en disco)
    if ordinal is not None:
        candidatos += [
            os.path.join(base, f"Semana {ordinal} - {semana}"),
            os.path.join(base, f"Semana {ordinal:02d} - {semana}"),
            os.path.join(base, f"Semana {ordinal}"),
            os.path.join(base, f"Semana {ordinal:02d}"),
        ]
    # Fallback plano (buscará por patrón dentro)
    candidatos.append(base)

    for p in candidatos:
        if os.path.isdir(p):
            return p
    return None


def _process_semana(semana, criterio, anio, participacion, resultados_dir, estaticos_dir, todas_semanas, *, cap_mode: str):
    print(f"\n(Generar instancia magdalena) ===== PROCESANDO SEMANA: {semana}  [cap_mode={cap_mode}] =====")

    # 2) Análisis de flujos (genera analisis_flujos_w{semana}_0.xlsx)
    run_analysis_flujos(semana, resultados_dir=resultados_dir, estaticos_flujos_dir=estaticos_dir,criterio_flujos=criterio,  debug=False)

    # 3) Evolución por turnos (CriterioII) → evolucion_turnos_w{semana}.xlsx
    out_sem_dir = os.path.join(resultados_dir, "instancias_magdalena", semana)
    os.makedirs(out_sem_dir, exist_ok=True)
    output_evo = os.path.join(out_sem_dir, f"evolucion_turnos_w{semana}.xlsx")

    if not os.path.isfile(output_evo):
        carpeta_estaticos = _resolver_carpeta_estaticos(semana, anio, criterio, estaticos_dir, todas_semanas)
        if carpeta_estaticos is None:
            print(
                f"ADVERTENCIA: No encontré carpeta de estáticos para {semana}. "
                f"Busqué bajo: {os.path.join(estaticos_dir, str(anio), criterio)} "
                f"(variantes 'Semana {{ISO}} - {semana}', 'Semana {{ordinal}} - {semana}', etc.). "
                f"Saltando evolución por turnos."
            )
        else:
            print(f"Generando evolución por turnos (criterio={criterio}) desde CSV en: {carpeta_estaticos}")
            try:
                criterioII_a_evolucion(semana, carpeta_estaticos, output_evo, criterio=criterio)
                print("Evolución por turnos completada.")
            except Exception as e:
                print(f"ADVERTENCIA: Falló la generación de evolución para {semana}: {e}. Continuo sin ella.")

    # 4) Generación de instancias (pasando cap_mode = 'bahia' | 'tier')
    print("Paso 4: Generando instancias…")
    try:
        generar_instancias(semana, resultados_dir=resultados_dir, estaticos_dir=estaticos_dir, participacion_C=participacion, cap_mode=cap_mode)
        print(f"Generación de instancias completada (cap_mode={cap_mode}).")
    except Exception as e:
        print(f"ADVERTENCIA: Falló la generación de instancias para {semana}: {e}. Continúo con la siguiente.")


def generar_instancias_coloracion(semanas, criterio, anio, participacion, resultados_dir, estaticos_dir, *, cap_mode: str = "bahia"):
    """
    cap_mode: 'bahia' o 'tier'
    """
    # 0) Prepara directorios base
    inst_base = os.path.join(resultados_dir, "instancias_magdalena")
    os.makedirs(inst_base, exist_ok=True)

    # 1) Crea subcarpetas por semana
    print("\n(Generar instancia magdalena) ===== CREANDO CARPETAS SEMANALES =====")
    for sem in semanas:
        path = os.path.join(inst_base, sem)
        os.makedirs(path, exist_ok=True)
        print(f" - {path}")
    print("(Generar instancia magdalena) ===== CARPETAS SEMANALES CREADAS =====")

    # 2) Procesa cada semana con el cap_mode elegido
    for sem in semanas:
        _process_semana(
            sem, criterio, anio, participacion, resultados_dir, estaticos_dir, semanas,
            cap_mode=cap_mode
        )

    print("\n(Generar instancia magdalena) ===== PROCESO COMPLETADO PARA TODAS LAS SEMANAS =====")
