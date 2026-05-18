"""
Helpers para generar listas de semanas ISO.

Se usan desde main.py para construir la lista de semanas a procesar, ya sea
explícita, por rango ISO o todas las del año.
"""

from datetime import date


def generar_semanas_iso(year: int) -> list[str]:
    """Lista todas las semanas ISO del año, formateadas como 'YYYY-MM-DD' (lunes)."""
    last_week = date(year, 12, 28).isocalendar()[1]
    return [date.fromisocalendar(year, w, 1).isoformat() for w in range(1, last_week + 1)]


def generar_semanas_rango(year: int, w_ini: int, w_fin: int) -> list[str]:
    """Lista las semanas ISO [w_ini, w_fin] del año como 'YYYY-MM-DD' (lunes)."""
    if w_fin < w_ini:
        raise ValueError("ISO_WEEK_FIN no puede ser menor que ISO_WEEK_INI")
    return [date.fromisocalendar(year, w, 1).isoformat() for w in range(w_ini, w_fin + 1)]


def resolver_semanas(semanas_fijas, anio, usar_rango, w_ini, w_fin) -> list[str]:
    """
    Resuelve la lista final de semanas a procesar siguiendo esta precedencia:
      1. `semanas_fijas` si no está vacía.
      2. Rango ISO si `usar_rango`.
      3. Todas las semanas del año.
    """
    if semanas_fijas:
        print(f"Usando lista fija de {len(semanas_fijas)} semana(s).")
        return semanas_fijas
    if usar_rango:
        semanas = generar_semanas_rango(anio, w_ini, w_fin)
        print(f"Generadas {len(semanas)} semanas por rango ISO [{w_ini}..{w_fin}] del {anio}.")
        return semanas
    semanas = generar_semanas_iso(anio)
    print(f"Generadas {len(semanas)} semanas ISO del año {anio}.")
    return semanas
