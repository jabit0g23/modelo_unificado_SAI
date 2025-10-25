# ARCHIVO: precalcular_flujos.py
import os
import sys
from datetime import date, timedelta

# Aseguramos que 'codigos' esté en el path para la importación
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "codigos"))

try:
    from leer_lineas import extraer_filas_por_fecha
except ImportError:
    print("Error: No se pudo importar 'extraer_filas_por_fecha' desde la carpeta 'codigos'.")
    print("Asegúrate de que 'codigos' exista y contenga 'leer_lineas.py'.")
    sys.exit(1)


# --- Configuración ---
# ▼▼▼ EDITA LAS OPCIONES DE FECHAS ▼▼▼

ANIO = 2023 # Año base para el rango ISO o el año completo

# Opción 1: Procesar un rango de semanas ISO (ej: de la 35 a la 40)
USAR_RANGO_ISO = False
ISO_WEEK_INI = 42
ISO_WEEK_FIN = 52

# Opción 2: Procesar desde una fecha de inicio hasta fin de año
# (Se ignora si USAR_RANGO_ISO = True)
PROCESAR_DESDE_FECHA = False
FECHA_INICIO = "2022-09-01" # Formato YYYY-MM-DD

# Si ambas (USAR_RANGO_ISO y PROCESAR_DESDE_FECHA) son False,
# se procesa el AÑO completo.
# ---------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "archivos_estaticos", "Flujos.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "archivos_estaticos", "flujos")

def generar_semanas_iso(year: int):
    """
    Genera todas las semanas ISO (Lunes) para un año dado.
    """
    try:
        last_week = date(year, 12, 28).isocalendar()[1]
        # Aseguramos que también funcione para años con 53 semanas
        if last_week == 1 and date(year, 12, 28).month == 12:
             last_week = 53
        return [date.fromisocalendar(year, w, 1).isoformat() for w in range(1, last_week + 1)]
    except ValueError:
         # Maneja el caso de años bisiestos que terminan en semana 1 (raro)
        try:
            last_week = 52
            return [date.fromisocalendar(year, w, 1).isoformat() for w in range(1, last_week + 1)]
        except Exception as e:
            print(f"Error crítico generando semanas para {year}: {e}")
            return []


def generar_semanas_rango(year: int, w_ini: int, w_fin: int):
    """
    Genera semanas ISO (Lunes) para un rango específico.
    (Copiado desde main.py)
    """
    if w_fin < w_ini:
        raise ValueError("ISO_WEEK_FIN no puede ser menor que ISO_WEEK_INI")
    return [date.fromisocalendar(year, w, 1).isoformat() for w in range(w_ini, w_fin + 1)]

def resolver_semanas():
    """
    Decide qué lista de semanas procesar basado en la configuración.
    """
    print("Resolviendo semanas a procesar...")
    
    # Generamos todas las semanas del año como base
    try:
        todas_las_semanas = generar_semanas_iso(ANIO)
        if not todas_las_semanas:
            raise ValueError(f"No se generaron semanas para el año {ANIO}.")
    except Exception as e:
        print(f"Error generando semanas base para el año {ANIO}: {e}")
        return []

    # Opción 1: Rango ISO
    if USAR_RANGO_ISO:
        print(f"Modo: Rango ISO [{ISO_WEEK_INI}-{ISO_WEEK_FIN}] del {ANIO}.")
        try:
            semanas = generar_semanas_rango(ANIO, ISO_WEEK_INI, ISO_WEEK_FIN)
            return semanas
        except Exception as e:
            print(f"Error generando rango ISO: {e}")
            return []
    
    # Opción 2: Desde Fecha
    if PROCESAR_DESDE_FECHA:
        print(f"Modo: Desde fecha [{FECHA_INICIO}] hasta fin del año {ANIO}.")
        try:
            # Validar y encontrar la semana ISO de inicio
            fecha_ini_dt = date.fromisoformat(FECHA_INICIO)
            # Encontrar el lunes de esa semana
            lunes_de_inicio = (fecha_ini_dt - timedelta(days=fecha_ini_dt.weekday())).isoformat()
            
            # Filtrar la lista completa
            semanas_filtradas = [s for s in todas_las_semanas if s >= lunes_de_inicio]
            if not semanas_filtradas:
                print(f"Advertencia: La fecha {FECHA_INICIO} es posterior a la última semana del año.")
            return semanas_filtradas
        except Exception as e:
            print(f"Error procesando fecha de inicio '{FECHA_INICIO}': {e}")
            return []

    # Opción 3: Año Completo (default)
    print(f"Modo: Año completo ({ANIO}).")
    return todas_las_semanas

def main():
    print(f"Iniciando pre-cálculo de flujos...")
    print(f"Archivo Maestro: {MASTER_CSV}")
    print(f"Directorio Salida: {OUTPUT_DIR}")
    
    if not os.path.exists(MASTER_CSV):
        print(f"ERROR: No se encuentra {MASTER_CSV}. Abortando.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    semanas = resolver_semanas()
    
    if not semanas:
        print("No se encontraron semanas para procesar. Revisa la configuración.")
        return

    print(f"Se procesarán {len(semanas)} semanas.")
    
    total_semanas = len(semanas)
    for i, semana in enumerate(semanas):
        print(f"--- Procesando semana {i+1}/{total_semanas}: {semana} ---")
        try:
            # Llamamos a la función modificada
            extraer_filas_por_fecha(semana, MASTER_CSV, OUTPUT_DIR)
        except Exception as e:
            print(f"ERROR al procesar la semana {semana}: {e}")
    
    print(f"\n✅ Pre-cálculo de flujos completado. {total_semanas} archivos generados en:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    main()