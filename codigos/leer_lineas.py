# ARCHIVO: codigos/leer_lineas.py
"""
Este archivo lee exactamente una semana desde Flujos.csv (flujos todos los años)
y genera un archivo de flujo de una semana Flujos_w{semana}.xlsx
"""

import pandas as pd
import os
from datetime import timedelta

def extraer_filas_por_fecha(start_date, master_csv_path, output_dir):
    """
    Extrae las filas para una semana ISO específica desde el archivo CSV maestro
    y las guarda en el directorio de salida especificado.
    
    Args:
        start_date (str): La fecha de inicio de la semana ISO (ej: "2022-08-29").
        master_csv_path (str): Ruta al archivo "Flujos.csv" maestro.
        output_dir (str): Carpeta donde se guardará el "Flujos_w{...}.xlsx".
    """
    
    try:
        df = pd.read_csv(master_csv_path, sep=";")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo maestro en: {master_csv_path}")
        return
    except Exception as e:
        print(f"ERROR al leer {master_csv_path}: {e}")
        return

    # La semana comienza el Lunes a las 08:00
    start_datetime = pd.to_datetime(start_date) + pd.Timedelta(hours=8)
    # Y termina 7 días después, a las 08:00 (exclusivo)
    end_datetime = start_datetime + pd.Timedelta(days=7)
    
    df['_ime_time_dt'] = pd.to_datetime(df['ime_time'], errors='coerce')
    
    # Filtrado por el rango de fechas
    df_filtrado = df[
        (df['_ime_time_dt'] >= start_datetime) & 
        (df['_ime_time_dt'] < end_datetime)
    ].copy()
    
    df_filtrado.drop(columns=['_ime_time_dt'], inplace=True)
    
    fecha_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    
    # Asegurar que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Construir la ruta de salida
    output_file = os.path.join(output_dir, f"Flujos_w{fecha_str}.xlsx")
    
    df_filtrado.to_excel(output_file, index=False, sheet_name="Sheet1")
    # Silenciamos el print para que no sature la consola en el pre-cálculo
    # print(f"Archivo generado: {output_file}")
