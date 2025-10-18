"""
Este archivo lee exactamente una semana desde Flujos.csv (flujos todos los años)
y genera un archivo de flujo de una semana Flujos_w{semana}.xlsx
"""

import pandas as pd
import os
from datetime import timedelta

def extraer_filas_por_fecha(start_date):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "archivos_estaticos", "Flujos.csv")

    df = pd.read_csv(csv_file, sep=";")

    start_datetime = pd.to_datetime(start_date) + pd.Timedelta(hours=8)
    end_datetime = start_datetime + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    
    df['_ime_time_dt'] = pd.to_datetime(df['ime_time'], errors='coerce')
    df_filtrado = df[(df['_ime_time_dt'] >= start_datetime) & (df['_ime_time_dt'] < end_datetime)].copy()
    df_filtrado.drop(columns=['_ime_time_dt'], inplace=True)
    
    fecha_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    output_file = os.path.join(script_dir, "..", "resultados_generados", "instancias_magdalena", f"{start_date}", f"Flujos_w{fecha_str}.xlsx")
    df_filtrado.to_excel(output_file, index=False, sheet_name="Sheet1")
    print(f"Archivo generado: {output_file}")

if __name__ == '__main__':
    extraer_filas_por_fecha("2022-08-29")
