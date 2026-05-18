import os
import glob
import pandas as pd

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
OUTPUT = os.path.join(os.path.dirname(__file__), "resumen_experimentos.xlsx")


def main():
    rows = []

    for carpeta in sorted(os.listdir(RESULTADOS_DIR)):
        ruta = os.path.join(RESULTADOS_DIR, carpeta)
        if not os.path.isdir(ruta):
            continue

        # Factibles coloracion
        archivos_k = glob.glob(os.path.join(ruta, "resultados_coloracion", "*", "resultado_*_K.xlsx"))
        total_factibles_col = len(archivos_k)

        # Promedio distancias (solo semanas factibles)
        distancias = []
        for archivo_dist in glob.glob(os.path.join(ruta, "resultados_coloracion", "*", "Distancias_Modelo_*.xlsx")):
            try:
                df = pd.read_excel(archivo_dist, usecols=["Distancia Total"])
                distancias.append(df["Distancia Total"].iloc[0])
            except Exception:
                pass
        promedio_distancias = round(sum(distancias) / len(distancias)) if distancias else None

        # Factibles gruas: total de archivos xlsx individuales (turnos)
        archivos_gruas = glob.glob(os.path.join(ruta, "resultados_gruas", "resultados_turno_*", "*.xlsx"))
        total_factibles_gruas = len(archivos_gruas)

        # Tiempo coloracion (suma solve_seconds)
        metrics_col_path = os.path.join(ruta, "metrics", "metrics_coloracion.csv")
        tiempo_coloracion = None
        if os.path.exists(metrics_col_path):
            df_col = pd.read_csv(metrics_col_path)
            tiempo_coloracion = round(df_col["solve_seconds"].sum(), 1)

        # Tiempo gruas (suma solve_seconds)
        metrics_gruas_path = os.path.join(ruta, "resultados_gruas", "metrics", "metrics_gruas.csv")
        tiempo_gruas = None
        if os.path.exists(metrics_gruas_path):
            df_gruas = pd.read_csv(metrics_gruas_path)
            tiempo_gruas = round(df_gruas["solve_seconds"].sum(), 1)

        rows.append({
            "experimento": carpeta,
            "total_factibles_coloracion": total_factibles_col,
            "promedio_distancias": promedio_distancias,
            "total_factibles_gruas": total_factibles_gruas,
            "tiempo_coloracion_s": tiempo_coloracion,
            "tiempo_gruas_s": tiempo_gruas,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_excel(OUTPUT, index=False)
    print(f"\nGuardado en: {OUTPUT}")


if __name__ == "__main__":
    main()
