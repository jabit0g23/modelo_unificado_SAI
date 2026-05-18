import pandas as pd
from pathlib import Path
import re


def normalizar_numero(x):
    """
    Convierte valores tipo '6,417,108' a número.
    Si ya viene numérico desde Excel, lo deja bien.
    """
    if pd.isna(x):
        return None

    if isinstance(x, (int, float)):
        return x

    s = str(x).strip()

    # Quita separador de miles tipo 6,417,108
    s = s.replace(",", "")

    try:
        # Si termina siendo entero, lo deja como int
        val = float(s)
        return int(val) if val.is_integer() else val
    except ValueError:
        return x


def main():
    # Carpeta donde está este script
    base_dir = Path(__file__).resolve().parent

    root_dir = base_dir / "resultados_generados_pila_criterio_iii"
    resultados_dir = root_dir / "resultados_magdalena"
    metrics_dir = root_dir / "metrics"
    output_file = metrics_dir / "Distancias_modelo.xlsx"

    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not resultados_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta esperada: {resultados_dir}"
        )

    patron_semana = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    filas = []

    # Recorre solo carpetas con formato semana, por ejemplo 2022-01-03
    semanas = sorted(
        [p for p in resultados_dir.iterdir() if p.is_dir() and patron_semana.match(p.name)]
    )

    if not semanas:
        print(f"No se encontraron carpetas de semana en: {resultados_dir}")
        return

    for semana_dir in semanas:
        semana = semana_dir.name
        archivo = semana_dir / f"Distancias_Modelo_{semana}_0.xlsx"

        if not archivo.exists():
            print(f"[AVISO] No existe el archivo para la semana {semana}: {archivo.name}")
            continue

        try:
            df = pd.read_excel(archivo, sheet_name="Resumen Semanal")

            if df.empty:
                print(f"[AVISO] La hoja 'Resumen Semanal' está vacía en {archivo}")
                continue

            # Normalmente debería haber una sola fila resumen, pero por si acaso
            for _, row in df.iterrows():
                fila = row.to_dict()

                # Normalizar columnas numéricas esperadas
                for col in [
                    "Distancia Total",
                    "Distancia LOAD",
                    "Distancia DLVR",
                    "Movimientos_DLVR",
                    "Movimientos_LOAD",
                ]:
                    if col in fila:
                        fila[col] = normalizar_numero(fila[col])

                filas.append(fila)

            print(f"[OK] Semana leída: {semana}")

        except Exception as e:
            print(f"[ERROR] Falló la lectura de {archivo}: {e}")

    if not filas:
        print("No se pudo leer ninguna semana. No se generó archivo de salida.")
        return

    df_final = pd.DataFrame(filas)

    # Ordena por semana si existe la columna
    if "Semana" in df_final.columns:
        df_final["Semana"] = pd.to_datetime(df_final["Semana"], errors="coerce")
        df_final = df_final.sort_values("Semana")
        df_final["Semana"] = df_final["Semana"].dt.strftime("%Y-%m-%d")

    # Guardar Excel final
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="Distancias", index=False)

    print(f"\nArchivo generado correctamente en:\n{output_file}")


if __name__ == "__main__":
    main()