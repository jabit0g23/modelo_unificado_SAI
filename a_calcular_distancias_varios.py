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

    s = str(x).strip().replace(",", "")

    try:
        val = float(s)
        return int(val) if val.is_integer() else val
    except ValueError:
        return x


def leer_resumen_semanal(archivo):
    """
    Lee la hoja 'Resumen Semanal' y devuelve la primera fila como dict.
    """
    df = pd.read_excel(archivo, sheet_name="Resumen Semanal")

    if df.empty:
        return None

    fila = df.iloc[0].to_dict()

    for col in [
        "Distancia Total",
        "Distancia LOAD",
        "Distancia DLVR",
        "Movimientos_DLVR",
        "Movimientos_LOAD",
    ]:
        if col in fila:
            fila[col] = normalizar_numero(fila[col])

    return fila


def leer_cantidad_segregaciones_unicas(archivo):
    """
    Busca una hoja que tenga la columna 'Segregacion'
    y devuelve la cantidad de valores únicos.
    """
    posibles_hojas = [
        "Resultados por Segregación",
        "Resultados por Segregacion",
        "Resultado por Segregación",
        "Resultado por Segregacion",
    ]

    xls = pd.ExcelFile(archivo)

    for hoja in posibles_hojas:
        if hoja in xls.sheet_names:
            df = pd.read_excel(archivo, sheet_name=hoja)
            if "Segregacion" in df.columns:
                return df["Segregacion"].dropna().nunique()

    for hoja in xls.sheet_names:
        df = pd.read_excel(archivo, sheet_name=hoja)
        if "Segregacion" in df.columns:
            return df["Segregacion"].dropna().nunique()

    return None


def construir_dataframe_salida(data_dict, columnas_ordenadas):
    """
    Convierte un diccionario tipo:
    {
        "2022-01-03": {"Ki70 - Criterio 2": 123, ...},
        ...
    }
    en un DataFrame ordenado.
    """
    if not data_dict:
        return pd.DataFrame(columns=["semana"] + columnas_ordenadas)

    semanas = sorted(data_dict.keys())
    filas = []

    for semana in semanas:
        fila = {"semana": semana}
        for col in columnas_ordenadas:
            fila[col] = data_dict.get(semana, {}).get(col, None)
        filas.append(fila)

    return pd.DataFrame(filas)


def aplicar_formato_sin_separadores(writer, nombre_hoja):
    """
    Aplica formato numérico sin separador de miles ni decimales
    a todas las columnas desde la segunda en adelante.
    """
    ws = writer.book[nombre_hoja]

    for row in ws.iter_rows(min_row=2, min_col=2):
        for cell in row:
            if cell.value is not None:
                cell.number_format = "0"


def main():
    base_dir = Path(__file__).resolve().parent

    escenarios = [
        ("resultados_generados_pila_criterio_ii_ki70", "Ki70 - Criterio 2"),
        ("resultados_generados_pila_criterio_iii_ki70", "Ki70 - Criterio 3"),
        ("resultados_generados_pila_criterio_ii_ki140", "Ki140 - Criterio 2"),
        ("resultados_generados_pila_criterio_iii_ki140", "Ki140 - Criterio 3"),
        ("resultados_generados_pila_criterio_ii_ki280", "Ki280 - Criterio 2"),
        ("resultados_generados_pila_criterio_iii_ki280", "Ki280 - Criterio 3"),
    ]

    columnas_salida = [
        "Ki70 - Criterio 2",
        "Ki70 - Criterio 3",
        "Ki140 - Criterio 2",
        "Ki140 - Criterio 3",
        "Ki280 - Criterio 2",
        "Ki280 - Criterio 3",
    ]

    patron_semana = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    data_distancia = {}
    data_mov_dlvr = {}
    data_mov_load = {}
    data_segregaciones = {}

    output_dir = base_dir / "metrics_consolidados_ki"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Consolidado_metricas_ki.xlsx"

    for root_name, nombre_columna in escenarios:
        root_dir = base_dir / root_name
        resultados_dir = root_dir / "resultados_magdalena"

        if not resultados_dir.exists():
            print(f"[AVISO] No existe: {resultados_dir}")
            continue

        semanas = sorted(
            [p for p in resultados_dir.iterdir() if p.is_dir() and patron_semana.match(p.name)]
        )

        if not semanas:
            print(f"[AVISO] No se encontraron semanas en: {resultados_dir}")
            continue

        print(f"\nProcesando escenario: {root_name}")

        for semana_dir in semanas:
            semana = semana_dir.name
            archivo = semana_dir / f"Distancias_Modelo_{semana}_0.xlsx"

            if not archivo.exists():
                print(f"[AVISO] No existe archivo para {semana}: {archivo}")
                continue

            try:
                resumen = leer_resumen_semanal(archivo)

                if resumen is None:
                    print(f"[AVISO] Hoja 'Resumen Semanal' vacía en {archivo}")
                    continue

                distancia_total = resumen.get("Distancia Total")
                mov_dlvr = resumen.get("Movimientos_DLVR")
                mov_load = resumen.get("Movimientos_LOAD")
                seg_unicas = leer_cantidad_segregaciones_unicas(archivo)

                if semana not in data_distancia:
                    data_distancia[semana] = {}
                if semana not in data_mov_dlvr:
                    data_mov_dlvr[semana] = {}
                if semana not in data_mov_load:
                    data_mov_load[semana] = {}
                if semana not in data_segregaciones:
                    data_segregaciones[semana] = {}

                data_distancia[semana][nombre_columna] = distancia_total
                data_mov_dlvr[semana][nombre_columna] = mov_dlvr
                data_mov_load[semana][nombre_columna] = mov_load
                data_segregaciones[semana][nombre_columna] = seg_unicas

                print(f"[OK] {root_name} | {semana}")

            except Exception as e:
                print(f"[ERROR] Falló lectura en {archivo}: {e}")

    df_distancia = construir_dataframe_salida(data_distancia, columnas_salida)
    df_mov_dlvr = construir_dataframe_salida(data_mov_dlvr, columnas_salida)
    df_mov_load = construir_dataframe_salida(data_mov_load, columnas_salida)
    df_segregaciones = construir_dataframe_salida(data_segregaciones, columnas_salida)

    # Asegurar tipo numérico entero nullable
    for df in [df_distancia, df_mov_dlvr, df_mov_load, df_segregaciones]:
        for col in df.columns[1:]:
            serie = pd.to_numeric(df[col], errors="coerce")
            df[col] = serie.round(0).astype("Int64")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_distancia.to_excel(writer, sheet_name="Distancia", index=False)
        df_mov_dlvr.to_excel(writer, sheet_name="Movimientos_DLVR", index=False)
        df_mov_load.to_excel(writer, sheet_name="Movimientos_LOAD", index=False)
        df_segregaciones.to_excel(writer, sheet_name="Resultados por Segregación", index=False)

        aplicar_formato_sin_separadores(writer, "Distancia")
        aplicar_formato_sin_separadores(writer, "Movimientos_DLVR")
        aplicar_formato_sin_separadores(writer, "Movimientos_LOAD")
        aplicar_formato_sin_separadores(writer, "Resultados por Segregación")

    print(f"\nArchivo generado correctamente en:\n{output_file}")


if __name__ == "__main__":
    main()