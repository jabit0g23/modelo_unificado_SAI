import os

def listar_y_guardar_contenido(directorio_raiz, carpetas_excluidas=None, archivo_salida="estructura_proyecto.txt"):
    if carpetas_excluidas is None:
        carpetas_excluidas = []

    with open(archivo_salida, "w", encoding="utf-8") as salida:
        for carpeta_actual, subcarpetas, archivos in os.walk(directorio_raiz):
            ruta_relativa = os.path.relpath(carpeta_actual, directorio_raiz)
            if any(excluida in ruta_relativa.split(os.sep) for excluida in carpetas_excluidas):
                continue

            for archivo in archivos:
                ruta_archivo = os.path.join(carpeta_actual, archivo)
                ruta_relativa_archivo = os.path.relpath(ruta_archivo, directorio_raiz)

                salida.write(f"ARCHIVO: {ruta_relativa_archivo}\n")

                try:
                    with open(ruta_archivo, "r", encoding="utf-8") as f:
                        contenido = f.read()
                        salida.write(contenido)
                except Exception as e:
                    salida.write(f"[Error al leer archivo: {e}]\n")


# ==== USO ====
if __name__ == "__main__":
    ruta_proyecto = "."  # o cambia por la ruta completa
    carpetas_excluir = ["venv", "__pycache__", ".git", "archivos_estaticos", "resultados_generados", "otros", "leer.py", "modelo_gruas_minmax.py"]

    listar_y_guardar_contenido(ruta_proyecto, carpetas_excluir, archivo_salida="estructura_proyecto.txt")
