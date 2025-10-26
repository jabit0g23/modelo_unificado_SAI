import re
import math
import os
import sys
from collections import defaultdict
import argparse
import pandas as pd
from pathlib import Path

# --- Constantes ---
IGNORAR_NOMBRES = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode"}
CAMILA_SUBFOLDER = "resultados_camila"
PREFIX_TO_REMOVE = "resultados_generados_"

# --- Función de Análisis IIS (MODIFICADA para 2 patrones y mejor salida) ---
def analyze_iis(filepath, prod_rs=20, prod_rtg=30, costanera_prefix='b'):
    """
    Analiza un archivo .ilp de IIS para 2 patrones de conflicto:
    1. Capacidad de grúas RS fuera de Costanera.
    2. Insuficiencia de inventario para carga sin recepción.
    Devuelve lista de resultados (dict) y el tipo de conflicto principal.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except Exception as e:
        # print(f"    ERROR: Al leer {filepath}: {e}") # Silenciado
        return None, None

    # --- 1. Leer y concatenar ---
    constraint_lines = {}
    current_constraint_name = None
    constraint_start_pattern = re.compile(r"^([\w\(\)\-\.]+)\s*:")
    for line_raw in lines:
        line = line_raw.strip();
        if not line or line.startswith('\\'): continue
        match_start = constraint_start_pattern.match(line)
        if match_start and not line.startswith(' '):
            current_constraint_name = match_start.group(1).strip()
            if current_constraint_name not in constraint_lines: constraint_lines[current_constraint_name] = line
            else: constraint_lines[current_constraint_name] += " " + (line.split(':', 1)[1].strip() if ':' in line else line)
        elif current_constraint_name and current_constraint_name in constraint_lines:
            constraint_lines[current_constraint_name] += " " + line

    # --- 2. Regex ---
    demand_re = re.compile(r"dem_(descarga|carga)\(s(\w+)_(\d+)\)")
    rtg_zero_re = re.compile(r"nRTG\((\w+)_(\d+)\)\s*=\s*0")
    max_block_name_re = re.compile(r"max_by_block\((\w+)_(\d+)\)")
    max_rtg_block_name_re = re.compile(r"max_rtg_block\((\w+)_(\d+)\)")
    diff_constr_name_re = re.compile(r"diff_constr\((\w+)_(\d+)\)")
    limit_re = re.compile(r"<=\s*(\d+)$")
    rhs_re = re.compile(r"=\s*(\d+(\.\d+)?)\s*$")
    inv_min_re = re.compile(r"inv_min\((\w+)_s(\w+)_(\d+)\)")
    dem_recibir_re = re.compile(r"dem_recibir\(s(\w+)\)")

    # --- 3. Parsear datos clave ---
    max_cranes_allowed = defaultdict(lambda: 3) # Default 3
    max_rtg_allowed = defaultdict(lambda: 2)    # Default 2
    forced_rtg_zero_points = set()
    demands_by_time_seg = defaultdict(lambda: defaultdict(float))
    zero_receiving_segs = set()
    inv_min_constraints_present = set()
    capacity_constraints_blocks_by_time = defaultdict(set) # (b,t) con constr. de capacidad

    for name, line in constraint_lines.items():
        # RTG Zero
        for match in rtg_zero_re.finditer(line):
            block, time_str = match.groups(); time = int(time_str)
            if not block.lower().startswith(costanera_prefix):
                forced_rtg_zero_points.add((block, time))
                capacity_constraints_blocks_by_time[time].add(block) # Es un punto de capacidad
        # Max Cranes
        m_max = max_block_name_re.match(name)
        if m_max:
            block, time_str = m_max.groups(); time = int(time_str)
            m_lim = limit_re.search(line)
            if m_lim: max_cranes_allowed[(block, time)] = int(m_lim.group(1))
            capacity_constraints_blocks_by_time[time].add(block)
        # Max RTG Cranes
        m_max_rtg = max_rtg_block_name_re.match(name)
        if m_max_rtg:
            block, time_str = m_max_rtg.groups(); time = int(time_str)
            m_lim = limit_re.search(line)
            if m_lim: max_rtg_allowed[(block, time)] = int(m_lim.group(1))
            capacity_constraints_blocks_by_time[time].add(block)
        # Diff Constr
        m_diff = diff_constr_name_re.match(name)
        if m_diff:
             block, time_str = m_diff.groups(); time = int(time_str)
             capacity_constraints_blocks_by_time[time].add(block)
        # Demand
        m_dem = demand_re.match(name)
        if m_dem:
            op, seg, time_str = m_dem.groups(); time = int(time_str)
            m_rhs = rhs_re.search(line)
            if m_rhs:
                demand = float(m_rhs.group(1)); key = f"{op}_s{seg}"
                demands_by_time_seg[time][key] = demand
        # Zero Receiving
        m_rec = dem_recibir_re.match(name)
        if m_rec:
            seg = m_rec.group(1); m_rhs = rhs_re.search(line)
            if m_rhs and float(m_rhs.group(1)) == 0: zero_receiving_segs.add(seg)
        # Inv Min
        m_inv = inv_min_re.match(name)
        if m_inv: b, s, t_str = m_inv.groups(); inv_min_constraints_present.add((b, s, int(t_str)))

    # --- 4. Análisis de Patrones y Resumen ---
    results = []
    processed_keys = set()

    # --- PATRÓN 1: Déficit Capacidad RS (No-Costanera) ---
    for block, time in sorted(list(forced_rtg_zero_points)):
        key = ("RS_Cap", block, time);
        if key in processed_keys: continue
        
        total_demand_at_time = sum(demands_by_time_seg.get(time, {}).values())
        if total_demand_at_time <= 0: continue # Solo si hay demanda en este tiempo

        max_allowed = max_cranes_allowed.get((block, time), 3)
        max_rs_allowed = max_allowed # nRTG = 0
        max_capacity_rs = max_rs_allowed * prod_rs
        deficit = max(0, total_demand_at_time - max_capacity_rs)

        # Solo reportar si hay déficit REAL
        if deficit > 0:
             needed_rs = math.ceil(total_demand_at_time / prod_rs) if prod_rs > 0 else float('inf')
             additional_rs = max(0, needed_rs - max_rs_allowed)
             results.append({
                "Conflict Type": "RS Capacity", "File Name": os.path.basename(filepath),
                "Block": block, "Time": time, "Segregation": "-",
                "Required Moves": total_demand_at_time,
                "Loading Demand": "-",
                "Demand Details": ", ".join([f"{k}: {v:.0f}" for k, v in sorted(demands_by_time_seg.get(time, {}).items())]),
                "Capacity Deficit (mov/hr)": deficit,
                "Additional RS Needed": additional_rs,
                "Key Constraints": "dem_*, max_by_block, rtg_solo_costanera"
            })
             processed_keys.add(key)

    # --- PATRÓN 2: Déficit Inventario (con posible Capacidad) ---
    for time, demands in demands_by_time_seg.items():
        for op_seg, demand_val in demands.items():
            if op_seg.startswith("carga_s"):
                seg_num = op_seg.split('s')[-1]
                if demand_val > 0 and seg_num in zero_receiving_segs:
                    key = ("Inventory", time, seg_num)
                    if key in processed_keys: continue
                    dem_carga_name = f"dem_carga(s{seg_num}_{time})"
                    dem_recibir_name = f"dem_recibir(s{seg_num})"
                    inv_min_related = any(s == seg_num for b, s, t in inv_min_constraints_present if t >= time)

                    if dem_carga_name in constraint_lines and dem_recibir_name in constraint_lines:
                        # Tratar de encontrar el bloque asociado
                        assoc_block = "-"
                        blocks_in_cap_constr = list(capacity_constraints_blocks_by_time.get(time, set()))
                        if len(blocks_in_cap_constr) == 1:
                            assoc_block = blocks_in_cap_constr[0]
                        
                        results.append({
                            "Conflict Type": "Inventory Shortage", "File Name": os.path.basename(filepath),
                            "Block": assoc_block, "Time": time, "Segregation": seg_num,
                            "Required Moves": "-", "Loading Demand": demand_val,
                            "Demand Details": f"{op_seg}: {demand_val:.0f}",
                            "Capacity Deficit (mov/hr)": "-",
                            "Additional RS Needed": "-",
                            "Key Constraints": "dem_carga, dem_recibir=0" + (", inv_min" if inv_min_related else "") + (f", (cap_{assoc_block})" if assoc_block != "-" else "")
                        })
                        processed_keys.add(key)


    if not results: return None, "No Conflict Pattern Found"
    conflict_types = {r.get("Conflict Type", "Unknown") for r in results}
    main_type = "Mixed" if len(conflict_types) > 1 else list(conflict_types)[0]
    return results, main_type

# --- (Función es_carpeta_valida sin cambios) ---
def es_carpeta_valida(p: Path) -> bool:
    return p.is_dir() and (p.name not in IGNORAR_NOMBRES) and (not p.name.startswith("."))

# --- Ejecución Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Analiza archivos IIS .ilp y genera resumen Excel.")
    parser.add_argument("directory", nargs='?', default=".", help="Directorio raíz (default: .)")
    parser.add_argument("-p", "--prod_rs", type=int, default=20, help="Productividad RS (mov/hr, default: 20)")
    parser.add_argument("--prod_rtg", type=int, default=30, help="Productividad RTG (mov/hr, default: 30)")
    parser.add_argument("-c", "--costanera_prefix", type=str, default="b", help="Prefijo Costanera (default: 'b')")
    parser.add_argument("-o", "--output", type=str, default="iis_summary_report.xlsx", help="Archivo Excel salida (default: iis_summary_report.xlsx)")

    args = parser.parse_args()
    root_directory = Path(args.directory).resolve(); productivity_rs = args.prod_rs
    productivity_rtg = args.prod_rtg; costanera_prefix = args.costanera_prefix.lower()
    output_filename = args.output; output_path = root_directory / output_filename
    if not root_directory.is_dir(): print(f"Error: Directorio raíz no existe: {root_directory}"); sys.exit(1)

    # --- 1. Encontrar Carpetas Principales ---
    main_folders = []; print(f"Buscando carpetas principales en: {root_directory}")
    for item in root_directory.iterdir():
        if es_carpeta_valida(item) and (item / CAMILA_SUBFOLDER).is_dir():
            main_folders.append(item); print(f"  -> Encontrada: {item.name}")
    if not main_folders: print(f"No se encontraron carpetas válidas con '{CAMILA_SUBFOLDER}' en {root_directory}"); sys.exit(0)

    # --- 2. Procesar y Escribir Excel ---
    print(f"\nIniciando análisis y escritura en: {output_path}")
    print(f"(Usando Prod RS = {productivity_rs} mov/hr, Prod RTG = {productivity_rtg} mov/hr, Prefijo Costanera = '{costanera_prefix}')")
    print("=========================================")

    overall_summary_data = []
    all_files_without_conflict_info = []

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            sheets_written = 0; processed_sheet_names = set()

            for main_folder_path in sorted(main_folders):
                main_folder_name_original = main_folder_path.name
                camila_path = main_folder_path / CAMILA_SUBFOLDER
                print(f"\n--- Procesando Carpeta: {main_folder_name_original} ---")

                sheet_results_data = []; all_iis_files_in_folder = list(camila_path.rglob('*.iis.ilp'))
                files_with_analyzed_conflict = set()

                for iis_filepath in all_iis_files_in_folder:
                    try: relative_path = iis_filepath.relative_to(camila_path); week_folder_name = relative_path.parts[0] if relative_path.parts else "raiz_camila"
                    except ValueError: week_folder_name = "desconocida"
                    print(f"  -> Analizando: {relative_path}")
                    analysis_results, conflict_type = analyze_iis(iis_filepath, prod_rs=productivity_rs, prod_rtg=productivity_rtg, costanera_prefix=costanera_prefix)

                    if analysis_results:
                        files_with_analyzed_conflict.add(iis_filepath.name)
                        print(f"      -> Conflicto: '{conflict_type}'. {len(analysis_results)} punto(s).")
                        for result_dict in analysis_results:
                             # ---> Rellenar con '-' <---
                             full_row_data = {
                                "Conflict Type": result_dict.get("Conflict Type", "-"),
                                "Week Folder": week_folder_name,
                                "IIS File Name": result_dict.get("File Name", iis_filepath.name),
                                "Block": result_dict.get("Block", "-"),
                                "Time": result_dict.get("Time", "-"),
                                "Segregation": result_dict.get("Segregation", "-"),
                                "Required Moves": result_dict.get("Required Moves", "-"),
                                "Loading Demand": result_dict.get("Loading Demand", "-"),
                                "Demand Details": result_dict.get("Demand Details", ""),
                                "Capacity Deficit (mov/hr)": result_dict.get("Capacity Deficit (mov/hr)", "-"),
                                "Additional RS Needed": result_dict.get("Additional RS Needed", "-"),
                                "Key Constraints": result_dict.get("Key Constraints", "")
                            }
                             sheet_results_data.append(full_row_data)

                # Identificar archivos SIN conflicto relevante
                for iis_filepath in all_iis_files_in_folder:
                     if iis_filepath.name not in files_with_analyzed_conflict:
                          try: relative_path = iis_filepath.relative_to(camila_path); week_folder_name = relative_path.parts[0] if relative_path.parts else "raiz_camila"
                          except ValueError: week_folder_name = "desconocida"
                          all_files_without_conflict_info.append({
                              "Carpeta Principal": main_folder_name_original,
                              "Week Folder": week_folder_name, "IIS File Name": iis_filepath.name
                          })

                # --- Nombre de Hoja ---
                sheet_name_base = main_folder_name_original
                if sheet_name_base.startswith(PREFIX_TO_REMOVE): sheet_name_base = sheet_name_base[len(PREFIX_TO_REMOVE):]
                safe_sheet_name = sheet_name_base[:31]; sheet_suffix = 1
                final_sheet_name = safe_sheet_name
                while final_sheet_name in processed_sheet_names: sheet_suffix += 1; base_len = 31 - len(f"_({sheet_suffix})"); final_sheet_name = f"{safe_sheet_name[:base_len]}_({sheet_suffix})"
                processed_sheet_names.add(final_sheet_name); print(f"  -> Nombre de hoja: '{final_sheet_name}'")

                # --- Escritura de Hoja Principal ---
                # ---> ORDEN DE COLUMNAS DEFINITIVO <---
                column_order = [
                    "Conflict Type", "Week Folder", "IIS File Name", "Block", "Time", "Segregation",
                    "Required Moves", "Loading Demand", "Demand Details",
                    "Capacity Deficit (mov/hr)", "Additional RS Needed",
                    "Key Constraints"
                ]
                if sheet_results_data:
                    df_sheet = pd.DataFrame(sheet_results_data)
                    # Asegurar que todas las columnas existan, rellenar con '-'
                    for col in column_order:
                        if col not in df_sheet.columns: df_sheet[col] = '-'
                    df_sheet.sort_values(by=['Week Folder', 'IIS File Name', 'Time', 'Block', 'Segregation'], inplace=True)
                    df_sheet = df_sheet[column_order] # Reordenar/Seleccionar
                    df_sheet.to_excel(writer, sheet_name=final_sheet_name, index=False)
                    sheets_written += 1; print(f"      -> Hoja generada con {len(df_sheet)} fila(s).")
                    # Recopilar para resumen
                    numeric_deficit = pd.to_numeric(df_sheet['Capacity Deficit (mov/hr)'], errors='coerce')
                    numeric_add_rs = pd.to_numeric(df_sheet['Additional RS Needed'], errors='coerce')
                    avg_deficit = numeric_deficit.mean() if numeric_deficit.notna().any() else 0
                    total_additional_rs = numeric_add_rs.sum() if numeric_add_rs.notna().any() else 0
                    num_critical_points = len(df_sheet)
                    num_iis_files_with_conflict = df_sheet['IIS File Name'].nunique()
                else: # Hoja placeholder
                    df_placeholder = pd.DataFrame([{"Mensaje": f"No se encontraron conflictos relevantes (Patrones definidos) en los {len(all_iis_files_in_folder)} archivo(s) .iis.ilp." if all_iis_files_in_folder else f"No se encontraron archivos .iis.ilp en '{CAMILA_SUBFOLDER}'."}])
                    # Añadir columnas vacías para consistencia si se desea, aunque no es estrictamente necesario
                    # for col in column_order:
                    #     if col not in df_placeholder.columns: df_placeholder[col] = ""
                    # df_placeholder = df_placeholder[column_order] # Ordenar
                    df_placeholder.to_excel(writer, sheet_name=final_sheet_name, index=False)
                    sheets_written += 1; print(f"      -> Hoja generada con mensaje informativo.")
                    avg_deficit = 0; total_additional_rs = 0; num_critical_points = 0; num_iis_files_with_conflict = 0

                # Añadir entrada al resumen general
                overall_summary_data.append({
                    "Carpeta Principal": main_folder_name_original, "Hoja Conflictos": final_sheet_name,
                    "IIS Procesados": len(all_iis_files_in_folder),
                    "IIS Conflicto Cap/Inv": num_iis_files_with_conflict,
                    "IIS Otros": len(all_iis_files_in_folder) - num_iis_files_with_conflict,
                    "Puntos Críticos": num_critical_points,
                    "Prom Deficit Cap (mov/hr)": avg_deficit,
                    "Total Add RS": total_additional_rs
                })


            # --- 3. Escribir Hoja "Otros IIS (Global)" ---
            other_iis_sheet_name = "Otros IIS (Global)"
            if all_files_without_conflict_info:
                 print(f"\n--- Generando Hoja '{other_iis_sheet_name}' ---")
                 df_all_others = pd.DataFrame(all_files_without_conflict_info)
                 df_all_others.sort_values(by=["Carpeta Principal", 'Week Folder', 'IIS File Name'], inplace=True)
                 df_all_others.to_excel(writer, sheet_name=other_iis_sheet_name, index=False)
                 sheets_written += 1; print(f"   -> Hoja '{other_iis_sheet_name}' generada con {len(df_all_others)} archivo(s).")
            else:
                 print(f"\n--- Generando Hoja '{other_iis_sheet_name}' ---")
                 df_nc_placeholder = pd.DataFrame([{"Mensaje": "Todos los .iis.ilp encontrados contenían un conflicto analizado." if any(f for f in main_folders if list((f / CAMILA_SUBFOLDER).rglob('*.iis.ilp'))) else "No se encontraron .iis.ilp."}])
                 df_nc_placeholder.to_excel(writer, sheet_name=other_iis_sheet_name, index=False)
                 sheets_written += 1; print(f"   -> Hoja '{other_iis_sheet_name}' generada con mensaje informativo.")


            # --- 4. Escribir Hoja de Resumen General ---
            if overall_summary_data:
                print("\n--- Generando Hoja de Resumen General ---")
                df_overall = pd.DataFrame(overall_summary_data)
                df_overall.sort_values(by="Carpeta Principal", inplace=True)
                df_overall.rename(columns={ "Prom Deficit Cap (mov/hr)": "Prom Deficit Cap (mov/hr)"}, inplace=True)
                summary_column_order = [
                    "Carpeta Principal", "Hoja Conflictos",
                    "IIS Procesados", "IIS Conflicto Cap/Inv", "IIS Otros",
                    "Puntos Críticos", "Prom Deficit Cap (mov/hr)", "Total Add RS"
                ]
                for col in summary_column_order:
                     if col not in df_overall.columns: df_overall[col] = 0
                df_overall = df_overall[summary_column_order]
                for col in ["Prom Deficit Cap (mov/hr)"]: df_overall[col] = df_overall[col].fillna(0).round(2)
                for col in ["Total Add RS", "IIS Procesados", "IIS Conflicto Cap/Inv", "IIS Otros", "Puntos Críticos"]:
                     df_overall[col] = pd.to_numeric(df_overall[col], errors='coerce').fillna(0).astype(int)
                df_overall.to_excel(writer, sheet_name="Resumen General", index=False)
                sheets_written += 1; print("   -> Hoja 'Resumen General' generada.")


            # --- Mensaje Final ---
            print("\n=========================================")
            if sheets_written > 0: print(f"\n✅ Resumen generado con {sheets_written} hoja(s) en: {output_path}")
            else: print("\nℹ️ No se generó ninguna hoja.")

    except ImportError: print("\n⚠️ Error: Librería 'openpyxl' necesaria (pip install openpyxl)")
    except PermissionError: print(f"\n❌ Error: Permiso denegado al escribir en '{output_path}'. ¿Archivo abierto?")
    except Exception as e: print(f"\n❌ Error inesperado al escribir Excel: {e}")