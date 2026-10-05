import os
import time
import numpy as np
import multiprocessing as mp
from ovito.io import import_file

# --- CONFIGURACIÓN Y PARÁMETROS GENERALES ---
Rc = 3.0
capacity = 16
data_file_path = '/home/fer/Documentos/Magister/Computo Alto Rendimiento/Atoms.data'

# Selección de número de procesos (Descomenta la opción que desees probar):
num_processes = 4
# num_processes = 2
# num_processes = 6
# num_processes = 8
# num_processes = 10


def fun_paralela(proc_id, start_atom, end_atom,
                 count_shared, ids_shared, mem_lock, file_lock,
                 time_shared, count_shape, ids_shape,
                 positions, ids, O, d):
    """
    Función que ejecuta cada proceso hijo para clasificar un bloque de átomos.
    """
    subcell_count = np.frombuffer(count_shared, dtype=np.int32).reshape(count_shape)
    subcell_atomids = np.frombuffer(ids_shared, dtype=np.int32).reshape(ids_shape)

    t_start_proc = time.perf_counter()
    restricted_time = 0.0

    for i in range(start_atom, end_atom):
        # --- ZONA SEGURA (Cálculo independiente) ---
        n = list(np.floor((positions[i] - O) / d).astype(np.uint16))

        # --- ZONA CRÍTICA (Acceso sincronizado a memoria compartida) ---
        mem_lock.acquire()
        t_crit_start = time.perf_counter()

        count = subcell_count[tuple(n)]
        subcell_count[tuple(n)] += 1
        n.append(count)
        subcell_atomids[tuple(n)] = ids[i]

        restricted_time += time.perf_counter() - t_crit_start
        mem_lock.release()

    total_proc_time = time.perf_counter() - t_start_proc
    free_time = total_proc_time - restricted_time

    # Almacenar tiempos individuales
    time_shared[2 * proc_id] = free_time
    time_shared[2 * proc_id + 1] = restricted_time

    # --- ESCRITURA EN DISCO RESTRINGIDA (Log de ejecución) ---
    file_lock.acquire()
    with open("log_execution.txt", "a") as f:
        f.write(f"Proceso P{proc_id}: procesó {end_atom - start_atom} átomos.\n")
    file_lock.release()


def main():
    # 1. Carga de datos de la simulación
    frame = next(import_file(data_file_path).frames)
    positions = frame.particles.positions.array
    ids = frame.particles.identifiers.array
    cell = frame.cell.matrix

    # --- FLEXIBILIDAD EN TAMAÑO DE SUMAS / DATOS ---
    # Permite recortar o procesar la totalidad de los átomos para probar distintos tiempos
    # SAMPLE_SIZE = 100000  # Ejemplo: para pruebas rápidas
    SAMPLE_SIZE = len(positions) # Procesa todos los átomos disponibles
    positions = positions[:SAMPLE_SIZE]
    ids = ids[:SAMPLE_SIZE]

    # 2. Geometría de la caja y subceldas
    L = cell.diagonal()
    O = cell[:, 3]
    nmax = np.floor(L / Rc).astype(np.uint16)
    d = L / nmax

    nx, ny, nz = (int(v) for v in nmax)
    total_atoms = len(positions)

    # Reparto equitativo de carga por rangos
    atom_splits = [
        (j * total_atoms // num_processes, (j + 1) * total_atoms // num_processes)
        for j in range(num_processes)
    ]

    count_shape = (nx, ny, nz)
    ids_shape = (nx, ny, nz, capacity)
    total_cells = nx * ny * nz

    # 3. Memoria compartida y cerrojos
    count_shared = mp.Array('i', total_cells, lock=False)
    ids_shared = mp.Array('i', total_cells * capacity, lock=False)
    time_shared = mp.Array('d', 2 * num_processes, lock=False)

    mem_lock = mp.Lock()   # Cerrojo para arrays compartidos
    file_lock = mp.Lock()  # Cerrojo para archivos

    open("log_execution.txt", "w").close()

    print("=" * 65)
    print(f"Átomos procesados : {total_atoms}")
    print(f"Dimensiones malla : {nx}x{ny}x{nz} ({total_cells} subceldas)")
    print(f"Número de procesos: {num_processes}")
    print("=" * 65)

    # 4. Lanzamiento de procesos
    process_list = [
        mp.Process(
            target=fun_paralela,
            args=(proc_id, start_atom, end_atom,
                  count_shared, ids_shared, mem_lock, file_lock,
                  time_shared, count_shape, ids_shape,
                  positions, ids, O, d)
        )
        for proc_id, (start_atom, end_atom) in enumerate(atom_splits)
    ]

    t_start = time.perf_counter()

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

    t_total = time.perf_counter() - t_start

    # 5. Salida en consola y reporte de métricas
    print(f"\nTiempo total (paralelo): {t_total:.4f} s\n")
    print("{:<6} {:>14} {:>18} {:>12}".format("proc", "zona libre", "zona restringida", "% restring."))
    print("-" * 55)

    for proc_id in range(num_processes):
        t_free = time_shared[2 * proc_id]
        t_restricted = time_shared[2 * proc_id + 1]
        pct = 100 * t_restricted / (t_free + t_restricted) if (t_free + t_restricted) > 0 else 0
        print(f"P{proc_id:<5} {t_free:>13.4f}s {t_restricted:>17.4f}s {pct:>11.1f}%")

    total_free = sum(time_shared[2 * j] for j in range(num_processes))
    total_restricted = sum(time_shared[2 * j + 1] for j in range(num_processes))
    total_pct = 100 * total_restricted / (total_free + total_restricted) if (total_free + total_restricted) > 0 else 0
    print("-" * 55)
    print(f"TOTAL  {total_free:>13.4f}s {total_restricted:>17.4f}s {total_pct:>11.1f}%\n")

    # 6. Verificación de integridad
    subcell_count = np.frombuffer(count_shared, dtype=np.int32).reshape(count_shape)
    subcell_atomids = np.frombuffer(ids_shared, dtype=np.int32).reshape(ids_shape)

    saved_atoms = int(subcell_count.sum())
    status = "OK" if saved_atoms == total_atoms else "FALTAN ÁTOMOS"
    print(f"Verificación: {saved_atoms} / {total_atoms} átomos clasificados -> {status}")

    # 7. EXPORTACIÓN DE RESULTADOS (.NPY)
    file_lock.acquire()
    np.save('subcell_ids.npy', subcell_atomids.astype(np.uint32))
    file_lock.release()
    
    print("Resultado guardado en 'subcell_ids.npy'\n")


if __name__ == '__main__':
    main()