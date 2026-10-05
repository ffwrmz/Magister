import os
import time
import numpy as np
import multiprocessing as mp
from ovito.io import import_file

NPY_FILE_PATH = 'subcell_ids.npy'
POSITIONS_FILE_PATH = '/home/fer/Documentos/Magister/Computo Alto Rendimiento/Atoms.data'
Rc = 3.0  # Radio de corte
PROCESS_LIST = [2, 4, 8]


#
def coordination_worker(proc_id, start_x, end_x, subcell_atomids, positions, id_map,
                        time_shared, communication_channel, is_decentralized):
    t_start = time.perf_counter()
    nx, ny, nz, capacity = subcell_atomids.shape
    rc_squared = Rc ** 2

    local_neighbors_sum = 0
    local_atoms_count = 0

    for ix in range(start_x, end_x):
        for iy in range(ny):
            for iz in range(nz):
                
                cell_atoms = subcell_atomids[ix, iy, iz]
                valid_atoms = cell_atoms[cell_atoms > 0]

                for i in valid_atoms:
                    idx_i = id_map[i]
                    if idx_i == -1: continue
                    
                    pos_i = positions[idx_i]
                    sphere_neighbors = 0

                    for dx in (-1, 0, 1):
                        vx = ix + dx
                        if vx < 0 or vx >= nx: continue
                        for dy in (-1, 0, 1):
                            vy = iy + dy
                            if vy < 0 or vy >= ny: continue
                            for dz in (-1, 0, 1):
                                vz = iz + dz
                                if vz < 0 or vz >= nz: continue

                                vec_atoms = subcell_atomids[vx, vy, vz]
                                for j in vec_atoms:
                                    if j <= 0 or j == i:
                                        continue
                                    
                                    idx_j = id_map[j]
                                    if idx_j == -1: continue

                                    pos_j = positions[idx_j]
                                    diff = pos_i - pos_j
                                    dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]

                                    if dist_sq <= rc_squared:
                                        sphere_neighbors += 1

                    local_neighbors_sum += sphere_neighbors
                    local_atoms_count += 1

    t_total = time.perf_counter() - t_start
    result = (proc_id, local_neighbors_sum, local_atoms_count)

    # Descentralizado (Pipe) vs Centralizado (Queue)
    if is_decentralized:
        communication_channel.send(result)
        communication_channel.close()
    else:
        communication_channel.put(result)

    time_shared[proc_id] = t_total


# Carga de archivos, tanto de subceldas creadas como de las posiciones en el archivo original
def load_simulation_data():
    subcell_atomids = np.load(NPY_FILE_PATH)
    
    frame = next(import_file(POSITIONS_FILE_PATH).frames)
    positions = frame.particles.positions.array
    atom_ids = frame.particles.identifiers.array

    # Mapea IDs reales a índices de la matriz
    max_id = int(np.max(atom_ids))
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    id_map[atom_ids] = np.arange(len(atom_ids), dtype=np.int32)

    return subcell_atomids, positions, id_map

# Centralizado utilizando Queue de multiprocessing.
def run_centralized_queue(num_proc, nx, subcell_atomids, positions, id_map, splits):
    time_shared = mp.Array('d', num_proc, lock=False)
    queue = mp.Queue()
    processes = [
        mp.Process(
            target=coordination_worker,
            args=(p_id, s_x, e_x, subcell_atomids, positions, id_map, time_shared, queue, False)
        )
        for p_id, (s_x, e_x) in enumerate(splits)
    ]

    t0 = time.perf_counter()
    for p in processes: p.start()

    total_sum, total_count = 0, 0
    for _ in range(num_proc):
        _, local_sum, local_count = queue.get()
        total_sum += local_sum
        total_count += local_count

    for p in processes: p.join()
    t_total = time.perf_counter() - t0
    average = total_sum / total_count if total_count > 0 else 0

    return t_total, average

# Descentralizado utilizando Pipes de multiprocessing.
def run_decentralized_pipe(num_proc, nx, subcell_atomids, positions, id_map, splits):
    time_shared = mp.Array('d', num_proc, lock=False)
    processes = []
    parent_connections = []

    for p_id, (s_x, e_x) in enumerate(splits):
        parent_conn, child_conn = mp.Pipe()
        parent_connections.append(parent_conn)
        p = mp.Process(
            target=coordination_worker,
            args=(p_id, s_x, e_x, subcell_atomids, positions, id_map, time_shared, child_conn, True)
        )
        processes.append(p)

    t0 = time.perf_counter()
    for p in processes: p.start()

    total_sum, total_count = 0, 0
    for conn in parent_connections:
        _, local_sum, local_count = conn.recv()
        total_sum += local_sum
        total_count += local_count

    for p in processes: p.join()
    t_total = time.perf_counter() - t0
    average = total_sum / total_count if total_count > 0 else 0

    return t_total, average

# Muestreo en pantalla
def display_performance_summary(results_table):
    print("\n" + "=" * 75)
    print("                 RESUMEN COMPARATIVO DE RENDIMIENTO")
    print("=" * 75)
    print(f"{'Procesos':<12} {'Centralizado':<22} {'Descentralizado':<24} {'Métricas':<20}")
    print("-" * 75)

    t_base_pipe = results_table[0][2]

    for num_proc, t_q, t_p in results_table:
        pipe_speedup = (t_base_pipe * 2) / t_p if num_proc == 2 else (results_table[0][2] * 2) / t_p
        pipe_efficiency = (pipe_speedup / num_proc) * 100
        
        print(f"{num_proc:<12} {t_q:<22.4f} {t_p:<24.4f} S: {pipe_speedup:.2f}x | Efic: {pipe_efficiency:.1f}%")

    print("=" * 75)


# main
def main():
    subcell_atomids, positions, id_map = load_simulation_data()
    if subcell_atomids is None:
        return

    nx, ny, nz, capacity = subcell_atomids.shape
    total_cells = nx * ny * nz

    print(f" Malla 3D: {nx}x{ny}x{nz} | Total Átomos: {len(positions):,}")

    results_table = []

    # Iteración automatizada sobre 2, 4 y 8 procesos
    for num_proc in PROCESS_LIST:
        print(f"\n[ {num_proc} procesos")
        
        chunk_x = nx // num_proc
        splits = [
            (i * chunk_x, nx if num_proc - 1 == i else (i + 1) * chunk_x)
            for i in range(num_proc)
        ]

        # Ejecutar Centralizado (Queue) y capturar su resultado promedio
        t_queue, avg_queue = run_centralized_queue(num_proc, nx, subcell_atomids, positions, id_map, splits)
        print(f"  -> Centralizado     : {t_queue:.4f} s (Promedio vecinos: {avg_queue:.4f})")

        # Ejecutar Descentralizado (Pipe) y capturar su resultado promedio
        t_pipe, avg_pipe = run_decentralized_pipe(num_proc, nx, subcell_atomids, positions, id_map, splits)
        print(f"  -> Descentralizado  : {t_pipe:.4f} s (Promedio vecinos: {avg_pipe:.4f})")

        results_table.append((num_proc, t_queue, t_pipe))

    # Mostrar la tabla de reporte final
    display_performance_summary(results_table)


if __name__ == '__main__':
    main()