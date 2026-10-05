import os
import time
import numpy as np
import multiprocessing as mp

# --- PARÁMETROS GENERALES ---
POSITIONS_FILE_PATH = '/home/fer/Documentos/Magister/Computo Alto Rendimiento/Atoms.data'
# 10 radios de corte entre 3 y 15 (actividad clase 4). Se invierte la lista
# para que el Pool reparta primero las tareas más largas (balanceo de carga).
RADIOS = np.linspace(3.0, 15.0, 10)[::-1]
DR = 0.02              # Ancho de cada bin del histograma
PERIODICO = True       # Bordes periódicos: celdas vecinas y distancias con imagen mínima
PROCESS_LIST = [1, 4]  # 1 proceso = referencia serial; 4 procesos = lo que pide la actividad
BLOCK_PAIRS = 20000    # Pares átomo-vecino por bloque (arreglos chicos que caben en caché)


# =====================================================================
# CARGA DE DATOS
# =====================================================================
def load_simulation_data(path):
    """Devuelve posiciones (N,3), origen (3) y largo de la caja (3)."""
    try:
        from ovito.io import import_file
        frame = next(import_file(path).frames)
        positions = np.ascontiguousarray(frame.particles.positions.array, dtype=np.float64)
        cell = frame.cell.matrix
        return positions, cell[:, 3].copy(), cell.diagonal().copy()
    except ImportError:
        # Lectura directa del archivo LAMMPS si ovito no está instalado
        lo, hi = np.zeros(3), np.zeros(3)
        with open(path) as f:
            for nline, line in enumerate(f):
                parts = line.split()
                for k, eje in enumerate(('xlo', 'ylo', 'zlo')):
                    if eje in parts:
                        lo[k], hi[k] = float(parts[0]), float(parts[1])
                if line.startswith('Atoms'):
                    skip = nline + 2
                    break
        data = np.loadtxt(path, skiprows=skip, usecols=(2, 3, 4))
        return np.ascontiguousarray(data), lo, hi - lo


# =====================================================================
# SUBDIVISIÓN EN SUBCELDAS (vectorizada, sin capacidad fija)
# =====================================================================
def build_subcells(positions, origin, L, rc):
    """
    Igual que AlgoritmoSubdivision.py, pero el tamaño de subcelda depende del
    radio de corte, y en vez de una matriz (nx,ny,nz,16) los átomos se ordenan
    por subcelda: los átomos de la subcelda c son sorted_pos[cell_start[c]:cell_start[c+1]].
    """
    nmax = np.maximum(np.floor(L / rc).astype(np.int64), 1)
    d = L / nmax
    n = np.floor((positions - origin) / d).astype(np.int64)
    n = np.clip(n, 0, nmax - 1)
    flat = (n[:, 0] * nmax[1] + n[:, 1]) * nmax[2] + n[:, 2]

    order = np.argsort(flat, kind='stable')
    sorted_pos = positions[order]
    counts = np.bincount(flat, minlength=int(np.prod(nmax)))
    cell_start = np.concatenate(([0], np.cumsum(counts)))
    return sorted_pos, cell_start, tuple(int(v) for v in nmax)


# =====================================================================
# TRABAJADOR DEL POOL
# =====================================================================
def init_worker(positions, origin, L):
    # Cada proceso del Pool recibe los datos una sola vez al iniciar
    global _positions, _origin, _L
    _positions, _origin, _L = positions, origin, L


def rdf_for_radius(rc):
    """
    Una tarea = un radio de corte. Arma las subceldas de tamaño rc, recorre
    cada subcelda contra sus 27 vecinas (con bordes periódicos) y devuelve g(r).
    """
    t_start = time.perf_counter()
    sorted_pos, cell_start, nmax = build_subcells(_positions, _origin, _L, rc)
    nx, ny, nz = nmax
    if PERIODICO and min(nmax) < 3:
        raise ValueError("Con bordes periódicos se necesitan al menos 3 subceldas por eje")

    nbins = int(round(rc / DR))
    hist = np.zeros(nbins, dtype=np.int64)
    rc_squared = rc * rc

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                c = (ix * ny + iy) * nz + iz
                pos_i = sorted_pos[cell_start[c]:cell_start[c + 1]]
                if len(pos_i) == 0:
                    continue

                # Reúne los átomos de las 27 subceldas vecinas. Si la vecina
                # está al otro lado de la caja (borde periódico), se guarda el
                # traslado +-L de esa celda: así no hace falta calcular la
                # imagen mínima para cada distancia.
                vecinos, traslados, cuantos = [], [], []
                for dx in (-1, 0, 1):
                    vx = ix + dx
                    if not PERIODICO and (vx < 0 or vx >= nx): continue
                    sx = _L[0] if vx >= nx else (-_L[0] if vx < 0 else 0.0)
                    for dy in (-1, 0, 1):
                        vy = iy + dy
                        if not PERIODICO and (vy < 0 or vy >= ny): continue
                        sy = _L[1] if vy >= ny else (-_L[1] if vy < 0 else 0.0)
                        for dz in (-1, 0, 1):
                            vz = iz + dz
                            if not PERIODICO and (vz < 0 or vz >= nz): continue
                            sz = _L[2] if vz >= nz else (-_L[2] if vz < 0 else 0.0)
                            v = ((vx % nx) * ny + (vy % ny)) * nz + (vz % nz)
                            atoms = sorted_pos[cell_start[v]:cell_start[v + 1]]
                            vecinos.append(atoms)
                            traslados.append((sx, sy, sz))
                            cuantos.append(len(atoms))
                pos_j = np.concatenate(vecinos)
                shift = np.repeat(np.array(traslados), cuantos, axis=0)
                xj, yj, zj = pos_j[:, 0], pos_j[:, 1], pos_j[:, 2]
                sx_j, sy_j, sz_j = shift[:, 0], shift[:, 1], shift[:, 2]

                # Distancias por bloques chicos de átomos, para que los arreglos
                # temporales quepan en la caché y los 4 procesos no compitan por
                # el ancho de banda de la memoria RAM.
                block = max(1, BLOCK_PAIRS // len(pos_j))
                for b in range(0, len(pos_i), block):
                    p = pos_i[b:b + block]
                    d = p[:, 0, None] - xj
                    d -= sx_j
                    dist_sq = d * d
                    d = p[:, 1, None] - yj
                    d -= sy_j
                    dist_sq += d * d
                    d = p[:, 2, None] - zj
                    d -= sz_j
                    dist_sq += d * d

                    # Se descarta el propio átomo (distancia 0) y lo que pasa del radio de corte
                    mask = (dist_sq > 0.0) & (dist_sq < rc_squared)
                    bins = (np.sqrt(dist_sq[mask]) / DR).astype(np.int64)
                    hist += np.bincount(bins, minlength=nbins)[:nbins]

    r, g = normalize_rdf(hist, len(_positions), _L)
    t_task = time.perf_counter() - t_start
    return rc, r, g, t_task, os.getpid()


def normalize_rdf(hist, n_atoms, L):
    # g(r) = H(r) / (N * (N/V) * volumen del cascarón)
    rho = n_atoms / np.prod(L)
    edges = np.arange(len(hist) + 1) * DR
    shell_vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    r = 0.5 * (edges[1:] + edges[:-1])
    g = hist / (n_atoms * rho * shell_vol)
    return r, g


# =====================================================================
# EJECUCIÓN CON POOL
# =====================================================================
def run_pool(num_proc, positions, origin, L):
    # No se asigna a priori qué proceso calcula qué radio: los radios van a la
    # cola del Pool y cada proceso libre toma el siguiente (imap_unordered).
    t0 = time.perf_counter()
    with mp.Pool(processes=num_proc, initializer=init_worker,
                 initargs=(positions, origin, L)) as pool:
        results = list(pool.imap_unordered(rdf_for_radius, RADIOS, chunksize=1))
    t_total = time.perf_counter() - t0
    return t_total, sorted(results, key=lambda res: res[0])


# Muestreo en pantalla
def display_results(results, num_proc, t_total):
    pids = sorted({res[4] for res in results})
    print("\n" + "=" * 78)
    print(f"        FUNCIÓN DE DISTRIBUCIÓN RADIAL ({num_proc} procesos)")
    print("=" * 78)
    print(f"{'Rc':<8} {'Tiempo tarea (s)':<18} {'g primer pico':<15} {'r primer pico':<15} {'Proceso':<8}")
    print("-" * 78)
    for rc, r, g, t_task, pid in results:
        k = int(np.argmax(g))
        print(f"{rc:<8.2f} {t_task:<18.4f} {g[k]:<15.4f} {r[k]:<15.4f} P{pids.index(pid)}")
    print("-" * 78)
    print(f"Procesos: {num_proc} | Tiempo total: {t_total:.4f} s")


def display_performance_summary(results_table):
    print("\n" + "=" * 60)
    print("             RESUMEN DE EFICIENCIA DEL POOL")
    print("=" * 60)
    print(f"{'Procesos':<10} {'Tiempo (s)':<14} {'Speedup':<12} {'Eficiencia':<12}")
    print("-" * 60)
    t_base = results_table[0][1] * results_table[0][0]
    for num_proc, t in results_table:
        speedup = t_base / t
        print(f"{num_proc:<10} {t:<14.4f} {speedup:<12.2f} {100 * speedup / num_proc:.1f}%")
    print("=" * 60)


# main
def main():
    path = POSITIONS_FILE_PATH if os.path.exists(POSITIONS_FILE_PATH) else 'Atoms.data'
    t0 = time.perf_counter()
    positions, origin, L = load_simulation_data(path)
    print(f"Lectura de datos: {time.perf_counter() - t0:.2f} s | Total Átomos: {len(positions):,}")
    print(f"Radios de corte (orden de envío al Pool): {np.round(RADIOS, 2)}")

    results_table = []
    results_ref = None
    for num_proc in PROCESS_LIST:
        t_total, results = run_pool(num_proc, positions, origin, L)
        display_results(results, num_proc, t_total)
        results_table.append((num_proc, t_total))
        if results_ref is None:
            results_ref = results
        elif any(not np.array_equal(a[2], b[2]) for a, b in zip(results, results_ref)):
            print("ADVERTENCIA: g(r) no coincide con la ejecución serial")

    display_performance_summary(results_table)


if __name__ == '__main__':
    main()
