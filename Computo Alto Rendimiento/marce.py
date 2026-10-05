import time
from pathlib import Path
import numpy as np
import multiprocessing as mp
from ovito.io import import_file

ruta_archivo = Path(__file__).resolve().parent / "Atoms.data"

DR = np.float32(0.3)

def cargar_datos(posiciones_iniciales, origen_inicial, largo_inicial):
    global _posiciones, _origen, _largo
    _posiciones = posiciones_iniciales
    _origen = origen_inicial
    _largo = largo_inicial

def construir_celdas(posiciones, origen, tamano_celda, celdas_por_eje):
    n = len(posiciones)
    celdas_x, celdas_y, celdas_z = celdas_por_eje

    celda_de_cada_atomo = np.floor((posiciones - origen) / tamano_celda).astype(np.int32)
    celda_de_cada_atomo[:, 0] = np.clip(celda_de_cada_atomo[:, 0], 0, celdas_x - 1)
    celda_de_cada_atomo[:, 1] = np.clip(celda_de_cada_atomo[:, 1], 0, celdas_y - 1)
    celda_de_cada_atomo[:, 2] = np.clip(celda_de_cada_atomo[:, 2], 0, celdas_z - 1)

    contador = np.zeros((celdas_x, celdas_y, celdas_z), dtype=np.int32)
    np.add.at(contador, (celda_de_cada_atomo[:, 0], celda_de_cada_atomo[:, 1], celda_de_cada_atomo[:, 2]), 1)

    capacidad = int(contador.max())
    ids_indices = -np.ones((celdas_x, celdas_y, celdas_z, capacidad), dtype=np.int32)
    ocupados_actual = np.zeros((celdas_x, celdas_y, celdas_z), dtype=np.int32)

    for i in range(n):
        cx, cy, cz = celda_de_cada_atomo[i]
        k = ocupados_actual[cx, cy, cz]
        ids_indices[cx, cy, cz, k] = i
        ocupados_actual[cx, cy, cz] += 1

    return ids_indices, celda_de_cada_atomo

# la adaptacion con rc arbitrarios
def calcular_rdf_para_radio(radio_corte):
    tiempo_inicio_tarea = time.perf_counter()

    posiciones = _posiciones
    origen = _origen
    largo = _largo
    n = len(posiciones)

    celdas_por_eje = tuple(max(int(np.floor(largo[d] / radio_corte)), 1) for d in range(3))
    tamano_celda = largo / np.array(celdas_por_eje, dtype=np.float32)
    celdas_x, celdas_y, celdas_z = celdas_por_eje

    ids_indices, celda_de_cada_atomo = construir_celdas(posiciones, origen, tamano_celda, celdas_por_eje)
    capacidad = ids_indices.shape[-1]

    nbins = max(int(np.ceil(radio_corte / DR)), 1)
    histograma = np.zeros(nbins, dtype=np.int64)
    radio_corte_cuadrado = np.float32(radio_corte) ** 2
    desplazamientos = np.array([-1, 0, 1])

    elementos_por_lote_max = 2_000_000
    tamano_lote = max(1, elementos_por_lote_max // (27 * capacidad))

    for inicio in range(0, n, tamano_lote):
        fin = min(inicio + tamano_lote, n)
        indices_lote = np.arange(inicio, fin)
        b = len(indices_lote)

        cx = celda_de_cada_atomo[indices_lote, 0]
        cy = celda_de_cada_atomo[indices_lote, 1]
        cz = celda_de_cada_atomo[indices_lote, 2]

        nx = (cx[:, None] + desplazamientos[None, :]) % celdas_x
        ny = (cy[:, None] + desplazamientos[None, :]) % celdas_y
        nz = (cz[:, None] + desplazamientos[None, :]) % celdas_z

        nx27 = np.repeat(nx, 9, axis=1)
        ny27 = np.tile(np.repeat(ny, 3, axis=1), (1, 3))
        nz27 = np.tile(nz, (1, 9))

        candidatos = ids_indices[nx27, ny27, nz27, :]

        posiciones_atomos = posiciones[indices_lote][:, None, None, :]
        posiciones_candidatos = posiciones[candidatos]

        delta = posiciones_atomos - posiciones_candidatos
        delta = delta - largo * np.round(delta / largo)
        dist_cuadrado = np.sum(delta * delta, axis=-1)

        valido = (candidatos >= 0) & (candidatos != indices_lote[:, None, None]) & (dist_cuadrado < radio_corte_cuadrado)
        distancias_validas = np.sqrt(dist_cuadrado[valido])

        if distancias_validas.size:
            indices_bin = (distancias_validas / DR).astype(np.int64)
            indices_bin = indices_bin[indices_bin < nbins]
            histograma += np.bincount(indices_bin, minlength=nbins)

    volumen_caja = largo[0] * largo[1] * largo[2]
    densidad_media = n / volumen_caja

    g_r = np.zeros(nbins)
    r_centros = np.zeros(nbins)
    for b in range(nbins):
        r_interior = b * DR
        r_exterior = (b + 1) * DR
        volumen_cascaron = (4.0 / 3.0) * np.pi * (r_exterior**3 - r_interior**3)
        g_r[b] = histograma[b] / (volumen_cascaron * n * densidad_media)
        r_centros[b] = (r_interior + r_exterior) / 2.0

    tiempo_tarea = time.perf_counter() - tiempo_inicio_tarea
    return radio_corte, r_centros, g_r, tiempo_tarea


if __name__ == '__main__':
    archivo = import_file(ruta_archivo)
    fotograma = archivo.compute()
    posiciones_iniciales = fotograma.particles.positions.array.astype(np.float32)
    matriz_celda = fotograma.cell.matrix
    origen_inicial = matriz_celda[:, 3].astype(np.float32)
    largo_inicial = np.array([matriz_celda[0, 0], matriz_celda[1, 1], matriz_celda[2, 2]], dtype=np.float32)

    radios = np.linspace(3, 15, 10)[::-1]
    procesos = 4
    tiempo_inicio = time.perf_counter()

    pool = mp.Pool(processes=procesos, initializer=cargar_datos, initargs=(posiciones_iniciales, origen_inicial, largo_inicial))
    resultados = pool.map(calcular_rdf_para_radio, radios)
    pool.close()
    pool.join()
    tiempo_total = time.perf_counter() - tiempo_inicio

    print("Rc", "tiempo_tarea(s)", "g_primeros_vecinos", "distancia_primeros_vecinos", sep="   ")
    for radio, r_centros, g_r, tiempo_tarea in sorted(resultados):
        indice_pico = int(np.argmax(g_r))
        print(round(radio, 4), round(tiempo_tarea, 4), round(g_r[indice_pico], 4), round(r_centros[indice_pico], 4), sep="    ")

    print("procesos:", procesos, "y tiempo total:", round(tiempo_total, 4), "s")