import os
import time
import numpy as np
import multiprocessing as mp
from ovito.io import import_file

radio_corte = 3.0
capacidad = 16
archivo_datos = '/home/fer/Documentos/Magister/Computo Alto Rendimiento/Atoms.data'


def fun_paralela(num_proceso, atomo_inicial, atomo_final,
                 memoria_contador, memoria_ids, cerrojo, memoria_tiempos,
                 forma_contador, forma_ids,
                 posiciones, identificadores, origen, tamano_celda):

    contador = np.frombuffer(memoria_contador, dtype=np.int32).reshape(forma_contador)
    ids_por_celda = np.frombuffer(memoria_ids, dtype=np.int32).reshape(forma_ids)

    tiempo_arranque = time.perf_counter()
    tiempo_restringido = 0.0

    for i in range(atomo_inicial, atomo_final):

        celda = list(np.floor((posiciones[i] - origen) / tamano_celda).astype(np.uint16))

        cerrojo.acquire()
        antes = time.perf_counter()
        ocupados = contador[tuple(celda)]
        contador[tuple(celda)] += 1
        celda.append(ocupados)
        ids_por_celda[tuple(celda)] = identificadores[i]
        tiempo_restringido += time.perf_counter() - antes
        cerrojo.release()

    tiempo_trabajado = time.perf_counter() - tiempo_arranque

    memoria_tiempos[2 * num_proceso] = tiempo_trabajado - tiempo_restringido
    memoria_tiempos[2 * num_proceso + 1] = tiempo_restringido


def main():
    fotograma = next(import_file(archivo_datos).frames)
    posiciones = fotograma.particles.positions.array
    identificadores = fotograma.particles.identifiers.array
    caja = fotograma.cell.matrix

    largo_caja = caja.diagonal()
    origen = caja[:, 3]
    num_celdas = np.floor(largo_caja / radio_corte).astype(np.uint16)
    tamano_celda = largo_caja / num_celdas
    celdas_x, celdas_y, celdas_z = (int(v) for v in num_celdas)

    total_atomos = len(posiciones)
    procesos = int(os.environ.get('P_NUM', os.cpu_count() or 4))
    reparto = [(j * total_atomos // procesos, (j + 1) * total_atomos // procesos)
               for j in range(procesos)]

    forma_contador = (celdas_x, celdas_y, celdas_z)
    forma_ids = (celdas_x, celdas_y, celdas_z, capacidad)
    total_celdas = celdas_x * celdas_y * celdas_z

    memoria_contador = mp.Array('i', total_celdas, lock=False)
    memoria_ids = mp.Array('i', total_celdas * capacidad, lock=False)
    memoria_tiempos = mp.Array('d', 2 * procesos, lock=False)

    cerrojo = mp.Lock()

    print("Atomos: {}  |  Subceldas: {}x{}x{} = {}  |  Procesos: {}".format(
        total_atomos, celdas_x, celdas_y, celdas_z, total_celdas, procesos))
    for num_proceso, (inicio, fin) in enumerate(reparto):
        print("   P{}: atomos [{:>6d}, {:>6d})  ->  {:>6d} atomos".format(
            num_proceso, inicio, fin, fin - inicio))

    tiempo_inicio = time.perf_counter()

    lista_procesos = [
        mp.Process(target=fun_paralela,
                   args=(num_proceso, inicio, fin,
                         memoria_contador, memoria_ids, cerrojo, memoria_tiempos,
                         forma_contador, forma_ids,
                         posiciones, identificadores, origen, tamano_celda))
        for num_proceso, (inicio, fin) in enumerate(reparto)]

    for proceso in lista_procesos:
        proceso.start()
    for proceso in lista_procesos:
        proceso.join()

    tiempo_total = time.perf_counter() - tiempo_inicio

    print("\nTiempo total (paralelo): {:.4f} s".format(tiempo_total))
    print("\n{:<6} {:>14} {:>18} {:>12}".format(
        "proc", "zona libre", "zona restringida", "% restring."))

    for num_proceso in range(procesos):
        tiempo_libre = memoria_tiempos[2 * num_proceso]
        tiempo_restringido = memoria_tiempos[2 * num_proceso + 1]
        print("{:<6} {:>13.4f}s {:>17.4f}s {:>11.1f}%".format(
            "P" + str(num_proceso), tiempo_libre, tiempo_restringido,
            100 * tiempo_restringido / (tiempo_libre + tiempo_restringido)))

    total_libre = sum(memoria_tiempos[2 * j] for j in range(procesos))
    total_restringido = sum(memoria_tiempos[2 * j + 1] for j in range(procesos))
    print("{:<6} {:>13.4f}s {:>17.4f}s {:>11.1f}%".format(
        "TOTAL", total_libre, total_restringido,
        100 * total_restringido / (total_libre + total_restringido)))

    contador = np.frombuffer(memoria_contador, dtype=np.int32).reshape(forma_contador)
    ids_por_celda = np.frombuffer(memoria_ids, dtype=np.int32).reshape(forma_ids)

    atomos_guardados = int(contador.sum())
    print("\nVerificacion: {} atomos clasificados de {}  ->  {}".format(
        atomos_guardados, total_atomos,
        "OK" if atomos_guardados == total_atomos else "FALTAN ATOMOS"))
    print("Maximo de atomos en una subcelda: {} (capacidad {})".format(
        int(contador.max()), capacidad))

    np.save('subcell_ids.npy', ids_por_celda.astype(np.uint32))
    print("Resultado guardado en subcell_ids.npy")


if __name__ == '__main__':
    main()

