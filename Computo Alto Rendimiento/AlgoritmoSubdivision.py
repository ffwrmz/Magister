import numpy as np
from ovito.io import import_file    # Necesitan instalar la libreria de ovito con 'pip install ovito'
import time

frame = next(import_file('/home/fer/Documentos/Magister/Computo Alto Rendimiento/Atoms.data').frames)
# Coordenadas x,y,z de cada atom
positions = frame.particles.positions.array
# Identificador de cada atom
ids = frame.particles.identifiers.array
# Matriz de la caja simulada
cell = frame.cell.matrix
# Subdivision L Dimensiones, O origen Rc tamaño de corte
L = cell.diagonal()
O = cell[:, 3]
Rc = 3.0
# Calcula cuantas subceldas hay en cada direccion
nmax = np.floor(L / Rc).astype(np.uint16)
# Tamaño real de cada celda
d = L / nmax

# Crea estructuras de almacenamiento
subcell_count = np.zeros(nmax, dtype=np.uint8)
# Crea arreglo tridimencional, cada posicion almacena cuantos 
# atomos hay dentro de esa subcelda
subcell_atomids = np.zeros(np.concat((nmax, [16])), dtype=np.uint32)

tini = time.perf_counter()
for i, atom in enumerate(positions):
    n = list(np.floor((atom-O) / d).astype(np.uint16))
    count = subcell_count[tuple(n)]
    subcell_count[tuple(n)] += 1
    n.append(count)
    subcell_atomids[tuple(n)] = ids[i]

t = time.perf_counter()-tini
print("Subdivision hecha en {:.4f} s".format(t))
np.save('subcell_ids.npy', subcell_atomids)