import os
import pathlib
import numpy as np
import ase.io
from ase import Atoms
from ase.io import read, write
# Ruta a la carpeta data/ dentro del paquete
_DATA_DIR = pathlib.Path(__file__).parent / "data"
def make_cell(material, n, method='ase'):
    """
    Genera una supercelda nxn del material indicado y la guarda en disco.
    Parameters
    ----------
    material : str
        Material a generar. Opciones: 'GRPH', 'BGDY', 'GDY'
    n : int
        Tamaño de la supercelda (n x n x 1)
    method : str, opcional
        Solo para GRPH. 'ase' genera desde cero (default), 'mj' usa el fichero MJ.
    Examples
    --------
    >>> make_cell('GRPH', 4)          # grafeno desde ASE
    >>> make_cell('GRPH', 8, 'mj')    # grafeno desde fichero MJ
    >>> make_cell('BGDY', 3)
    >>> make_cell('GDY', 2)
    """
    if material == 'GRPH':
        if method == 'ase':
            from ase.build import graphene
            unit = graphene()
        elif method == 'mj':
            unit = read(str(_DATA_DIR / "graphene_4x4_MJ_centered.xyz"))
            unit.center()
        else:
            raise ValueError(f"Método '{method}' no reconocido para GRPH. Opciones: 'ase', 'mj'")
    elif material == 'BGDY':
        # Boron-Graphdiyne
        # Extracted from: https://pubs.rsc.org/en/content/articlelanding/2018/ta/c8ta02627k
        cell = [
            [11.8467556014, 0.0, 0.0],
            [5.9233778007, 10.2595913032, 0.0],
            [0.0, 0.0, 20.0]
        ]
        positions_frac = [
            (0.4070313359871207, 0.1859316676351952, 0.5),
            (0.4672120785149971, 0.0655705418981327, 0.5),
            (0.4070313102567411, 0.4070303586203110, 0.5),
            (0.4672113650511491, 0.4672112185712862, 0.5),
            (0.5327836106014843, 0.5327839635983835, 0.5),
            (0.5929637664429919, 0.5929648549585735, 0.5),
            (0.5929638727348205, 0.8140640368300751, 0.5),
            (0.5327845410920844, 0.9344251783120043, 0.5),
            (0.1859315430662534, 0.4070331135330179, 0.5),
            (0.0655704174789662, 0.4672136888685117, 0.5),
            (0.8140642497539758, 0.5929627139400750, 0.5),
            (0.9344251513748954, 0.5327831630738729, 0.5),
            (0.3333313296681197, 0.3333314582972733, 0.5),
            (0.6666641611622097, 0.6666642439564683, 0.5),
        ]
        symbols = ['C'] * 12 + ['B'] * 2
        positions = [np.dot(pos, cell) for pos in positions_frac]
        unit = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])
    elif material == 'GDY':
        # gamma-Graphdiyne
        # Extracted from relaxed 2x2 supercell (MACE-HOC potential)
        # a = b ≈ 9.455 Å, γ ≈ 60°, 18 C atoms per unit cell
        cell = np.array([
            [9.45543695, 0.00269498, 0.0],
            [4.72771848, 8.18872055, 0.0],
            [0.0,        0.0,        14.0022172],
        ])
        positions_frac = np.array([
            (0.348483, 0.499704, 0.500159),  # sp2 (benzene)
            (0.200804, 0.499825, 0.500210),  # sp  (C≡C inner)
            (0.070202, 0.500571, 0.500234),  # sp  (C≡C outer)
            (0.651069, 0.499985, 0.500089),  # sp2 (benzene)
            (0.798789, 0.499826, 0.500091),  # sp  (C≡C inner)
            (0.929442, 0.498989, 0.500103),  # sp  (C≡C outer)
            (0.348422, 0.651072, 0.500057),  # sp2 (benzene)
            (0.200831, 0.798762, 0.499914),  # sp  (C≡C inner)
            (0.070694, 0.929142, 0.499796),  # sp  (C≡C outer)
            (0.499668, 0.651248, 0.500085),  # sp2 (benzene)
            (0.499296, 0.799074, 0.500080),  # sp  (C≡C inner)
            (0.498634, 0.929646, 0.500092),  # sp  (C≡C outer)
            (0.499922, 0.348407, 0.500154),  # sp2 (benzene)
            (0.500323, 0.200584, 0.500196),  # sp  (C≡C inner)
            (0.501068, 0.069985, 0.500214),  # sp  (C≡C outer)
            (0.651151, 0.348611, 0.500055),  # sp2 (benzene)
            (0.798896, 0.201072, 0.499906),  # sp  (C≡C inner)
            (0.929190, 0.070860, 0.499781),  # sp  (C≡C outer)
        ])
        symbols = ['C'] * 18
        positions = positions_frac @ cell   # producto matricial correcto
        unit = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])
    else:
        raise ValueError(f"Material '{material}' no reconocido. Opciones: 'GRPH', 'BGDY', 'GDY'")
    name = f"{n}x{n}"
    supercel = unit * [n, n, 1]
    os.makedirs(f"{material}/{name}", exist_ok=True)
    ase.io.write(f"{material}/{name}/{material.lower()}_{name}.xyz", supercel)
    print(f"✔ {material} {name} — {len(supercel)} átomos")
