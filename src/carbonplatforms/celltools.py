import os
import pathlib
import numpy as np
import ase.io
from ase import Atoms
from ase.io import read, write

_DATA_DIR = pathlib.Path(__file__).parent / "data"


def build_gamma_graphdiyne():
    """
    Unit cell of gamma-graphdiyne (γ-GDY) extracted from a relaxed 2x2 supercell.
    
    cell = np.array([
        [9.45543695, 0.00269498, 0.0],
        [4.72771848, 8.18872055, 0.0],
        [0.0,        0.0,        14.0022172],
    ])
    positions = np.array([
        [ 5.65751585,  4.09287801,  7.00333486],  
        [ 4.26171852,  4.09346746,  7.00404853],
        [ 3.03035092,  4.09922335,  7.00438148], 
        [ 8.51993135,  4.09599095,  7.00234908],  
        [ 9.91593055,  4.09508536,  7.00238103], 
        [11.14736447,  4.08858695,  7.00254780], 
        [ 6.37256632,  5.33238265,  7.00190738],  
        [ 5.67526647,  6.54138322,  6.99990413],  
        [ 5.06116713,  7.60867547,  6.99825265], 
        [ 7.80349757,  5.33423333,  7.00229545], 
        [ 8.49886400,  6.54474266,  7.00223005],  
        [ 9.10990186,  7.61395232,  7.00239376],  
        [ 6.37414806,  2.85435758,  7.00325852],  
        [ 5.67907334,  1.64387258,  7.00384310], 
        [ 5.06868970,  0.57443803,  7.00410623], 
        [ 7.80504711,  2.85643116,  7.00186931],  
        [ 8.50452430,  1.64867511,  6.99979041],  
        [ 9.12090150,  0.58275892,  6.99804203], 
    ])
    gdy = Atoms(
        symbols=['C'] * 18,
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )
    gdy.center(vacuum=7.5, axis=2)
    return gdy

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
    >>> make_cell('GDY', 2)           # γ-graphdiyne desde función interna
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
        unit = build_gamma_graphdiyne()   # generada internamente, sin fichero externo

    else:
        raise ValueError(f"Material '{material}' no reconocido. Opciones: 'GRPH', 'BGDY', 'GDY'")

    name = f"{n}x{n}"
    supercel = unit * [n, n, 1]
    os.makedirs(f"{material}/{name}", exist_ok=True)
    ase.io.write(f"{material}/{name}/{material.lower()}_{name}.xyz", supercel)
    print(f"✔ {material} {name} — {len(supercel)} átomos")
