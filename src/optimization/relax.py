import os
from ase.optimize import BFGS, FIRE, LBFGSLineSearch, FIRE2
from ase.io import write, read
from ase.filters import UnitCellFilter


def remove_if_exists(*filenames):
    """Elimina los ficheros indicados si existen."""
    for fname in filenames:
        if os.path.isfile(fname):
            os.remove(fname)

def relax_structure(atoms, calculator, optimizer="BFGS",
                     fmax=0.001, output_file="relax", steps=10000,
                     relax_cell=False):
    """
    Relaja una estructura con distintos optimizadores de ASE.
    Si relax_cell=True, también se relaja la celda (volumen y forma).
    """

    atoms = atoms.copy()
    atoms.calc = calculator

    optimizers = {
        "BFGS": BFGS,
        "LBFGSLineSearch": LBFGSLineSearch,
        "FIRE": FIRE,
        "FIRE2": FIRE2
    }

    if optimizer not in optimizers:
        raise ValueError(f"Optimizador no soportado: {optimizer}")

    # Nombres de archivos
    suffix = "_cell" if relax_cell else ""
    traj_file = f"{output_file}_{optimizer}{suffix}.traj"
    log_file = f"{output_file}_{optimizer}{suffix}.log"
    xyz_file = f"{output_file}_{optimizer}{suffix}.xyz"
    full_xyz_file = f"{output_file}_{optimizer}{suffix}_full.xyz"

    # Eliminar archivos antiguos
    remove_if_exists(traj_file, log_file, xyz_file, full_xyz_file)

    # Aplicar UnitCellFilter si se quiere relajar la celda
    system = UnitCellFilter(atoms) if relax_cell else atoms

    # Ejecutar optimización
    opt = optimizers[optimizer](system,
                                trajectory=traj_file,
                                logfile=log_file)
    opt.run(fmax=fmax, steps=steps)

    # Guardar estructura final
    write(xyz_file, atoms)

    # Guardar toda la trayectoria
    images = read(traj_file, ":")
    write(full_xyz_file, images)

    return atoms  
