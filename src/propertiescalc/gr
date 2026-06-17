import numpy as np

def g_r_2d_all_pairs(traj, rmax, nbins):

    dr = rmax / nbins
    bins = np.linspace(0.0, rmax, nbins+1)

    # Inicializar contadores
    g_counts_total = np.zeros(nbins)
    g_counts_CC = np.zeros(nbins)
    g_counts_BC = np.zeros(nbins)
    g_counts_BB = np.zeros(nbins)

    # Primer frame para celda y PBC
    atoms0 = traj[0]
    cell = atoms0.cell
    pbc = atoms0.pbc

    assert pbc[0] and pbc[1], "System must be periodic in x and y"
    Lx, Ly = cell[0,0], cell[1,1]
    area = Lx * Ly

    # Contar pares posibles
    symbols0 = atoms0.get_chemical_symbols()

    N_total = len(symbols0)

    N_C = sum(s == 'C' for s in symbols0)
    N_B = sum(s == 'B' for s in symbols0)

    N_pairs_total = N_total * (N_total - 1) / 2
    N_pairs_CC = N_C * (N_C - 1) / 2
    N_pairs_BC = N_B * N_C
    N_pairs_BB = N_B * (N_B - 1) / 2

    # Recorrer frames
    for atoms in traj:

        pos = atoms.positions[:, :2]
        symbols = atoms.get_chemical_symbols()
        N_atoms = len(symbols)

        for i in range(N_atoms):
            for j in range(i+1, N_atoms):

                # Distancia mínima imagen x-y

                dx = pos[j,0] - pos[i,0]
                dy = pos[j,1] - pos[i,1]
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
                r = np.sqrt(dx*dx + dy*dy)

                if r >= rmax:
                    continue

                bin_index = int(r / dr)

                # Contador total

                g_counts_total[bin_index] += 1

                # Contadores parciales
                si, sj = symbols[i], symbols[j]
                if si=='C' and sj=='C':
                    g_counts_CC[bin_index] += 1

                elif (si=='B' and sj=='C') or (si=='C' and sj=='B'):
                    g_counts_BC[bin_index] += 1

                elif si=='B' and sj=='B':
                    g_counts_BB[bin_index] += 1

    # Normalización 2D

    r_vals = 0.5 * (bins[1:] + bins[:-1])

    g_total = np.zeros_like(r_vals)
    g_CC = np.zeros_like(r_vals)
    g_BC = np.zeros_like(r_vals)
    g_BB = np.zeros_like(r_vals)

    for i, r in enumerate(r_vals):

        shell_area = 2 * np.pi * r * dr
        g_total[i] = g_counts_total[i] / (len(traj) * N_pairs_total * shell_area / area)
        g_CC[i] = g_counts_CC[i] / (len(traj) * N_pairs_CC * shell_area / area)
        g_BC[i] = g_counts_BC[i] / (len(traj) * N_pairs_BC * shell_area / area)
        g_BB[i] = g_counts_BB[i] / (len(traj) * N_pairs_BB * shell_area / area)

    g_dict = {'total': g_total, 'C-C': g_CC, 'B-C': g_BC, 'B-B': g_BB}

    return r_vals, g_dict

import numpy as np

def g_r_2d_tot(traj, rmax, nbins):

    dr = rmax / nbins
    bins = np.linspace(0.0, rmax, nbins+1)

    # Inicializar contadores
    g_counts_total = np.zeros(nbins)

    # Primer frame para celda y PBC
    atoms0 = traj[0]
    cell = atoms0.cell
    pbc = atoms0.pbc

    assert pbc[0] and pbc[1], "System must be periodic in x and y"
    Lx, Ly = cell[0,0], cell[1,1]
    area = Lx * Ly

    # Contar pares posibles
    symbols0 = atoms0.get_chemical_symbols()

    N_total = len(symbols0)

    N_pairs_total = N_total * (N_total - 1) / 2

    # Recorrer frames
    for atoms in traj:

        pos = atoms.positions[:, :2]
        symbols = atoms.get_chemical_symbols()
        N_atoms = len(symbols)

        for i in range(N_atoms):
            for j in range(i+1, N_atoms):

                # Distancia mínima imagen x-y

                dx = pos[j,0] - pos[i,0]
                dy = pos[j,1] - pos[i,1]
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
                r = np.sqrt(dx*dx + dy*dy)

                if r >= rmax:
                    continue

                bin_index = int(r / dr)

                # Contador total

                g_counts_total[bin_index] += 1

    # Normalización 2D

    r_vals = 0.5 * (bins[1:] + bins[:-1])

    g_total = np.zeros_like(r_vals)

    for i, r in enumerate(r_vals):

        shell_area = 2 * np.pi * r * dr
        g_total[i] = g_counts_total[i] / (len(traj) * N_pairs_total * shell_area / area)

    return r_vals, g_total

