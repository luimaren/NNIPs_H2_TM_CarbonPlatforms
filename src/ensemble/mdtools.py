import os
import time
import numpy as np

from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary,ZeroRotation
from ase.io import Trajectory, write, read
from ase.constraints import FixCartesian
from ase import units

def NoseHooverMD(
    init_conf,
    temp,
    calc,
    fname,
    save_interval,
    nsteps,
    init_temp=None,
    timestep_fs=1.0,
    tdamp_fs=1000.0,    
    log_interval=100,    
):
    conf = init_conf.copy()
    conf.calc = calc

    if init_temp is None:
        init_temp = temp
    if abs(init_temp - temp) > 200:
        print(f" Delta_T = {abs(init_temp - temp):.0f} K — considera equilibración larga")

    # ── Inicialización térmica ────────────────────────────────────────
    
    MaxwellBoltzmannDistribution(conf, temperature_K=init_temp)
    Stationary(conf)    
    ZeroRotation(conf)   

    # ── Limpiar archivos antiguos ─────────────────────────────────────
    
    for ext in [".traj", ".xyz", "_thermo.npz"]:
        if os.path.exists(fname + ext):
            os.remove(fname + ext)

    traj = Trajectory(fname + ".traj", "w", conf)

    dyn = NoseHooverChainNVT(
        atoms=conf,
        timestep=timestep_fs * units.fs,
        temperature_K=temp,
        tdamp=tdamp_fs * units.fs,
    )

    # ── Callbacks ────────────────────────────────────────────────────
    
    dyn.attach(traj.write, interval=save_interval)

    # Log termodinámico
    log_data = {"step": [], "T_K": [], "Epot": [], "Ekin": [], "Etot": []}

    def log_thermo():
        log_data["step"].append(dyn.get_number_of_steps())
        log_data["T_K"].append(conf.get_temperature())
        epot = conf.get_potential_energy() / len(conf)
        ekin = conf.get_kinetic_energy()   / len(conf)
        log_data["Epot"].append(epot)
        log_data["Ekin"].append(ekin)
        log_data["Etot"].append(epot + ekin)

    dyn.attach(log_thermo, interval=log_interval)

    # ── Dinámica ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dyn.run(nsteps)
    t1 = time.perf_counter()
    traj.close()

    # ── Exportar XYZ ──────────────────────────────────────────────────
    write(fname + ".xyz", Trajectory(fname + ".traj"))

    # ── Guardar log termodinámico ─────────────────────────────────────
    np.savez(fname + "_thermo.npz", **{k: np.array(v) for k, v in log_data.items()})

    # ── Resumen ───────────────────────────────────────────────────────
    total_time     = t1 - t0
    time_per_step  = total_time / nsteps
    T_arr          = np.array(log_data["T_K"])
    print("Nose–Hoover MD completada")
    print(f"  Pasos          : {nsteps}")
    print(f"  Tiempo total   : {total_time:.2f} s")
    print(f"  Tiempo/paso    : {time_per_step:.4f} s/step")
    print(f"  T media        : {T_arr.mean():.1f} ± {T_arr.std():.1f} K")
    print(f"  T final        : {T_arr[-1]:.1f} K")

    return {
        "total_time_s":    total_time,
        "time_per_step_s": time_per_step,
        "traj_file":       fname + ".traj",
        "xyz_file":        fname + ".xyz",
        "thermo_file":     fname + "_thermo.npz",
        "T_mean_K":        T_arr.mean(),
        "T_std_K":         T_arr.std(),
    }

def NoseHooverMD_2D(init_conf, temp, calc, fname, save_interval,
                   nsteps, init_temp=None,
                   timestep_fs=1.0, tdamp_fs=100.0):

    conf = init_conf.copy()
    conf.calc = calc

    if init_temp is None:
        init_temp = temp

    # Restricción: bloquear eje z
    constraint = FixCartesian(range(len(conf)), mask=(False, False, True))
    conf.set_constraint(constraint)

    # Inicialización térmica
    MaxwellBoltzmannDistribution(conf, temperature_K=init_temp)
    Stationary(conf)
    ZeroRotation(conf)

    # Eliminar velocidades en z
    vel = conf.get_velocities()
    vel[:, 2] = 0.0
    conf.set_velocities(vel)

    # Limpiar archivos antiguos
    for ext in [".traj", ".xyz"]:
        if os.path.exists(fname + ext):
            os.remove(fname + ext)

    traj = Trajectory(fname + ".traj", "w", conf)

    # Dinámica Nose–Hoover chain
    dyn = NoseHooverChainNVT(
        atoms=conf,
        timestep=timestep_fs * units.fs,
        temperature_K=temp,
        tdamp=tdamp_fs * units.fs
    )

    # Envolver átomos periódicamente
    dyn.attach(conf.wrap, interval=1)

    # Guardar trayectoria
    dyn.attach(traj.write, interval=save_interval)

    # Medición de tiempo
    t0 = time.perf_counter()
    dyn.run(nsteps)
    t1 = time.perf_counter()

    traj.close()

    # Exportar XYZ
    write(fname + ".xyz", Trajectory(fname + ".traj"))

    total_time = t1 - t0
    time_per_step = total_time / nsteps

    print("Nose–Hoover MD (2D, z fijo) completada")
    print(f"  Pasos: {nsteps}")
    print(f"  Tiempo total: {total_time:.2f} s")
    print(f"  Tiempo por paso: {time_per_step:.4f} s/step")

    return {
        "total_time_s": total_time,
        "time_per_step_s": time_per_step,
        "traj_file": fname + ".traj",
        "xyz_file": fname + ".xyz"
    }

def LangevinMD(init_conf, temp, calc, fname, save_interval, nsteps, init_temp=None, timestep_fs=1.0, friction=0.01):


    conf = init_conf.copy()
    conf.calc = calc

    # Inicialización térmica
    if init_temp is None:
        init_temp = temp

    MaxwellBoltzmannDistribution(conf, temperature_K=init_temp)
    Stationary(conf)
    ZeroRotation(conf)

    # Eliminar archivos previos
    for ext in [".traj", ".xyz"]:
        if os.path.exists(fname + ext):
            os.remove(fname + ext)

    traj = Trajectory(fname + ".traj", "w", conf)

    # Dinámica de Langevin
    dyn = Langevin(
        conf,
        timestep_fs * units.fs,
        temperature_K=temp,
        friction=friction
    )

    dyn.attach(traj.write, interval=save_interval)

    # Medición del tiempo
    t0 = time.perf_counter()
    dyn.run(nsteps)
    t1 = time.perf_counter()

    traj.close()

    # Exportar a XYZ
    write(fname + ".xyz", Trajectory(fname + ".traj"))

    total_time = t1 - t0
    time_per_step = total_time / nsteps

    print(f"MD completada:")
    print(f"  Pasos: {nsteps}")
    print(f"  Tiempo total: {total_time:.2f} s")
    print(f"  Tiempo por paso: {time_per_step:.4f} s/step")

    return {
        "total_time_s": total_time,
        "time_per_step_s": time_per_step,
        "traj_file": fname + ".traj",
        "xyz_file": fname + ".xyz",
    }
