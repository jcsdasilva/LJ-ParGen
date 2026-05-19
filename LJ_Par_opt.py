#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
================================================================================
Lennard-Jones Parameter Optimization Framework
================================================================================
Description:
    This program performs automated non-linear parameter optimization for 
    non-bonded interactions. It refines Lennard-Jones (12-6) parameters 
    coupled with Coulombic electrostatic potentials by fitting classical 
    interaction energy surfaces against reference quantum mechanics data 
    calculated via Density Functional Theory (DFT).

Authors: 
    Júlio C. S. Da Silva and Roberta Dias
    Instituto de Química e Biotecnologia
    Universidade Federal de Alagoas (UFAL)
    
E-mail: julio.silva@iqb.ufal.br and roberta.dias@iqb.ufal.br

Version: 1.0
Language: Python 3
Dependencies: numpy, matplotlib, scipy
================================================================================
"""





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse


# --- Command line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--epsilon_r", type=float, default=10.0, help="Dielectric constant")
args = parser.parse_args()
epsilon_r = args.epsilon_r

# --- Reading the input.txt file ---
frames = []
pair_data = []
unique_atoms = {"Mo": [], "O": [], "C": [], "H": []}

with open("input.txt", "r") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        frame_id = int(lines[i].strip())
        na, nb, refa, refb, edft = lines[i+1].strip().split()
        na, nb, refa, refb = map(int, [na, nb, refa, refb])
        edft = float(edft)
        atoms = []
        for j in range(na + nb):
            parts = lines[i+2+j].strip().split()
            symbol = parts[0]
            x, y, z = map(float, parts[1:4])
            charge = float(parts[4])
            epsilon = float(parts[5])
            sigma = float(parts[6])
            atoms.append({"symbol": symbol, "xyz": np.array([x, y, z]), "charge": charge, "epsilon": epsilon, "sigma": sigma})
            if symbol in unique_atoms:
                unique_atoms[symbol].append((epsilon, sigma))
        frames.append({"id": frame_id, "na": na, "nb": nb, "refa": refa, "refb": refb, "edft": edft, "atoms": atoms})
        i += 2 + na + nb

# --- Initial parameters estimation ---
def avg_params(atom_list):
    arr = np.array(atom_list)
    return np.mean(arr[:, 0]), np.mean(arr[:, 1])

eps_mo, sig_mo = avg_params(unique_atoms["Mo"])
eps_o, sig_o = avg_params(unique_atoms["O"])
eps_c, sig_c = avg_params(unique_atoms["C"])
eps_h, sig_h = avg_params(unique_atoms["H"])

initial_guess = [eps_mo, sig_mo, eps_o, sig_o, eps_c, sig_c, eps_h, sig_h]
bounds = [(0.01, 3.0), (1.5, 6.0)] * 4

# --- Energy calculation ---
def compute_energies(params):
    eps_mo, sig_mo, eps_o, sig_o, eps_c, sig_c, eps_h, sig_h = params
    lj_energies, coul_energies, total_energies, distances = [], [], [], []
    pair_data.clear()

    for frame in frames:
        atoms = frame["atoms"]
        na, nb = frame["na"], frame["nb"]
        refa, refb = frame["refa"] - 1, frame["refb"] - 1

        for a in atoms:
            if a["symbol"] == "Mo": a["epsilon"], a["sigma"] = eps_mo, sig_mo
            elif a["symbol"] == "O": a["epsilon"], a["sigma"] = eps_o, sig_o
            elif a["symbol"] == "C": a["epsilon"], a["sigma"] = eps_c, sig_c
            elif a["symbol"] == "H": a["epsilon"], a["sigma"] = eps_h, sig_h

        lj_total = 0.0
        coul_total = 0.0
        for i in range(na):
            for j in range(na, na + nb):
                ai, aj = atoms[i], atoms[j]
                rij = np.linalg.norm(ai["xyz"] - aj["xyz"])
                sigma_ij = 0.5 * (ai["sigma"] + aj["sigma"])
                epsilon_ij = np.sqrt(ai["epsilon"] * aj["epsilon"])
                sr6 = (sigma_ij / rij) ** 6
                sr12 = sr6 * sr6
                lj = 4 * epsilon_ij * (sr12 - sr6)
                coul = 332.06371 * ai["charge"] * aj["charge"] / (epsilon_r * rij)
                lj_total += lj
                coul_total += coul

                pair_data.append({"frame": frame["id"], "atom_i": ai["symbol"], "atom_j": aj["symbol"], "r_ij": rij, "V_LJ_ij": lj})

        lj_energies.append(lj_total)
        coul_energies.append(coul_total)
        total_energies.append(lj_total + coul_total)
        distances.append(np.linalg.norm(atoms[refa]["xyz"] - atoms[refb]["xyz"]))

    return np.array(distances), np.array(lj_energies), np.array(coul_energies), np.array(total_energies)

# --- Optimization process ---
edft_values = np.array([f["edft"] for f in frames])

def cost(params):
    fixed = initial_guess.copy()
    fixed[0:4] = params[0:4]
    _, _, _, total_vals = compute_energies(fixed)
    err = np.mean(np.abs(total_vals - edft_values))
    reg = 0.1 * np.sum((params[0:4] - np.array(initial_guess[0:4]))**2)
    return err + reg

opt = minimize(cost, initial_guess[0:4], bounds=bounds[0:4], method="L-BFGS-B")
params_opt = initial_guess.copy()
params_opt[0:4] = opt.x


distances, lj_e, coul_e, total_e = compute_energies(params_opt)
abs_errors = np.abs(total_e - edft_values)

# --- Writing the final output file ---
with open("output.txt", "w") as out:
    out.write(f"# Dielectric constant used: epsilon_r = {epsilon_r}\n")
    out.write("# Initial parameters\n")
    for lbl, e, s in zip(["Mo", "O", "C", "H"], initial_guess[::2], initial_guess[1::2]):
        out.write(f"{lbl}: epsilon = {e:.6f}, sigma = {s:.6f}\n")
    out.write("# Optimized parameters\n")
    for lbl, e, s in zip(["Mo", "O", "C", "H"], params_opt[::2], params_opt[1::2]):
        out.write(f"{lbl}: epsilon = {e:.6f}, sigma = {s:.6f}\n")
    out.write("\nr(Å)\tE_DFT\tE_LJ\tE_Coul\tE_Total\tAbs_error\n")
    for r, edft, lj, cl, tot, err in zip(distances, edft_values, lj_e, coul_e, total_e, abs_errors):
        out.write(f"{r:.4f}\t{edft:.6f}\t{lj:.6f}\t{cl:.6f}\t{tot:.6f}\t{err:.6f}\n")

 #--- Plotting potential curves ---
idx = np.argsort(distances)
plt.figure(figsize=(8, 6))

# Espessura das linhas (lw)
plt.plot(distances[idx], edft_values[idx], 'ko', label="PBE-D3(BJ)", markersize=5) 
plt.plot(distances[idx], lj_e[idx], 'b--', label="LJ", lw=2.5)
plt.plot(distances[idx], coul_e[idx], 'g:', label="Coulomb", lw=2.5)
plt.plot(distances[idx], total_e[idx], 'r-', label="Total (Coul+LJ)", lw=2.5)

plt.fill_between(distances[idx], total_e[idx] - abs_errors[idx], total_e[idx] + abs_errors[idx], alpha=0.2, color="red")
plt.xlabel("Distance Mo-C (Å)", fontsize=14, weight='bold')
plt.ylabel(r"$\mathbf{U_{int}}$ (kcal/mol)", fontsize=14, weight='bold')


plt.tick_params(axis='both', which='major', labelsize=14)


plt.legend(fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.savefig("Eint_MoO4-CH4.tiff", dpi=600)
plt.close()

