#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: Par-LJGen Lennard-Jones + Coulomb Parameter Optimization - Version 0.1
Generates pairwise interaction data, histograms, and comparison plots.
Authors: Roberta Dias and Julio C. S. Da Silva
Date: 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv
import argparse
from collections import defaultdict

# --- Passagem de Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--epsilon_r", type=float, default=10.0, help="Constante dielétrica")
args = parser.parse_args()
epsilon_r = args.epsilon_r

# --- Leitura do arquivo input.txt ---
frames = []
pair_data = []
unique_atoms = {"Mo": [], "O": [], "C": []}

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

# --- Parâmetros iniciais ---
def avg_params(atom_list):
    arr = np.array(atom_list)
    return np.mean(arr[:, 0]), np.mean(arr[:, 1])

eps_mo, sig_mo = avg_params(unique_atoms["Mo"])
eps_o, sig_o = avg_params(unique_atoms["O"])
eps_c, sig_c = avg_params(unique_atoms["C"])

initial_guess = [eps_mo, sig_mo, eps_o, sig_o, eps_c, sig_c]
bounds = [(0.01, 3.0), (1.5, 6.0)] * 4

# --- Cálculo da energia de interação---
def compute_energies(params):
    eps_mo, sig_mo, eps_o, sig_o, eps_c, sig_c = params
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

# --- Processo de Otimização ---
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

# --- Resultado final ---
distances, lj_e, coul_e, total_e = compute_energies(params_opt)
abs_errors = np.abs(total_e - edft_values)

# --- Output final ---
with open("output.txt", "w") as out:
    out.write(f"# Constante dielétrica usada: epsilon_r = {epsilon_r}\n")
    out.write("# Parâmetros iniciais\n")
    for lbl, e, s in zip(["Mo", "O", "C"], initial_guess[::2], initial_guess[1::2]):
        out.write(f"{lbl}: epsilon = {e:.6f}, sigma = {s:.6f}\n")
    out.write("# Parâmetros otimizados\n")
    for lbl, e, s in zip(["Mo", "O", "C"], params_opt[::2], params_opt[1::2]):
        out.write(f"{lbl}: epsilon = {e:.6f}, sigma = {s:.6f}\n")
    out.write("\nr(Å)\tE_DFT\tE_LJ\tE_Coul\tE_Total\tErro_abs\n")
    for r, edft, lj, cl, tot, err in zip(distances, edft_values, lj_e, coul_e, total_e, abs_errors):
        out.write(f"{r:.4f}\t{edft:.6f}\t{lj:.6f}\t{cl:.6f}\t{tot:.6f}\t{err:.6f}\n")

# --- Gráficos ---
idx = np.argsort(distances)
plt.figure(figsize=(8, 6))
plt.plot(distances[idx], edft_values[idx], 'ko', label="DFT-D3")
plt.plot(distances[idx], lj_e[idx], 'b--', label="LJ")
plt.plot(distances[idx], coul_e[idx], 'g:', label="Coulomb")
plt.plot(distances[idx], total_e[idx], 'r-', label="Total (Coul_LJ)")
plt.fill_between(distances[idx], total_e[idx] - abs_errors[idx], total_e[idx] + abs_errors[idx], alpha=0.2, color="red")
plt.xlabel("Distância Mo-C (Å)")
plt.ylabel("Energia (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Curvas_Einteracao_MoO4-CO2.tiff", dpi=300)
plt.close()

# --- Exportação de valores das interações de pares---
def export_pairwise():
    with open("pairwise_interactions.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["frame", "atom_i", "atom_j", "r_ij", "V_LJ_ij"])
        writer.writeheader()
        for entry in pair_data:
            writer.writerow(entry)

    # Dispersão
    rij = [d["r_ij"] for d in pair_data]
    vij = [d["V_LJ_ij"] for d in pair_data]
    plt.figure()
    plt.scatter(rij, vij, s=10, alpha=0.6)
    plt.xlabel("r_ij (Å)")
    plt.ylabel("V_LJ_ij (kcal/mol)")
    plt.title("Dispersão pares i-j")
    plt.tight_layout()
    plt.savefig("pares_lj_dispersao.tiff", dpi=300)
    plt.close()

    # Histogramas
    grouped = defaultdict(list)
    for d in pair_data:
        k = f"{d['atom_i']}-{d['atom_j']}"
        grouped[k].append(d["V_LJ_ij"])

    for k, v in grouped.items():
        plt.figure()
        plt.hist(v, bins=30, alpha=0.7)
        plt.title(f"Histograma: {k}")
        plt.xlabel("V_LJ_ij (kcal/mol)")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"hist_{k.replace('-', '_')}.tiff", dpi=300)
        plt.close()

export_pairwise()
