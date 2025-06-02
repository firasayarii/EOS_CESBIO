from concurrent.futures import ThreadPoolExecutor, as_completed
import pytools4dart as ptd
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from functions import Edit_a_opt_depth,Edit_altitude,Edit_a_sca_albedo,Edit_g1,Edit_reflectance,Edit_Transmittance_gaz_absorption,Edit_Transmittance_gaz_scattering,Edit_zenith_angle,calculate_alpha,Edit_lambda,EXTRACT_BOA_RT
import os 

def run_simulation(i, row, n):
    sim=ptd.simulation(f'RTM_reference{n}')
    (Tg_scat,Tg_abs,Ta_abs,SSA,GOD,AOD,AODS,SZA,Z,R_scence,
     g1,Cos_SZA,mu_g,mu_a,muprime_g,muprime_a,mprime_g,mprime_a,
     _, _) = row

    # Mettre à jour les fichiers spécifiques pour cette simulation
    Edit_zenith_angle(SZA, n)
    Edit_reflectance(R_scence, n)
    Edit_altitude(Z, n)
    Edit_g1(g1, n)
    Edit_a_sca_albedo(SSA, n)
    Edit_a_opt_depth(AOD, n)
    Edit_Transmittance_gaz_absorption(Tg_abs, n)
    Edit_Transmittance_gaz_scattering(Tg_scat, n)

    # Lancer la simulation Dart
    sim.run.full()  # si sim accepte un identifiant, sinon ignore

    print(f'Simulation numéro {i} (n={n}) terminée')

    BOA = EXTRACT_BOA_RT(n)
    alpha = calculate_alpha(SZA, Z, Tg_abs, Tg_scat, AOD, SSA, n)

    return i, BOA, alpha

data = pd.read_csv('data.csv')
df = data[0:3].copy()

batch_size = 3
num_simulations = len(df)
num_batches = math.ceil(num_simulations / batch_size)

output_file = "results.csv"
if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
    with open(output_file, "w") as f:
        f.write("index,BOA_RT,alpha\n")

f = open(output_file, "a")

for batch in range(num_batches):
    start = batch * batch_size
    end = min(start + batch_size, num_simulations)
    print(f"🌀 Lancement du batch {batch + 1}/{num_batches}")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, row_idx in enumerate(range(start, end)):
            global_index = df.index[row_idx]  # 👈 ici on garde l'index original de `data`
            row = df.loc[global_index]
            n = i
            futures.append(executor.submit(run_simulation, global_index, row, n))

        for future in as_completed(futures):
            i, BOA, alpha = future.result()
            f.write(f"{i},{BOA},{alpha}\n")  # 👈 `i` est bien l’index global (34575, etc.)

    print(f"✅ Batch {batch + 1} terminé.")

f.close()