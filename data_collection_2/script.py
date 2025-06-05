from functions import *
import pytools4dart as ptd
import pandas as pd
import random
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Définir les modèles de gaz et les longueurs d'onde
Gaz_Model = ['USSTD76', 'TROPICAL']
intervals = [
    np.linspace(0.3, 1.35, 650, dtype=np.float32),
    np.linspace(1.35, 1.410, 15, dtype=np.float32),
    np.linspace(1.410, 1.82, 250, dtype=np.float32),
    np.linspace(1.82, 1.93, 15, dtype=np.float32),
    np.linspace(1.93, 2.5, 270, dtype=np.float32),
]
lamda_values = np.concatenate(intervals)
batches=[]
batches[0] = lamda_values[0:350]
batches[1] = lamda_values[350:650]
batches[2] = lamda_values[650:900]      # → valeurs 200 à 399
batches[3] = lamda_values[900:930]
batches[4] = lamda_values[930:1200]
#lamda_values=[0.4,0.56,0.8,1.3]
colonnes = ["Tg_scat", "Tg_abs", "AOD", "SSA", "g1", "Z", "Zenith_Angle", "reflectance", "BOA_RT"]

# Fonction qui traite un batch de 200 longueurs d’onde
def run_batch(lambdas, batch_id):
    print(f"Batch {batch_id} started")
    sim = ptd.simulation(f'RTM_reference{batch_id}')
    Edit_lambda(lambdas,batch_id)
    l=len(lambdas)

    csv_path = f"batch_{batch_id}.csv"
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=colonnes).to_csv(csv_path, index=False)

    for i in range(600):
        model_gaz = random.choice(Gaz_Model)
        k_gaz = k_gaz_TROPICAL() if model_gaz == 'TROPICAL' else k_gaz_USDT()
        K_aero = K_aerosol()
        reflectance = np.random.rand(l).tolist()
        reflectance.insert(0, 1)
        theta = random.randint(0, 89)
        z = z_value()

        Edit_GasModel(model_gaz,batch_id)
        Edit_AerosolFactor(K_aero,batch_id)
        Edit_GasFactor(k_gaz,batch_id)
        Edit_reflectance(reflectance,batch_id)
        Edit_zenith_angle(theta,batch_id)
        Edit_altitude(z,batch_id)
        sim.run.full()

        atmosphere_params = atmosphere_param(z, theta,batch_id)
        BOA_RT = EXTRACT_BOA_RT(batch_id,l)

        df_temp = pd.DataFrame(atmosphere_params, columns=colonnes[:-2])
        df_temp['reflectance'] = reflectance[1:]
        df_temp['BOA_RT'] = BOA_RT
        df_temp.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"-----------------simulation num {i} in Batch {batch_id} finished.----------------------------")


# Point d’entrée principal
if __name__ == "__main__":
    # Créer l'en-tête s’il n’existe pas
    if not os.path.exists("new_data.csv"):
        pd.DataFrame(columns=colonnes).to_csv("new_data.csv", index=False)
    # Lancer les 6 processus en parallèle
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_batch, batch, i) for i, batch in enumerate(batches)]
        for future in futures:
            future.result()  # Assure-toi que tout est bien terminé

    # Fusionner les fichiers batchs dans un seul fichier final
    dfs = [pd.read_csv(f"batch_{i}.csv") for i in range(4)]
    final_df = pd.concat(dfs)
    final_df.to_csv("new_data.csv", index=False)

    # Nettoyer les fichiers temporaires
    '''for i in range(2):
        os.remove(f"batch_{i}.csv")'''
