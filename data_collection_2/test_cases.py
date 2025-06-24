from functions import *
import pytools4dart as ptd
import pandas as pd
import random
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import product

sim=ptd.simulation('RTM_reference0clear')
R_list = [0.1,0.4,0.7]
SZA_list = [15,30,60]
Z_list = [0,500,1000]
delta_lambda=[0,0.02,0.06]
k_aerosol=[1,2]
k_gaz=[1,2]
gaz_model=['USSTD76','TROPICAL']
all_combinations = list(product(R_list, SZA_list, Z_list, delta_lambda,k_aerosol,k_gaz,gaz_model))
colonnes = ["Tg_scat", "Tg_abs", "AOD", "SSA", "g1", "Z","Zenith_Angle", "reflectance","k_a","k_g","gaz_model","dl","BOA_RT"]

all_combinations=[[0.2,30,100,0.5,1,1,'USSTD76']]
for i in range(len(all_combinations)):
    R, SZA, Z, dl, ka, kg, g_model = all_combinations[i]
    #print(f"Processing combination {i+1}: Reflectance={R}, SZA={SZA}, Z={Z}")
    csv_path = "TEST_CASES.csv"
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=colonnes).to_csv(csv_path, index=False)
    
    Edit_GasModel(g_model,0)
    Edit_AerosolFactor(ka,0)
    Edit_GasFactor(kg,0)
    Edit_reflectance(R,0)
    Edit_delta_lambda(dl,0)
    Edit_zenith_angle(SZA,0)
    Edit_altitude(Z,0)
    sim.run.full()
    

    atmosphere_params = atmosphere_param(Z, SZA,0)
    BOA_RT = EXTRACT_BOA_RT(0,2)  

    df_temp = pd.DataFrame(atmosphere_params, columns=colonnes[:-6])
    df_temp['reflectance'] = R
    df_temp['K_aerosol']=ka
    df_temp['K_gaz']=kg
    df_temp['gaz_model']=g_model
    df_temp['BOA_RT'] = BOA_RT
    df_temp.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"-----------------simulation num {i} finished.----------------------------")
    



