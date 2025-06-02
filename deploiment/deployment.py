import numpy as np
import os
import subprocess
from pathlib import Path
import sys
from functions import *
from tensorflow.keras.models import load_model
import os


# Chemin de travail
working_dir = Path.cwd()

# Extraire le nom du dossier courant (ex: 'Analytic')
simulation = working_dir.name

# Définir les chemins vers les fichiers nécessaires
#phase_path = working_dir / 'input' / 'phase.xml'
atmosphere_nc = working_dir / 'output' / 'atmosphere.nc'
maket_scn= working_dir / 'output' / 'maket.scn'
phase_scn=  working_dir / 'output' / 'phase.scn'
simulation_proprerties=  working_dir / 'output' / 'simulation.properties.txt'
atmosphere_xml = find_files_xml(working_dir, 'atmosphere.xml')
directions_path = find_files_xml(working_dir, 'directions.xml')
coeff_diff_path = find_files_xml(working_dir, 'coeff_diff.xml')
maket_path = find_files_xml(working_dir,'maket.xml')
# Définir les répertoires DART
DART_LOCAL = find_paths(working_dir,'user_data')
DART_HOME = find_paths(working_dir,'DART')
AI_Tools= DART_HOME / 'tools' / 'AI'
if sys.platform == 'win32':
    DART_TOOLS = DART_HOME / 'tools' / 'windows'
else:
    DART_TOOLS = DART_HOME / 'tools' / 'linux'

#print(working_dir)
#print(directions_path)
scaler_y_path=AI_Tools / 'scaler_y_DART.pkl'
scaler_y=joblib.load(scaler_y_path)
SZA=get_SZA(directions_path)
z=get_altitude(maket_path)
reflectance=get_reflectance(maket_scn,phase_scn)
atmosphere_lists=atmosphere_param(atmosphere_nc,atmosphere_xml)
print(atmosphere_lists)
df=prepare_features(atmosphere_lists,SZA,z,reflectance,AI_Tools)
#df.to_csv(AI_Tools / 'new_samples_features.csv',index=False)
model=load_model(AI_Tools / 'deep_model.h5')
prediction=model.predict(df)
print(prediction)
predictions_list = prediction.flatten().tolist()
predictions_array = np.array(predictions_list).reshape(-1, 1)

# Inverser la mise à l’échelle
predictions_original_scale = scaler_y.inverse_transform(predictions_array)
predictions_list=predictions_original_scale.flatten().tolist()
E_TOA=EXTRACT_TOA(simulation_proprerties,SZA)
E_BOA=calculate_BOA_TOTAL(SZA,z,predictions_list,atmosphere_lists,E_TOA)
E_direct=EXTRACT_E_direct(phase_scn,SZA)
if len(E_BOA)==len(E_direct) :
    E_diffus=[i-j for i,j in zip(E_BOA,E_direct)]
else :
    E_diffus=[E_BOA[0]-i for i in E_direct]



#print(df.shape)
#print("Répertoire courant :", os.getcwd())

edit_phase_scn(phase_scn, E_diffus)
