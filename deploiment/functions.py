import xml.etree.ElementTree as ET
import pandas as pd
import joblib
from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import math
import re
from tensorflow.keras.models import load_model

def get_RTMode(phase):
    tree = ET.parse(phase)  # Remplace avec ton chemin
    root = tree.getroot()

    # Trouver la balise AtmosphereRadiativeTransfer
    try :
        toa_to_boa_value = None
        for elem in root.iter('AtmosphereRadiativeTransfer'):
            toa_to_boa_value = elem.get('TOAtoBOA')
            break  # On s'arrête dès qu'on trouve la première

        # Affichage du résultat
        if toa_to_boa_value is not None:
            return int(toa_to_boa_value)
    except:
        raise ValueError("--------- Atmosphere Radiative Transfer Mode NOT FOUND ---------")

def find_paths(path: Path, keyword: str):
    for parent in path.parents:
        if keyword in parent.name:
            return parent
    raise ValueError(f"--------- NO PARENT DIRECTORY CONTAINING '{keyword}' FOUND. ---------")





def find_files_xml(working_dir,file):
    try : 
        # Premier emplacement : working_dir/input/atmosphere.xml
        path1 = working_dir / 'input' / file
        if path1.exists():
            return path1

        # Deuxième emplacement : working_dir/../../input/atmosphere.xml
        path2 = working_dir.parent.parent / 'input' / file
        if path2.exists():
            return path2

        # Si aucun des deux n'est trouvé
    except :
        raise ValueError(f'--------- FILE {file} NOT FOUND. ---------')


def atmosphere_param(atmosphere_nc,atmosphere_xml):
    tree = ET.parse(atmosphere_xml)  # Remplace 'votre_fichier.xml' par le nom réel
    root = tree.getroot()
    l=[]
    # Trouver l'élément <IsAtmosphere> et lire l'attribut 'typeOfAtmosphere'
    is_atmosphere_elem = root.find('.//IsAtmosphere')  # Recherche récursive
    type_value = int(is_atmosphere_elem.get('typeOfAtmosphere'))
    if type_value == 1 :

        with Dataset(atmosphere_nc, "r") as nc_file:
            spectral_number=nc_file.groups['Transmittance']['All_gas_transmittance'].shape[0]
            for i in range(spectral_number):
                tg_scat = nc_file.groups['Transmittance']['All_gas_transmittance'][i][0]
                tg_abs = np.prod(nc_file.groups['Transmittance']['All_gas_transmittance'][i][1:15])
                optical_depth = nc_file.groups['Transmittance']['Aerosols optical_depth'][i][0]
                albedo = nc_file.groups['Transmittance']['Aerosol_single_scattering albedo'][i][0]
                a = nc_file.groups['Henyey Greenstein']['a'][i][0]
                g1 = nc_file.groups['Henyey Greenstein']['g1'][i][0]
                g2 = nc_file.groups['Henyey Greenstein']['g2'][i][0]
                l.append([tg_scat, tg_abs, optical_depth, albedo, a, g1])
            return l
    else :

        with Dataset(atmosphere_nc, "r") as nc_file:
            GOD = nc_file.getncattr('Gas_optical_depth_scattering')
            tg_scat=math.exp(-GOD)
            tg_abs = nc_file.getncattr('Transmittance_of_gases_absorb')
            optical_depth = nc_file.getncattr('Total_aerosols_optical_depth')
            albedo = nc_file.getncattr('Aerosol_albedo')
            a = nc_file.getncattr('Henyey_Greenstein_a')
            g1 = nc_file.getncattr('Henyey_Greenstein_g1')
            g2 = nc_file.getncattr('Henyey_Greenstein_g2')
            l.append([tg_scat, tg_abs, optical_depth, albedo, a, g1])
            return l
        
def prepare_features(atmosphere_list,SZA,z,reflectance_values):


    #bins_a = np.load(AI_path / "mprime_a_bins_DART.npy")
    #bins_g = np.load(AI_path / 'mprime_g_bins_DART.npy')
    
    scaler=joblib.load( "scaler_SS_BOA_DART.pkl")
    #encoder=joblib.load(AI_path / "encoder_OH_DART.pkl")

    def muprime(z,h,µ):
        RAYON_TERRESTRE=6371
        eta = (RAYON_TERRESTRE*1000 + z) / h
        root = (eta*µ)**2  + 2 * eta + 1
        sum = (root)**0.5 - eta * µ
        if sum > 0 :
            return 1/sum
        return 1 
    
    new_data=[]
    if len(atmosphere_list) == len(reflectance_values) :
        
        for i in range(len(atmosphere_list)) :
            Hg=9000
            Ha=2000
            Tg_scat, Tg_abs, AOD, SSA, a, g1  = atmosphere_list[i]
            GOD=-math.log(Tg_scat)
            µ=math.cos(math.radians(SZA))
            Ta_abs=math.exp(-AOD*(1-SSA))
            AODS=AOD*SSA
            mu_g=(6371*1000+z)/Hg
            mu_a=(6371*1000+z)/Ha
            muprime_g=muprime(z,Hg,µ)
            muprime_a=muprime(z,Ha,µ)
            mprime_g=math.exp(-z/Hg)/muprime_g
            mprime_a=math.exp(-z/Ha)/muprime_a
            #bin_g = np.digitize(mprime_g, bins_g)
            #bin_a = np.digitize(mprime_a, bins_a)
            new_l = [Tg_scat,Tg_abs, Ta_abs,SSA,GOD,AOD, AODS,SZA,z,reflectance_values[i],g1, µ,mu_g, mu_a, muprime_g, muprime_a, mprime_g, mprime_a]
            new_data.append(new_l)
    else :

        for i in range(len(atmosphere_list)) :
            Hg=9000
            Ha=2000
            Tg_scat, Tg_abs, AOD, SSA, a, g1 = atmosphere_list[i]
            GOD=-math.log(Tg_scat)
            µ=math.cos(math.radians(SZA))
            Ta_abs=math.exp(-AOD*(1-SSA))
            AODS=AOD*SSA
            mu_g=(6371*1000+z)/Hg
            mu_a=(6371*1000+z)/Ha
            muprime_g=muprime(z,Hg,µ)
            muprime_a=muprime(z,Ha,µ)
            mprime_g=math.exp(-z/Hg)/muprime_g
            mprime_a=math.exp(-z/Ha)/muprime_a
            #bin_g = np.digitize(mprime_g, bins_g)
            #bin_a = np.digitize(mprime_a, bins_a)
            new_l = [Tg_scat,Tg_abs, Ta_abs,SSA,GOD,AOD, AODS,SZA,z,reflectance_values[0],g1, µ,mu_g, mu_a, muprime_g, muprime_a, mprime_g, mprime_a]
            new_data.append(new_l)
    columns = [
    'Tg_scat','Tg_abs','Ta_abs','SSA','GOD','AOD','AODS','SZA','Z','R_scene','g1','Cos(SZA)','mu_g', 'mu_a', 'muprime_g',
    'muprime_a', 'mprime_g','mprime_a']
    df = pd.DataFrame(new_data, columns=columns)
    #df.to_csv(AI_path / 'new_samples_features_2.csv',index=False)
    cols_to_scale = df.columns
    #cols_to_encode = ['mprime_a_bin', 'mprime_g_bin']
    X_scaled = scaler.transform(df[cols_to_scale])
    #X_bins_encoded = encoder.transform(df[cols_to_encode])
    #encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)
    #df.to_csv('new_samples_features_16.csv',index=False)
    # Construction du DataFrame final
    df_scaled = pd.DataFrame(
            X_scaled,
            columns=cols_to_scale,
            index=df.index  # Pour conserver les bons index
        )

    return df_scaled

def get_SZA(directions_path) :
    # Parser le XML
    tree = ET.parse(directions_path)
    root = tree.getroot()
    # Chercher la balise <SunViewingAngles>
    sva = root.find(".//SunViewingAngles")
    # Extraire la valeur de l'attribut sunViewingZenithAngle
    if sva is not None:
        zenith_angle = sva.get("sunViewingZenithAngle")
        return float(zenith_angle)
    else:
        raise ValueError("--------- Sun Viewing Zenith Angle NOT FOUND. AI MODE WORKS ONLY WITH VIEWING ANGLE. ---------")
    
def get_reflectance(macket_scn,phase_scn):
    try :
        pattern_id = re.compile(r"scene\.objects\.object[\d_\-]+\.dartNameId=fg0")
        with open(macket_scn, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if pattern_id.search(line):
                break
        target_line = lines[i + 4].strip()
        value = target_line.split('=', 1)[1]
        pattern_ref = re.compile(fr"scene\.materials\.{re.escape(value)}\.ref=")
        for i, line in enumerate(lines):
            if pattern_ref.search(line):
                break
        ref = line.split('=', 1)[1]
        ref = ref.strip()
        pattern_value = re.compile(fr"scene\.materials\.{re.escape(ref)}\.(kd|kr)=(.+)")
        with open(phase_scn, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if pattern_value.search(line):
                values = line.split('=', 1)[1]
                reflectance_values=values.split(' ')
                reflectance_values=[float(i.strip()) for i in reflectance_values]
                return reflectance_values
    except :
        raise ValueError("--------- SCENE REFLECTANCE VALUE NOT FOUND. ---------") 
        


def get_altitude(directions_path):
    # Parser le XML
    tree = ET.parse(directions_path)
    root = tree.getroot()
    # Chercher la balise <SunViewingAngles>
    values = root.find(".//LatLon")
    # Extraire la valeur de l'attribut sunViewingZenithAngle
    if values is not None:
        altitude = values.get("altitude")
        return float(altitude)
    else:
        raise ValueError("--------- maket.xml FILE NOT FOUND. ---------")


def edit_phase_scn(filepath, new_value):
    parameter='scene.lights.sky.color'
    with open(filepath, 'r') as file:
        lines = file.readlines()

    with open(filepath, 'w') as file:
        for line in lines:
            if line.strip().startswith(parameter + "="):
                # Convertir chaque élément de la liste en string et les joindre avec des espaces
                value_str = ' '.join(str(val) for val in new_value)
                file.write(f"{parameter}={value_str}\n")
            else:
                file.write(line)

def EXTRACT_TOA(path, SZA):
    try:
        pattern = re.compile(r"phase\.band(\d+)\.spectral\.TOA:\s*([0-9.+-eE]+)")
        E_TOA = []

        with open(path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    band_index = int(match.group(1))
                    value = float(match.group(2))
                    corrected_value = value * math.cos(math.radians(SZA))


                    # Allonger la liste si nécessaire
                    while len(E_TOA) <= band_index:
                        E_TOA.append(0.0)  # Remplir avec des zéros


                    E_TOA[band_index] = corrected_value

        return E_TOA

    
    except Exception as e:
        print(f"--------- TOA IRRADIANCE VALUES NOT FOUND --------- \n{e}")
        

    

def EXTRACT_E_direct(path,SZA):
    try :
        with open(path, 'r') as file:
            for line in file:  # ← ici, on lit ligne par ligne
                if line.strip().startswith("scene.lights.sunlight.color="):
                    valeur_str = line.strip().split("=")[1]
                    return [ float(val)*math.cos(math.radians(SZA)) for val in valeur_str.split() ]
        return None  
    except :

        raise ValueError("--------- E_DIRECT VALUES NOT FOUND. ---------")





def get_spectral_mode(phase_path):
    try :
        tree = ET.parse(phase_path)  # Remplace par le chemin réel
        root = tree.getroot()

        # Initialiser la liste pour stocker les valeurs
        spectral_dart_modes = []

        # Parcourir tous les éléments SpectralIntervalsProperties
        for elem in root.iter('SpectralIntervalsProperties'):
            val = elem.get('spectralDartMode')
            if val is not None:
                spectral_dart_modes.append(int(val))
    except :
        raise ValueError("--------- SPECTRAL MODE NOT FOUND. ---------") 

    return spectral_dart_modes

def EXTRACT_E_diffus(path):
    try :
        with open(path, 'r') as file:
            for line in file:  # ← ici, on lit ligne par ligne
                if line.strip().startswith("scene.lights.sky.color="):
                    valeur_str = line.strip().split("=")[1]
                    return [ float(val) for val in valeur_str.split() ]
        return None  
    except :

        raise ValueError("--------- E_DIFFUS VALUES NOT FOUND. ---------")
    
def E_diffus_final_values(new,old,sm):
    
    l=[new[i] if sm[i]==0 else old[i] for i in range(len(new)) ]
    return l
    
def calculate_BOA_TOTAL_BOA(predictions, E_TOA):   
    return [p * e if p >= 0 else 0 for p, e in zip(predictions, E_TOA)]

def launch_ai(simulation_path):
    working_dir = Path(simulation_path)
    atmosphere_nc = working_dir / 'output' / 'atmosphere.nc'
    maket_scn= working_dir / 'output' / 'maket.scn'
    phase_scn=  working_dir / 'output' / 'phase.scn'
    simulation_properties = working_dir / 'output' / 'simulation.properties.txt'

    atmosphere_xml = find_files_xml(working_dir, 'atmosphere.xml')
    directions_path = find_files_xml(working_dir, 'directions.xml')
    maket_path = find_files_xml(working_dir, 'maket.xml')
    phase_path = find_files_xml(working_dir, 'phase.xml')

    DART_HOME = find_paths(working_dir,'DART')
    

    mode = get_RTMode(phase_path)
    if mode != 1:
        print("-------- AI MODEL WORKS ONLY WITH ANALYTICAL MODE -----------")
        return

    # Préparation des données
    scaler_y = joblib.load('scaler_y_SS_BOA_DART.pkl')
    SZA = get_SZA(directions_path)
    z = get_altitude(maket_path)
    reflectance = get_reflectance(maket_scn, phase_scn)
    atmosphere_lists = atmosphere_param(atmosphere_nc, atmosphere_xml)
    df = prepare_features(atmosphere_lists, SZA, z, reflectance)

    # Prédiction
    model = load_model('deep_model_v2.h5')
    prediction = model.predict(df)
    predictions_array = prediction.reshape(-1, 1)
    predictions_original_scale = scaler_y.inverse_transform(predictions_array)
    predictions_list = predictions_original_scale.flatten().tolist()

    # Calculs finaux
    E_TOA = EXTRACT_TOA(simulation_properties, SZA)
    E_BOA = calculate_BOA_TOTAL_BOA(predictions_list, E_TOA)
    E_direct = EXTRACT_E_direct(phase_scn, SZA)

    if len(E_BOA) == len(E_direct):
        E_diffus = [i - j for i, j in zip(E_BOA, E_direct)]
    else:
        E_diffus = [E_BOA[0] - i for i in E_direct]

    spectral_mode = get_spectral_mode(phase_path)
    E_diffus_old = EXTRACT_E_diffus(phase_scn)
    E_diffus = E_diffus_final_values(E_diffus, E_diffus_old, spectral_mode)

    edit_phase_scn(phase_scn, E_diffus)
    print("✅ AI processing finished successfully.")
