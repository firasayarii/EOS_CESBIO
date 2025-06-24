import pytools4dart as ptd
import xml.etree.ElementTree as ET
from netCDF4 import Dataset
import math
import numpy as np
import re
from scipy.stats import truncnorm
import random 

def Edit_lambda(lamda_list,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.phase.set_nodes(meanLambda=lamda_list)
    sim.core.phase.set_nodes(deltaLambda=0)
    sim.write(overwrite=True)

def Edit_delta_lambda(delta_lambda,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.phase.set_nodes(deltaLambda=delta_lambda)
    sim.write(overwrite=True)
    
    
def Edit_reflectance(reflectance_list,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.coeff_diff.set_nodes(reflectanceFactor=reflectance_list)
    sim.write(overwrite=True)


def Edit_zenith_angle(new_value,batch_id):
    simu=ptd.simulation(f'RTM_reference{batch_id}')
    simu.core.directions.set_nodes(sunViewingZenithAngle=new_value)
    simu.write(overwrite=True)

def Edit_altitude(new_value,batch_id):
    simu=ptd.simulation(f'RTM_reference{batch_id}')
    simu.core.maket.set_nodes(altitude=new_value)
    simu.write(overwrite=True)

def Edit_AerosolFactor(new_value,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.atmosphere.set_nodes(mulFactorH2O=new_value)
    sim.write(overwrite=True)

def Edit_GasFactor(new_value,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.atmosphere.set_nodes(aerosolOptDepthFactor=new_value)
    sim.write(overwrite=True)  

def Edit_GasModel(new_value,batch_id):
    sim=ptd.simulation(f'RTM_reference{batch_id}')
    sim.core.atmosphere.set_nodes(gasModelName=new_value)
    sim.core.atmosphere.set_nodes(gasCumulativeModelName=new_value)
    sim.core.atmosphere.set_nodes(temperatureModelName=new_value)
    sim.write(overwrite=True)

def k_gaz_USDT():
    x = np.linspace(0, 2.1, 1000)

    weights = np.piecewise(
        x,
        [x < 0.2, x >= 0.2],
        [lambda x: x / 0.2, 1]
    )

    weights /= weights.sum()

    # Tirer UNE valeur
    value = np.random.choice(x, size=1, p=weights)[0]
    return value

def k_gaz_TROPICAL():
    x = np.linspace(0, 2.0, 1000)  

    weights = np.piecewise(
        x,
        [x < 1.37, 
         (x >= 1.38) & (x < 1.5), 
         (x >= 1.5) & (x <= 1.75),
         x > 1.75],
        [0,  # Avant 1.38, poids nul
         1,  # Plateau constant
         lambda x: (1.75 - x) / (1.75 - 1.5),  # Descente linéaire
         0]  # Après 1.75, poids nul
    )

    weights /= weights.sum()

    return np.random.choice(x, size=1, p=weights)[0]

def K_aerosol():
    x = np.linspace(0, 7, 1000)
    
    # Accentuer la montée vers 1 et la descente après 1
    weights = np.piecewise(
        x,
        [x <= 1, x > 1],
        [
            lambda x: (x / 1) ** 2,        # Monte plus vite vers 1
            lambda x: ((7 - x) / (7 - 1)) ** 2  # Descend plus vite après 1
        ]
    )

    weights = np.maximum(weights, 0)  # sécurité
    weights /= weights.sum()

    return np.random.choice(x, size=1, p=weights)[0]

def z_value():
    def a():
        scale = 720  # plus petit = plus concentré vers 0
        while True:
            z = np.random.exponential(scale)
            if z <= 6000:
                return int(z)
    z_values = [a() for _ in range(6000)]
    return random.choice(z_values)




    

def muprime(z,h,µ):
    RAYON_TERRESTRE=6371
    eta = (RAYON_TERRESTRE*1000 + z) / h
    root = (eta*µ)**2  + 2 * eta + 1
    sum = (root)**0.5 - eta * µ
    if sum > 0 :
        return 1/sum
    return 1 

def atmosphere_param(z,theta,batch_id):
    atmosphere_nc=f'/home/fayari/DART1425/user_data/simulations/RTM_reference{batch_id}/output/atmosphere.nc'
    atmosphere_xml=f'/home/fayari/DART1425/user_data/simulations/RTM_reference{batch_id}/input/atmosphere.xml'
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
                g1 = nc_file.groups['Henyey Greenstein']['g1'][i][0]
                l.append([tg_scat, tg_abs, optical_depth, albedo, g1,z,theta])
            return l
    else :

        with Dataset(atmosphere_nc, "r") as nc_file:
            GOD = nc_file.getncattr('Gas_optical_depth_scattering')
            tg_scat=math.exp(-GOD)
            tg_abs = nc_file.getncattr('Transmittance_of_gases_absorb')
            optical_depth = nc_file.getncattr('Total_aerosols_optical_depth')
            albedo = nc_file.getncattr('Aerosol_albedo')           
            g1 = nc_file.getncattr('Henyey_Greenstein_g1')
            l.append([tg_scat, tg_abs, optical_depth, albedo, g1,z,theta])
            return l
        

def EXTRACT_BOA_RT(batch_id,l):
    with open(f'/home/fayari/DART1425/user_data/simulations/RTM_reference{batch_id}/output/dart.txt', 'r') as file:
        boa_values = []
        for line in file:
            if "BOA Total" in line:
                # Extraire tous les nombres de la ligne
                values = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                boa_values.extend([float(v) for v in values])
    # Retourner les 100 dernières valeurs
    return boa_values[-l:]






'''def Edit_g1(new_value,n):
    simu=ptd.simulation('RTM_reference0')
    simu.core.atmosphere.set_nodes(g1=new_value)
    simu.write(overwrite=True)'''
