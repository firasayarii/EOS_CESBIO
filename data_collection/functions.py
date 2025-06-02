import numpy as np
import os
import shutil
from netCDF4 import Dataset
import pytools4dart as ptd
import math


def Edit_a_sca_albedo(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.atmosphere.set_nodes(aerosolAlbedo=new_value)
    simu.write(overwrite=True)

def Edit_a_opt_depth(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.atmosphere.set_nodes(aerosolOpticalDepth=new_value)
    simu.write(overwrite=True)

def Edit_Transmittance_gaz_scattering(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    a=-math.log(new_value)
    simu.core.atmosphere.set_nodes(gasOpticalDepth=a)
    simu.write(overwrite=True)

def Edit_Transmittance_gaz_absorption(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.atmosphere.set_nodes(transmittanceOfGases=new_value)
    simu.write(overwrite=True)



def Edit_g1(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.atmosphere.set_nodes(g1=new_value)
    simu.write(overwrite=True)



def Edit_zenith_angle(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.directions.set_nodes(sunViewingZenithAngle=new_value)
    simu.write(overwrite=True)


def Edit_reflectance(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.coeff_diff.set_nodes(reflectanceFactor=new_value)
    simu.write(overwrite=True)

def Edit_altitude(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.maket.set_nodes(altitude=new_value)
    simu.write(overwrite=True)

def Edit_lambda(new_value,n):
    simu=ptd.simulation(f'RTM_reference{n}')
    simu.core.phase.set_nodes(meanLambda=new_value)
    simu.write(overwrite=True)


def EXTRACT_BOA_RT(n):
    with open(f'/home/fayari/DART1425/user_data/simulations/RTM_reference{n}/output/dart.txt','r') as file :
        boa_values = []
        for line in file :
            if "BOA Total" in line:
                value = float(line.split("*")[-1].strip())  
                boa_values.append(value)
    return boa_values[-1]  




    
def calculate_alpha(thetha,z,Tg_abs,Tg_scat,optical_depth,albedo,n):   
    def EXTRACT_BOA_RT(n):
        with open(f'/home/fayari/DART1425/user_data/simulations/RTM_reference{n}/output/dart.txt','r') as file :
            boa_values = []
            for line in file :
                if "BOA Total" in line:
                    value = float(line.split("*")[-1].strip())  
                    boa_values.append(value)
        return boa_values[-1]  
    def muprime(z,h,µ):
        RAYON_TERRESTRE=6371
        eta = (RAYON_TERRESTRE*1000 + z) / h
        root = (eta*µ)**2  + 2 * eta + 1
        sum = (root)**0.5 - eta * µ
        if sum > 0 :
            return 1/sum
        return 1 
    BOA_RT=EXTRACT_BOA_RT(n)
    E_TOA=1000
    Ha=2000
    Hg=9000
    mu_a=(6371*1000+z)/Ha
    mu_g=(6371*1000+z)/Hg
    angle_rad = math.radians(thetha)
    µ=math.cos(angle_rad)  
    Y_a=muprime(z,Ha,µ)
    Y_g=muprime(z,Hg,µ)
    Ma=math.exp(-z/Ha)/Y_a
    Mg=math.exp(-z/Hg)/Y_g
    delta_a_scat=optical_depth*albedo
    delta_g_scat=-math.log(Tg_scat)
    Ta_abs=math.exp(-optical_depth*(1-albedo))
    numerator= (E_TOA/BOA_RT)*(Tg_abs**(Mg))*(Ta_abs**(Ma))-1
    denominator=(delta_g_scat * Mg) + (delta_a_scat * Ma) / 3
    alpha= numerator / denominator
    return alpha