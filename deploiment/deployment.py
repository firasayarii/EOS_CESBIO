# -*- coding: utf-8 -*- 

'''
Created on 18 sept. 2014
@author: guilleuxj

Atmosphere BOA with IA launcher from DART code
'''
from functions import *
import argparse

class launcher(object):

    def __init__(self, absPathToSimulation):
        self._absPathToSimulation = absPathToSimulation

    def run(self):
        print(" - Reading entry parameters: ")        
        print("   Current simulation: "+self._absPathToSimulation)        
        print(" - Currently not implemented, exiting ... ")  
        launch_ai(self._absPathToSimulation)     

if __name__ == '__main__':
    # Launcher, do not modify below this line
    parser = argparse.ArgumentParser()
    parser.add_argument("absPathToSimulation", help="Absolute path to DART simulation",nargs=1)
    args = parser.parse_args() 
    app = launcher(args.absPathToSimulation[0])
    app.run()