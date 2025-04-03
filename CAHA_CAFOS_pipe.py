# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:56:11 2024

@author: Gerard Garcia
"""

import os
from pathlib import Path
import calibration as calib

# Set directories as a global variable
# DATADIR = input('Input directory where your data are: ')
# directory = Path(DATADIR)
# PLOTDIR = os.path.join(DATADIR, "plots")

calib.init()

def general_calibrations():
    print('*************************')
    print('MASTER BIAS')
    print('*************************')
    calib.apply_master_bias(plot=True)
    print('Master BIAS applied')
    
    print('*************************')
    print('MASTER FLAT')
    print('*************************')
    calib.apply_master_flat(plot=True)
    print('Master FLAT applied')
    
    print('*************************')
    print('WAVELENGTH CALIBRATIONS')
    print('*************************')
    calib.wavelength_calibration(plot=True)

def science():
    print('*************************')
    print('SKY SUBTRACTION')
    print('*************************')
    calib.sky_substraction()
    
    print('*************************')
    print('ALIGNMENT')
    print('*************************')
    x_min, x_max, y_min, y_max = calib.spec_align()
    
    print('*************************')
    print('SPECTRUM EXTRACTION')
    print('*************************')
    calib.spec_extract(x_min, x_max, y_min, y_max)
    
    
def flux_calibration(std_flux_mags=False):
    print('*************************')
    print('FLUX CALIBRATION')
    print('*************************')
    calib.flux_calib(std_flux_mags=std_flux_mags)