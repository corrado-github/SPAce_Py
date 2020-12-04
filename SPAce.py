import numpy as np
import pandas as pd
import os, re, sys
import pdb
import argparse

from scipy.stats import norm
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy.optimize import leastsq,least_squares

from sklearn.utils import shuffle
import pickle
import time

import SPAce_classes as classes
###########################################################################
## To pass None keyword as command line argument
def none_or_str(value):
    if value == 'None':
        return None
    return value

###########################################
#def compute_gamma_1(ew, sigma, logg, teff):
#    x = np.array([1., ew, sigma, logg, teff], dtype=float)
#    transf = np.zeros(15)
#
#    count = -1
#    for i in range(5):
#        for j in np.arange(i,5):
#            count += 1
#            transf[count] = x[i]*x[j]
#
#    gamma = np.dot(transf,gamL_coeff)
#
#    return np.max([gamma,0.01])

###########################################
#def compute_gamma_label(ew, sigma, logg, teff):
#    x_label = ['1', 'ew', 'sigma', 'logg', 'teff']
#    x_label = ['1', 'x0', 'x1', 'x2', 'x3']
#    x_dict = {'1': '1', 'x0': 'ew', 'x1': 'sigma', 'x2': 'logg', 'x3': 'teff'}
#    x = np.array([ew, sigma, logg, teff], dtype=float)
#
#    transf = np.zeros(15)
#
#    count = -1
#    for i in range(5):
#        for j in np.arange(i,5):
#            count += 1
#            if i==j:
#                print('count=', count, ' coeff=', gamL_coeff[count], x_label[i] + '^2')
#            else:
#                print('count ', count, ' coeff=', gamL_coeff[count], x_label[i] + '*' + x_label[j])
#
#    poly = PolynomialFeatures(degree=2)
#    pdb.set_trace()
#    x_poly = poly.fit_transform([x])
#    list_names = poly.get_feature_names()
#
############################################
def compute_gauss_profile(wave_centre, wave, fwhm, ew):

    strength = norm.pdf(wave-wave_centre, scale=fwhm/2.35)*ew
    return strength[0]

###########################################
def compute_voigt_profile(wave, sigma, ew, logg, teff):

    gamma = compute_gamma(ew, sigma, logg, teff)
    sigma_arr = np.ones(len(wave))*sigma
    gamma_arr = np.ones(len(wave))*gamma
    strength = voigt_profile(wave, sigma_arr, gamma_arr)*ew
    return strength
###########################################
def compute_residuals(variables, spec_obj, llist, ML_models_dict,scaler, poly, norm_rad_pix):
    global continuum
    
    spec_obj.fwhm = variables[3]
    spec_obj.RV = variables[4]

    spec_obj.make_model(llist, ML_models_dict, variables, scaler, poly)
    spec_obj.fit_continuum(norm_rad_pix)

    residuals = np.divide(spec_obj.model_flux - spec_obj.flux, spec_obj.sig_noise)

    return residuals
#####################################

#####################################
def compute_SN(flux_obs, flux_model, sig_noise, rej_wave_bool):
    resid_abs = np.abs(flux_obs - flux_model)
    box_1D_kernel = Box1DKernel(50)
    sig_noise[~rej_wave_bool] = convolve(resid_abs[~rej_wave_bool], box_1D_kernel)

    return sig_noise
#####################################
def select_user_interval_ll(llist, spec_obj, wlranges_list):

    boole2drop_ll = np.ones(len(llist)).astype(bool)

    for wave1, wave2 in wlranges_list:
        wave_inf = np.max([spec_obj.wave[0], np.min([wave1, wave2])])
        wave_sup = np.min([spec_obj.wave[-1], np.max([wave1, wave2])])
        #set a boolean
        boole2keep_local_ll = (llist.wavelength > wave_inf) & (llist.wavelength < wave_sup)
        #combine the two booleans
        boole2drop_ll = boole2drop_ll & ~boole2keep_local_ll
    #apply them
    idx2drop_ll = llist[boole2drop_ll].index
    #drop the rows not included in the user wavelength ranges
    llist.drop(idx2drop_ll, inplace=True)

    return llist
#####################################
def compute_norm_rad_pix(wave, norm_rad):

    #norm_rad must be larger than 5 angstrom and smaller than the width of the spectrum
    norm_rad = np.min([np.max([5., norm_rad]), len(wave)])
    #compute the width in pixels
    norm_rad_pix = np.argmin(np.abs(wave-wave[0] - norm_rad))

    return norm_rad_pix
#############################################
#def set_up_sig_noise(SN_sp_file, wave):
#
#    if SN_sp_file==True:
#        SN_array = pd.read_csv(SN_sp_file, index_col=0)
#        sig_noise = 1./SN_array.values
#    else:
#        sig_noise = np.ones(len(wave))/100. #we assume initially SN=100
#    
#    return sig_noise
#############################################
#def update_sig_noise(wave, sig_noise):
#
#    wave_rej = SPdat.w_rej_op + SPdat.w_rej_nlte + SPdat.w_rej_unknown + SPdat.w_rej_bad
#    rad_rej = SPdat.r_rej_op + SPdat.r_rej_nlte + SPdat.r_rej_unknown + SPdat.r_rej_bad
#
#    for w_rej, r_rej in zip(wave_rej, rad_rej):
#        boole = (np.abs(wave-w_rej) <= r_rej)
#        sig_noise[boole] = 10.
#
#    return sig_noise
#############################################
def space_proc(infiles, GCOG_dir, wlranges_list, working_dir, fwhm_ini, RV_ini, norm_rad, SN_sp_file):
    global continuum

    poly = PolynomialFeatures(degree=2)

    llist = pd.read_csv(GCOG_dir + 'linelist.csv', index_col=0)#, nrows=5000)


    ML_models_dict = {}

    #spec_file = infiles[0]
    #spec_ = pd.read_csv(spec_file, delimiter='\s+',index_col=None, header=0, names=['wave', 'flux'])
    #instantiate the spectrum object
    spec_obj = classes.spectrum()
    #load the spectrum
    spec_obj.load_obs_spec(infiles[0])

    #select the part of the llist and spec chosen by the user
    llist = select_user_interval_ll(llist, spec_obj, wlranges_list)
    #select the part of the spec chosen by the user
    spec_obj.select_user_interval_sp(wlranges_list)


    #check if there is a SN file and set up a sig_noise array
#    sig_noise = set_up_sig_noise(SN_sp_file, wave_obs)
#    sig_noise = update_sig_noise(wave_obs, sig_noise)
    spec_obj.set_up_sig_noise(SN_sp_file)
    spec_obj.update_sig_noise()

#    flux_obs = 1. - spec_[(spec_.wave<4850) & (spec_.wave>4800)].flux.values
#    wave_obs = spec_[(spec_.wave<4850) & (spec_.wave>4800)].wave.values

    #wave = np.arange(4800.0-5., llist.wavelength.iloc[-1],0.3)
    #flux = np.zeros(len(wave))
    #spec_df = pd.DataFrame({'flux': flux}, index=wave)

    teff = 5000.
    logg = 3.0
    met = -0.3
    scaler = pickle.load(open(GCOG_dir + 'scaler_NN', 'rb'))
    variables = scaler.transform(np.array([[teff, logg, met, 0.0]]))[0]

#    disp = np.diff(wave_obs)
#    fwhm = np.max([fwhm, disp[0]*3]) # fwhm must be at least 3 pixels wide
    spec_obj.initialize_disp_fwhm_RV(fwhm_ini, RV_ini)
    #append fwhm and rv
    variables = np.append(variables[0:3], [fwhm_ini, RV_ini])
    #define norm_rad in pixels
    norm_rad_pix = compute_norm_rad_pix(spec_obj.wave, norm_rad)

    for idx in llist.index.tolist():
        model_name = idx + '_NN_model'
        ML_models_dict[idx] = pickle.load(open(GCOG_dir + model_name, 'rb'))

    #pars_scaled_obs = scaler.transform(np.array([[5000, 4.2, 0.0, 0.0]]))[0]
    #flux = make_model(llist, ML_models_dict, wave_obs, pars_scaled_obs)

#    out = least_squares(compute_residuals, variables, args=(wave_obs, flux_obs, sig_noise, llist, ML_models_dict, scaler, poly, norm_rad_pix), method='lm', diff_step=0.0001, ftol=0.1)
    out = least_squares(compute_residuals, variables, args=(spec_obj, llist, ML_models_dict, scaler, poly, norm_rad_pix), method='trf', xtol=0.01)
    
    spec_obj.make_model(llist, ML_models_dict, out.x, scaler, poly)

    plt.plot(spec_obj.wave, np.divide(spec_obj.flux,spec_obj.continuum), color='green', linewidth=3, linestyle='dashed')
    plt.plot(spec_obj.wave, spec_obj.flux, color='black', linewidth=1, linestyle='dashed')
    plt.plot(spec_obj.wave, spec_obj.model_flux, color='blue')
    plt.plot(spec_obj.wave, spec_obj.continuum, color='red')
    plt.plot(spec_obj.wave, spec_obj.sig_noise, color='violet')
    plt.show()

########################################################################
def space_options(options=None):

    parser = argparse.ArgumentParser()


    parser.add_argument("--infiles", type=none_or_str, default=None,
        required=True, help="input files", nargs='*')
    parser.add_argument("--wlranges", type=none_or_str, default=None,
        required=False, help="wavelenght range array for each elements of the setup", nargs='*')
    parser.add_argument("--working_dir", 
        help='Directory where SP_Ace can write its outputs', type=none_or_str, default=None, required=True)
    parser.add_argument('--GCOG_dir', 
        help='Directory of the GCOG library', type=none_or_str, default=None, required=True)
    parser.add_argument('--RV_ini', 
        help='Initial Radial Velocity', default=0.0, required=False)
    parser.add_argument('--fwhm', 
        help='FWHM (initial guess)', default=None, required=True)
    parser.add_argument('--norm_rad', 
        help='FWHM (initial guess)', default=30, required=False)
    parser.add_argument('--SN_sp_file', 
        help='Signal-to-Noise file as input', default=False, required=False)



    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.infiles is None:
        raise Exception('You need to specify the spectra you want to measure')
    infiles = args.infiles

    #working directory where SP_Ace write its outputs (that will be deleted after being read).
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
        print("WORKING DIRECTORY: %s Created!" %(args.working_dir))
    working_dir=args.working_dir+os.path.sep
    working_dir=working_dir.replace(' ','')


    #GCOG directory
    if not os.path.exists(args.GCOG_dir):
        raise Exception('The GCOG LIBRARY path seems wrong!')
    GCOG_dir=args.GCOG_dir+os.path.sep
    GCOG_dir=GCOG_dir.replace(' ','')

    wlranges = None
    wlranges_list = []
    if args.wlranges is not None:
        for item in args.wlranges:
            wlranges_list.append([float(x) for x in item.split(",")])


    space_proc(infiles, GCOG_dir, wlranges_list, working_dir, float(args.fwhm), float(args.RV_ini), float(args.norm_rad), args.SN_sp_file)
########################################################################
if __name__ == '__main__':

    current_dir = os.getcwd()
    work_dir = current_dir + '/work'

    blue_range = '5212.0,5712.0'
    red_range = '6300.0,6860.0'

    option= [
    '--infiles', '/home/corrado/workTux2/sp_test_space/elodie/R20/01964-SN100_N.asc', 
    '--wlranges', blue_range,
    '--working_dir', work_dir,
    '--GCOG_dir', '/home/corrado/workTux2/EW_library_SPACE2.2/libGCOG_ML/',
    '--RV_ini', '0.0',
    '--fwhm', '0.2',
    '--norm_rad', '10.0',
#    '--SN_sp_file', 'False',
#    '--error_est', 'False',
    #the following options are not necessary and have default values
#    '--Salaris_MH', 'True',

#    '--ABD_loop', 'True',


    ] 
    space_options(options=option)
#else:
#option=None
#space_options(options=option)