import numpy as np
import pandas as pd
import os, re, sys
import pdb
import argparse

from scipy.stats import norm
from scipy.special import voigt_profile
from astropy.modeling.models import Voigt1D
import matplotlib.pyplot as plt
from SPace_data import gamL_coeff

from astropy.convolution import convolve, Box1DKernel

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy.optimize import leastsq,least_squares

from sklearn.utils import shuffle
import pickle
import time
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
def compute_gamma(ew, sigma, logg, teff, poly):
    x = np.array([ew, sigma, logg, teff], dtype=float)

    x_poly = poly.fit_transform([x])
    gamma = np.dot(x_poly[0],gamL_coeff)

    return np.max([gamma,0.01])

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
#def compute_voigt_profile(wave, sigma, ew, logg, teff):
#
#    gamma = compute_gamma(ew, sigma, logg, teff)
#    sigma_arr = np.ones(len(wave))*sigma
#    gamma_arr = np.ones(len(wave))*gamma
#    strength = voigt_profile(wave, sigma_arr, gamma_arr)*ew
#    return strength
###########################################
def compute_voigtD1_profile(wave, wave_c, fwhm, ew, logg, teff, poly):

    gamma = compute_gamma(ew, fwhm/2.35, logg, teff, poly)
    profile = Voigt1D(x_0=wave_c, fwhm_L=gamma*2, fwhm_G=fwhm)
    norm_area = np.dot(profile(wave[0:-1]),np.diff(wave)).sum()
#    print('ew, area ' , ew, norm_area)
    return profile(wave)*ew/norm_area
###########################################
def make_model(llist, ML_models_dict, wave_arr, variables, scaler, poly):

#    pdb.set_trace()
    pars_scaled = variables[0:3]

    pars_scaled = np.append(pars_scaled, 0.0) #add the El/M equal to zero
    fwhm = np.abs(variables[3])
    rv = variables[4]

    model_arr = np.zeros((len(ML_models_dict),len(wave_arr)))


    pars = scaler.inverse_transform([pars_scaled])[0]
    print(pars[0:3], fwhm, rv)

    shift = 1.0+rv/299792.0 #doppler shift to apply to wavelength

    for i, (index, model) in enumerate(ML_models_dict.items()):

        wave_centre = llist.wavelength.loc[index]
        wave_rv_shifted = wave_centre * shift
        ew = model.predict([pars_scaled])[0]
        if ew<1.:
            continue
        pos_ini = np.argmin(np.abs(wave_arr-(wave_centre-3*fwhm)))
        pos_end = np.argmin(np.abs(wave_arr-(wave_centre+3*fwhm)))

        model_arr[i,pos_ini:pos_end] = compute_voigtD1_profile(wave_arr[pos_ini:pos_end], wave_rv_shifted, fwhm, ew/1000., pars[1], pars[0], poly)

    return 1.0 - model_arr.sum(axis=0)
###########################################
def compute_residuals(variables, wave, flux_obs, llist, ML_models_dict,scaler, poly, norm_rad_pix):
    global continuum
    
    flux_model = make_model(llist, ML_models_dict, wave, variables, scaler, poly)
    continuum = fit_continuum(flux_obs, flux_model, norm_rad_pix)
    residuals = flux_model - np.divide(flux_obs, continuum)

    return residuals
#####################################
def fit_continuum(flux_obs, flux_model, norm_rad_pix):

    resid = flux_obs - flux_model
    box_1D_kernel = Box1DKernel(norm_rad_pix*2)
    smooth_resid = convolve(resid, box_1D_kernel)
    return 1.0 + smooth_resid
#####################################
def select_user_interval_ll(llist, wlranges_list):

    boole2drop = np.ones(len(llist)).astype(bool)

    for wave1, wave2 in wlranges_list:
        wave_inf = np.min([wave1, wave2])
        wave_sup = np.max([wave1, wave2])
        boole2keep_local = (llist.wavelength >= wave_inf) & (llist.wavelength <= wave_sup)
        boole2drop = boole2drop & ~boole2keep_local
    idx2drop = llist[boole2drop].index
    llist.drop(idx2drop, inplace=True)

    return llist
#####################################
def select_user_interval_spec(spec, wlranges_list):

    boole2drop = np.ones(len(spec)).astype(bool)

    for wave1, wave2 in wlranges_list:
        wave_inf = np.min([wave1, wave2])
        wave_sup = np.max([wave1, wave2])
        boole2keep_local = (spec.wave >= wave_inf) & (spec.wave <= wave_sup)
        boole2drop = boole2drop & ~boole2keep_local
    idx2drop = spec[boole2drop].index
    spec.drop(idx2drop, inplace=True)

    return spec.wave, spec.flux
###########################################
def compute_norm_rad_pix(wave_obs, norm_rad):

    #norm_rad must be larger than 5 angstrom and smaller than the width of the spectrum
    norm_rad = np.min([np.max([5., norm_rad]), len(wave_obs)])
    #compute the width in pixels
    norm_rad_pix = np.argmin(np.abs(wave_obs-wave_obs.iloc[0] - norm_rad))

    return norm_rad_pix
#############################################
def space_proc(infiles, GCOG_dir, wlranges_list, working_dir, fwhm, RV_ini, norm_rad):
    global continuum

    poly = PolynomialFeatures(degree=2)

    llist = pd.read_csv(GCOG_dir + 'linelist.csv', index_col=0)#, nrows=5000)
    #select the part of the llist chosen by the user
    llist = select_user_interval_ll(llist, wlranges_list)


    ML_models_dict = {}

    spec_file = infiles[0]
    spec_ = pd.read_csv(spec_file, delimiter='\s+',index_col=None, header=0, names=['wave', 'flux'])
    wave_obs, flux_obs = select_user_interval_spec(spec_, wlranges_list)

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
    #append fwhm and rv
    variables = np.append(variables[0:3], [fwhm, RV_ini])
    #define norm_rad in pixels
    norm_rad_pix = compute_norm_rad_pix(wave_obs, norm_rad)

    for idx in llist.index.tolist():
        model_name = idx + '_NN_model'
        ML_models_dict[idx] = pickle.load(open(GCOG_dir + model_name, 'rb'))

    #pars_scaled_obs = scaler.transform(np.array([[5000, 4.2, 0.0, 0.0]]))[0]
    #flux = make_model(llist, ML_models_dict, wave_obs, pars_scaled_obs)


    out = least_squares(compute_residuals, variables, args=(wave_obs, flux_obs, llist, ML_models_dict, scaler, poly, norm_rad_pix), method='lm', diff_step=0.01, gtol=0.1)
    
    flux_model = make_model(llist, ML_models_dict, wave_obs, out.x, scaler, poly)

#    smooth_resid = fit_continuum(flux_obs, flux_model)

    plt.plot(wave_obs, np.divide(flux_obs,continuum), color='green', linewidth=3, linestyle='dashed')
    plt.plot(wave_obs, flux_obs, color='black', linewidth=1, linestyle='dashed')
    plt.plot(wave_obs, flux_model, color='blue')
    plt.plot(wave_obs, continuum, color='red')
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


    space_proc(infiles, GCOG_dir, wlranges_list, working_dir, float(args.fwhm), float(args.RV_ini), float(args.norm_rad))
########################################################################
if __name__ == '__main__':

    current_dir = os.getcwd()
    work_dir = current_dir + '/work'

    blue_range = '5250.0,5300.0'
    red_range = '6300.0,6860.0'

    option= [
    '--infiles', '/home/corrado/workTux2/sp_test_space/elodie/R20/01964-SN100_N.asc', 
    '--wlranges', blue_range,
    '--working_dir', work_dir,
    '--GCOG_dir', '/home/corrado/workTux2/EW_library_SPACE2.2/libGCOG_ML/',
    '--RV_ini', '0.0',
    '--fwhm', '0.2',
    '--norm_rad', '10.0',
#    '--error_est', 'False',
    #the following options are not necessary and have default values
#    '--Salaris_MH', 'True',

#    '--ABD_loop', 'True',
#    '--SN_sp_file', 'True',

    ] 
    space_options(options=option)
#else:
#option=None
#space_options(options=option)