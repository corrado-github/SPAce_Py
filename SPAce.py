import numpy as np
import pandas as pd
import os, re, sys
import pdb
import argparse

from scipy.stats import norm
from scipy.special import voigt_profile
from astropy.modeling.models import Voigt1D
import matplotlib.pyplot as plt
from SPace_ML_data import gamL_coeff

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
#def make_model_1(llist, ML_models_dict, spec_df, pars_scaled):
#
#    model_df = spec_df.copy()
#    model_df.flux = 0.0
#
##    pdb.set_trace()
#    pars = scaler.inverse_transform([pars_scaled])[0]
#
#    for index, model in ML_models_dict.items():
#
#        wave_centre = llist.wavelength.loc[index]
#        ew = ML_models_dict[index].predict([pars_scaled])[0]
#        if ew<1.:
#            continue
#        boole_w = (np.abs(model_df.index - wave_centre) <= 3*fwhm)
#        wave_interval = model_df[boole_w].index - wave_centre
#        idx = model_df.index[boole_w]
#        model_df.loc[idx,'flux'] += compute_voigtD1_profile(model_df[boole_w].index, wave_centre, fwhm, ew/1000., pars[1], pars[0])
#
#    return model_df
###########################################
def make_model(llist, ML_models_dict, wave_arr, pars_scaled, scaler, fwhm, poly):

#    pdb.set_trace()
    model_arr = np.zeros((len(ML_models_dict),len(wave_arr)))


    pars = scaler.inverse_transform([pars_scaled])[0]

    for i, (index, model) in enumerate(ML_models_dict.items()):

        wave_centre = llist.wavelength.loc[index]
        ew = model.predict([pars_scaled])[0]
        if ew<1.:
            continue
        pos_ini = np.argmin(np.abs(wave_arr-(wave_centre-3*fwhm)))
        pos_end = np.argmin(np.abs(wave_arr-(wave_centre+3*fwhm)))

        model_arr[i,pos_ini:pos_end] = compute_voigtD1_profile(wave_arr[pos_ini:pos_end], wave_centre, fwhm, ew/1000., pars[1], pars[0], poly)

    return model_arr.sum(axis=0)
###########################################
def compute_residuals(pars, wave, spec_obs, llist, ML_models_dict,scaler, fwhm, poly):

    print(scaler.inverse_transform(np.array([pars])))
    model_df = make_model(llist, ML_models_dict, wave, pars, scaler, fwhm, poly)
#    residuals = model_df.flux.values - spec_df_obs.flux.values
    residuals = model_df - spec_obs
    return residuals
###########################################
def space_proc(infiles, GCOG_dir, wlranges, working_dir):

    poly = PolynomialFeatures(degree=2)

    llist = pd.read_csv(GCOG_dir + 'linelist.csv', index_col=0, nrows=500)

    ML_models_dict = {}

    spec_file = infiles[0]
    spec_ = pd.read_csv(spec_file, delimiter='\s+',index_col=None, header=0, names=['wave', 'flux'])
    spec_obs = 1. - spec_[(spec_.wave<4850) & (spec_.wave>4800)].flux.values
    wave = spec_[(spec_.wave<4850) & (spec_.wave>4800)].wave.values

    #wave = np.arange(4800.0-5., llist.wavelength.iloc[-1],0.3)
    #flux = np.zeros(len(wave))
    #spec_df = pd.DataFrame({'flux': flux}, index=wave)

    fwhm = 0.2
    scaler = pickle.load(open(GCOG_dir + 'scaler_NN', 'rb'))
    pars_scaled_obs = scaler.transform(np.array([[5000, 4.2, 0.0, 0.0]]))[0]
    pars_scaled_ini = scaler.transform(np.array([[5500, 4.0, -0.3, 0.1]]))[0]


    for idx in llist.index.tolist():
        model_name = idx + '_NN_model'
        ML_models_dict[idx] = pickle.load(open(GCOG_dir + model_name, 'rb'))

    #spec_obs = make_model(llist, ML_models_dict, wave, pars_scaled_obs)


    out = least_squares(compute_residuals, pars_scaled_ini, args=(wave, spec_obs, llist, ML_models_dict, scaler, fwhm, poly), method='lm', diff_step=0.01, xtol=0.1)
    
    spec_model = make_model(llist, ML_models_dict, wave, out.x, scaler, fwhm, poly)

    plt.plot(wave, spec_obs, color='green', linewidth=3, linestyle='dashed')
    plt.plot(wave, spec_model, color='blue')
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
        print("WORKING DIRECTORY: %s Created!" %(args.WORKING_DIR))
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


    space_proc(infiles, GCOG_dir, wlranges, working_dir)
########################################################################
if __name__ == '__main__':

    current_dir = os.getcwd()
    work_dir = current_dir + '/work'

    blue_range = '4900.0,5900.0'
    red_range = '6300.0,6860.0'

    option= [
    '--infiles', '/home/corrado/workTux2/sp_test_space/elodie/R20/01964-SN100_N.asc', 
    '--wlranges', red_range, blue_range,
    '--working_dir', work_dir,
    '--GCOG_dir', '/home/corrado/astro/space_ML_GCOG/libGCOG_ML/',
#    '--error_est', 'False',
    #the following options are not necessary and have default values
#    '--Salaris_MH', 'True',
#    '--RV_ini', 'True',
#    '--ABD_loop', 'True',
#    '--SN_sp_file', 'True',
#    '--norm_rad', '30.0',
    ] 
    space_options(options=option)
#else:
#option=None
#space_options(options=option)