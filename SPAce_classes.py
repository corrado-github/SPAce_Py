import numpy as np
import pandas as pd
import os, re, sys
import pdb
from astropy.modeling.models import Voigt1D
from astropy.convolution import convolve, Box1DKernel
from sklearn.neural_network import MLPRegressor

import SPace_data as SPdat
###########################################
def compute_gamma(ew, sigma, logg, teff, poly):
    x = np.array([ew, sigma, logg, teff], dtype=float)

    x_poly = poly.fit_transform([x])
    gamma = np.dot(x_poly[0],SPdat.gamL_coeff)

    return np.max([gamma,0.01])
###########################################
def compute_voigtD1_profile(wave, wave_c, fwhm, gamma, ew, logg, teff, poly):


    profile = Voigt1D(x_0=wave_c, fwhm_L=gamma*2, fwhm_G=fwhm)
    profile_arr = profile(wave)
    norm_area = np.dot(profile_arr[0:-1],np.diff(wave)).sum()
#    print('ew, area ' , ew, norm_area)
    return profile_arr*ew/norm_area

###########################################

class spectrum():

    def __init__(self):
        self.disp = None
        self.fwhm = None
        self.RV = None
        self.wave = None
        self.flux = None
        self.model_flux = None
        self.continuum = None
        self.norm_flux = None
        self.obs_spec_df = None
        self.sig_noise = None
        
    ####
    def load_obs_spec(self, file):
        self.obs_spec_df = pd.read_csv(file, delimiter='\s+',index_col=None, header=0, names=['wave', 'flux'])
        self.wave = self.obs_spec_df.wave.values
        self.flux = self.obs_spec_df.flux.values
    ####
    def initialize_disp_fwhm_RV(self, fwhm_ini, RV_ini):
        self.disp = np.diff(self.wave)
        self.fwhm = np.max([fwhm_ini, self.disp[0]*3]) # fwhm must be at least 3 pixels wide
        self.RV = RV_ini
    ####
    def select_user_interval_sp(self, wlranges_list):

        boole2drop_sp = np.ones(len(self.wave)).astype(bool)

        for wave1, wave2 in wlranges_list:
            wave_inf = np.max([self.wave[0], np.min([wave1, wave2])])
            wave_sup = np.min([self.wave[-1], np.max([wave1, wave2])])
            #set a boolean
            boole2keep_local_sp = (self.wave >= wave_inf) & (self.wave <= wave_sup)
        #combine the two booleans
        boole2drop_sp = boole2drop_sp & ~boole2keep_local_sp
        #apply them
        idx2drop_sp = self.obs_spec_df[boole2drop_sp].index
        #drop the rows not included in the user wavelength ranges
        self.obs_spec_df.drop(idx2drop_sp, inplace=True)
        self.wave = self.obs_spec_df.wave.values
        self.flux = self.obs_spec_df.flux.values

    ####
    def set_up_sig_noise(self, SN_sp_file):

        if SN_sp_file==True:
            SN_array = pd.read_csv(SN_sp_file, index_col=0)
            self.sig_noise = 1./SN_array.values
        else:
            self.sig_noise = np.ones(len(self.wave))/100. #we assume initially SN=100
    ####
    def update_sig_noise(self):

        wave_rej = SPdat.w_rej_op + SPdat.w_rej_nlte + SPdat.w_rej_unknown + SPdat.w_rej_bad
        rad_rej = SPdat.r_rej_op + SPdat.r_rej_nlte + SPdat.r_rej_unknown + SPdat.r_rej_bad

        for w_rej, r_rej in zip(wave_rej, rad_rej):
            boole = (np.abs(self.wave-w_rej) <= r_rej)
            self.sig_noise[boole] = 10.
    ####
    def make_model(self, llist, ML_models_dict, variables, scaler, poly):
        #    pdb.set_trace()
        pars_scaled = variables[0:3]

        pars_scaled = np.append(pars_scaled, 0.4) #add the El/M equal to zero, which correspond to 0.4 after scaling
        sigma = self.fwhm/2.35

        model_arr = np.zeros((len(ML_models_dict),len(self.wave)))


        pars = scaler.inverse_transform([pars_scaled])[0]
        print(pars[0:3], self.fwhm, self.RV)

        shift = 1.0+self.RV/299792.0 #doppler shift to apply to wavelength

        for i, (index, model) in enumerate(ML_models_dict.items()):

            wave_centre = llist.wavelength.loc[index]
            wave_rv_shifted = wave_centre * shift

            ew = model.predict([pars_scaled])[0]
            gamma = compute_gamma(ew/1000., sigma, pars[1], pars[0], poly)
            width = self.fwhm + gamma
            if ew<1.:
                continue
            pos_ini = np.argmin(np.abs(self.wave-(wave_centre-3*width)))
            pos_end = np.argmin(np.abs(self.wave-(wave_centre+3*width)))

            model_arr[i,pos_ini:pos_end] = compute_voigtD1_profile(self.wave[pos_ini:pos_end], wave_rv_shifted, self.fwhm, gamma, ew/1000., pars[1], pars[0], poly)

        self.model_flux = 1.0 - model_arr.sum(axis=0)

    ####
    def fit_continuum(self, norm_rad_pix):

        resid = self.flux - self.model_flux
        box_1D_kernel = Box1DKernel(norm_rad_pix*2)
        smooth_resid = convolve(resid, box_1D_kernel)
        self.continuum = 1.0 + smooth_resid    
    ####
    def normalize_obs_spec(self):
        self.norm_flux = np.divide(self.flux, self.continuum)
    ####