import numpy as np
import pandas as pd
import os, re, sys
import pdb
import matplotlib.pyplot as plt
#from multiprocessing import Pool

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import pickle
import time
########################################
def extract_pars(file_name):

    ll = file_name.replace('.ew_out','').split('-')
    Teff = int(ll[1].replace('t',''))
    logg = float(ll[2].replace('g',''))/10.
    if re.search('am', ll[0]):
        mh = -float(ll[0].replace('am',''))/10.
    else:
        mh = float(ll[0].replace('ap',''))/10.

    return Teff, logg, mh
########################################
path = '/home/corrado/workTux2/EW_library_SPACE2.2/libEWOP/'
dir_out = '/home/corrado/workTux2/EW_library_SPACE2.2/libGCOG_ML/'
ew_cols=['em04','em02','ep00','ep02','ep04','ep06']
enh_list = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
pars_cols = ['Teff','logg','MH','El/M']

#upload a file to collect the indexes
df = pd.read_csv(path + 'am14-t4600-g54.ew_out', index_col=0)
#set the dataframe that will contain the abundances discrepancies
df_abd_diff = pd.DataFrame(index=df.index, columns=['EW_00_05_max','EW_00_05_med','EW_05_20_max','EW_05_20_med','EW_20_50_max','EW_20_50_med','EW_50_max','EW_50_med'])

#instantiate the scaler
scaler = MinMaxScaler()

#to avoid to upload the whole EW library, let's upload it for few lines per time 
# this will set the start and stop rows to be read from the EW library files
start_list = []
step = 50
i = -step


while True:
    i = i + step + 1
    start_list.append(i)
    if i+step >= len(df):
        break

start_time = time.time()
#now that we have the start_stop_list, read few lines per file for each EW file
#and put it into a dataframe
for ini_row in start_list[0:20]:

    pars_list = []
    idx_list = []

    for file_name in os.listdir(path):

        df_ew = pd.read_csv(path + file_name, index_col=0, names=df.columns, skiprows=ini_row, nrows=step+1)
        Teff, logg, mh = extract_pars(file_name)

        for idx in df_ew.index:
            for i,val in enumerate(enh_list):
                pars_list.append([Teff, logg, mh, val, df_ew.loc[idx, ew_cols[i]]])
                idx_list.append(idx)


    #transform into df
    df_TGM_ew = pd.DataFrame(pars_list, index=idx_list, columns=['Teff','logg','MH','El/M','EW'])

    #define X_train that is the same for every line
    X_train = df_TGM_ew.loc[df_ew.index[0], pars_cols]
    #scale the values
    X_train_scaled = scaler.fit_transform(X_train)
    pickle.dump(scaler, open('scaler_NN', 'wb'))

    #the df_TGM_ew dataframe contains the EW of few lines for the whole stellar parameter space
    #compute the model for each line
    for idx in df_ew.index:
        print('process ', idx)

        y_train = df_TGM_ew.loc[idx, 'EW'] + 0.01 #add 0.01 to avoid that some EW=0.0

        print('compute clf for idx ', idx)
        clf = MLPRegressor(hidden_layer_sizes=(10,10),alpha=0.0001, solver='sgd',activation='logistic',random_state=1, max_iter=10000, learning_rate='adaptive').fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_train_scaled)
        #clip to small positive EWs
        y_pred[y_pred<0.01] = 0.01
        #compute the discrepancy as abd
        abd_diff = np.abs(np.log10((y_pred-y_train)/y_train + 1.))
        #select different sets of EWs to compute the max abd_diff
        boole_05 = (y_train > 0.) & (y_train <= 5.)
        boole_20 = (y_train > 5.) & (y_train <= 20.)
        boole_50 = (y_train > 20.) & (y_train <= 50.)
        boole_ = (y_train > 50.)
        list_max_med = []
        for boo in [boole_05, boole_20, boole_50, boole_]:
            if boo.sum()>0:
                list_max_med.append(np.max(abd_diff[boo]))
                list_max_med.append(np.median(abd_diff[boo]))
            else:
                list_max_med.append(np.nan)
                list_max_med.append(np.nan)

        df_abd_diff.loc[idx] = list_max_med

        #save the model
        name_model = idx + '_NN_model'
        print('write ', name_model)
#        pickle.dump(clf, open(dir_out + name_model, 'wb'))


#save the df_abd_diff
df_abd_diff.to_csv(dir_out + 'abd_diff.csv')
print('tot time ', (time.time() - start_time)/60.)
