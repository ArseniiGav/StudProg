import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from scipy import stats

ifname = input('Input file name: ', )#"processed_0.hdf5"

x_c_array = []
y_c_array = []
z_c_array = []
allHits_array = []
#minTime_array = []
Edep_array = []
hitTimeLPMT_array = []

f_in = h5py.File(ifname, "r")

shape = len(f_in['lpmt_hits']['hitTime'])
evtIDs = [i for i in range(0, shape)]

for evtID in tqdm(evtIDs):
    
    lpmt_hits = f_in['lpmt_hits']
    lpmt_pos = f_in['lpmt_pos']

    spmt_hits = f_in['spmt_hits']
    spmt_pos = f_in['spmt_pos']
    
    lpmt_x, lpmt_y, lpmt_z = [],[],[]

    for pos in np.array(lpmt_pos):
        lpmt_x.append(pos[1])
        lpmt_y.append(pos[2])
        lpmt_z.append(pos[3])

    lpmt_x = np.array(lpmt_x)/1000.
    lpmt_y = np.array(lpmt_y)/1000.
    lpmt_z = np.array(lpmt_z)/1000.
    
    lpmt_s = np.zeros(len(lpmt_x))
    for pmtID in lpmt_hits['pmtID'][evtID]:
        lpmt_s[pmtID] += 1
        
    lpmt_x = lpmt_x[lpmt_s>0]
    lpmt_y = lpmt_y[lpmt_s>0]
    lpmt_z = lpmt_z[lpmt_s>0]
    lpmt_s = lpmt_s[lpmt_s>0]
    
    allHits = sum(lpmt_s)
    
    x_c = sum(lpmt_x*lpmt_s)/allHits
    y_c = sum(lpmt_y*lpmt_s)/allHits
    z_c = sum(lpmt_z*lpmt_s)/allHits
    
#    minTime = f_in['lpmt_hits']['hitTime'][evtID].min()
    
    hitTimeLPMT = f_in['lpmt_hits']['hitTime'][evtID]
    hitTimeLPMT_array.append(hitTimeLPMT)
    
    x_c_array.append(x_c)
    y_c_array.append(y_c)
    z_c_array.append(z_c)
    
 #   minTime_array.append(minTime)
    allHits_array.append(allHits)
    Edep_array.append(f_in['true_info']['edep'][evtID])
    
f_in.close()

from scipy.optimize import curve_fit

def func(x, A, beta, C):
    return A*np.exp(-beta*x) + C

def approximated(x, y):
    popt, _ = curve_fit(func, x, y, maxfev=10**6)    
    A, beta, C = popt
    return func(x, A, beta, C), popt

params = []
mean = [hitTimeLPMT_array[i].mean() for i in range(len(hitTimeLPMT_array))]

for i in  tqdm(range(len(evtIDs))): 
    qq = stats.probplot(hitTimeLPMT_array[i], dist='expon', rvalue=True)
    rvalue = qq[1][2]
    x = np.array([qq[0][0][0],qq[0][0][-1]])
    
    params.append(approximated(qq[0][0], qq[0][1])[1])

data = pd.DataFrame({
                     'x_c_lpmt': x_c_array,
                     'y_c_lpmt': y_c_array,
                     'z_c_lpmt': z_c_array,
                     'A' : np.array(params).T[0],
                     'beta' : np.array(params).T[1],
                     'C' : np.array(params).T[2],
                     'mean' : mean,
 #                    'minTime_lpmt': minTime_array,
                     'allHits_lpmt': allHits_array,
                     'Edep': Edep_array
                    }
)

data.to_csv('ProcessedData.csv', index=False)