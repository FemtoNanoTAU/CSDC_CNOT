import scipy 
import pandas as pd
import numpy as np

quant_cnot = pd.read_csv('cnot_cc4_quantum_results.csv')
types = ['Uniform', 'Composite']
labels = []
fids_list = []
fids_stds = []
fids_means = []
for type in types:
    labels.append(f'{type}')
    means = quant_cnot[quant_cnot['type']==type][f'Mean'].to_numpy()
    fids = 1 - means
    fids_list.append(fids)
    stds = quant_cnot[quant_cnot['type']==type][f'dMean'].to_numpy()
    dX2 = np.mean([mean**2 + std**2 for mean, std in zip(means, stds)])
    fids_means.append(np.mean(means))
    fids_stds.append(np.sqrt(np.abs(dX2 - np.mean(means)**2)))
res = scipy.stats.ttest_ind(fids_list[0], fids_list[1], alternative='greater')
print(res)
print(res.confidence_interval())