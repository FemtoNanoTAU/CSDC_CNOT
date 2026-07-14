import os
import numpy as np
import pandas as pd
import glob

gdrive_path = os.path.join(os.getcwd(), '..', '..')
dirname = os.path.join(os.getcwd(), 'cnot_quantum_results')

def get_single_index_from_logical_index(logical_index):
    # 0    1     2     3     4     5     6     7     8     9     10     11     12     13     14     15
    # step ustep rate1 rate2 rate3 rate4 dark1 dark2 dark3 dark4 rate12 rate13 rate14 rate23 rate24 rate34 dark12 dark13 dark14 dark23 dark24 dark34 rate123 rate124 rate134 rate234 
    if logical_index == 11:
        return 2, 4
    if logical_index == 12:
        return 2, 5
    if logical_index == 13:
        return 3, 4
    if logical_index == 14:
        return 3, 5

def get_rates(filenames, logical_indices, exclude_rate_list):
    for i, filename in enumerate(filenames):
        if i == 0:
            rates = np.loadtxt(filename, skiprows=6)
        else:
            rates = np.vstack((rates, np.loadtxt(filename, skiprows=6)))
    steps = [x[1] / 256 + x[0] for x in rates[:, 0:2]]
    unique_steps = np.unique(steps)
    rates_means = np.zeros((len(unique_steps), len(logical_indices)))
    rates_stds = np.zeros((len(unique_steps), len(logical_indices)))
    for s, step in enumerate(unique_steps):
        step_indices = np.where(steps == step)
        y_curr = [[] for _ in range(len(logical_indices))]
        for step_ind in step_indices[0]:
            exclude_sum = 0
            for exclude_rate in exclude_rate_list:
                exclude_sum += rates[step_ind, exclude_rate]
            if exclude_sum == 0:    
                for i, rate_index in enumerate(logical_indices):
                    # y_curr[i].append(rates[step_ind, rate_index])
                    sing1, sing2 = get_single_index_from_logical_index(rate_index)
                    y_curr[i].append(rates[step_ind, rate_index] / rates[step_ind, sing1] / rates[step_ind, sing2])
        for i, rate_index in enumerate(logical_indices):
            if len(y_curr[i]) == 0:
                rates_means[s, i] = 0
                rates_stds[s, i] = 0
            else:
                rates_means[s, i] = np.mean(y_curr[i])
                rates_stds[s, i] = np.std(y_curr[i])/np.sqrt(len(y_curr[i]))
    return unique_steps, rates_means, rates_stds

cells_uniform = [12, 15, 26, 31, 8] 
cells_composite = [13, 17, 20, 28, 37]
data = {'cell': [], 'type':[], 'C0T0': [], 'dC0T0': [], 'C0T1': [], 'dC0T1': [], 'C1T0': [], 'dC1T0': [], 'C1T1': [], 'dC1T1': []}
inputs = [('C0T0', 0, [11, 12], [10, 13, 14, 15]),
          ('C0T1', 1, [11, 12], [10, 13, 14, 15]),
          ('C1T0', 1, [13, 14], [10, 11, 12, 15]),
          ('C1T1', 0, [13, 14], [10, 11, 12, 15])]
for cell in cells_uniform + cells_composite:
    data['cell'].append(cell)
    if cell in cells_uniform:
        data['type'].append('Uniform')
    else:
        data['type'].append('Composite')
    for input_state, desired_index, logical_indices, exclude_rate_list in inputs:
        filenames = glob.glob(os.path.join(dirname, f"*CC4_CNOT{cell}_{input_state}_fine_scan.txt"))
        if len(filenames) == 0:
            print(f"No filenames found for cell {cell} and input state {input_state}")
            data[input_state].append(0)
            data[f'd{input_state}'].append(0)
            continue
        unique_steps, rates_means, rates_stds = get_rates(filenames, logical_indices, exclude_rate_list)
        probs =  rates_means[:, desired_index] / np.sum(rates_means, axis=1)
        dprobs = np.sqrt( \
            rates_stds[:, desired_index]**2 * \
            (np.sum(rates_means, axis=1) - rates_means[:, desired_index])**2 + \
            rates_means[:, desired_index]**2 * \
            (np.sum(rates_stds, axis=1) - rates_stds[:, desired_index])**2 \
        ) / np.sum(rates_means, axis=1)**2
        maxidx = np.argmax(probs)
        data[input_state].append(probs[maxidx])
        data[f'd{input_state}'].append(dprobs[maxidx])
cc4_cnots_df = pd.DataFrame(data)

cc4_cnots_df['Mean'] = (cc4_cnots_df['C0T0'] + \
                                cc4_cnots_df['C0T1'] + \
                                cc4_cnots_df['C1T1'] + \
                                cc4_cnots_df['C1T0'] ) / 4

cc4_cnots_df['dMean'] = 1/np.sqrt(1/cc4_cnots_df['dC0T0']**2 + \
                                                1/cc4_cnots_df['dC0T1']**2 + \
                                                1/cc4_cnots_df['dC1T0']**2 + \
                                                1/cc4_cnots_df['dC1T1']**2)

data = {'cell': [], 'type':[], 'C0T0': [], 'dC0T0': [], 'C0T1': [], 'dC0T1': [], 'C1T0': [], 'dC1T0': [], 'C1T1': [], 'dC1T1': [], 'Mean': [], 'dMean': []}
for type in ['Uniform', 'Composite']:
    data['cell'].append('Mean')
    data['type'].append(type)
    for input_state in ['C0T0', 'C0T1', 'C1T0', 'C1T1', 'Mean']:
        means = cc4_cnots_df.loc[cc4_cnots_df["type"]==type][input_state].to_numpy()
        stds = cc4_cnots_df.loc[cc4_cnots_df["type"]==type][f'd{input_state}'].to_numpy()
        dX2 = np.mean([mean**2 + std**2 for mean, std in zip(means, stds)])
        tot_mean = np.mean(means)
        tot_std = np.sqrt(np.abs(dX2 - np.mean(means)**2))
        data[input_state].append(tot_mean)
        data[f'd{input_state}'].append(tot_std)
        print(f'{type}: {input_state}: {tot_mean*100:.2f}% +- {tot_std*100:.2f}%')

cc4_cnots_df = pd.concat([cc4_cnots_df, pd.DataFrame(data)])
cc4_cnots_df.to_csv('cnot_cc4_quantum_results.csv', index=False)
print(cc4_cnots_df.head(15))