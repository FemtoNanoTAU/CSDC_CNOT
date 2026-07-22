import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

with open('ideal_classical_unitary.pkl', 'rb') as f:
    ideal_classical_unitary = pickle.load(f)

def get_norm(Q, half_description):
    if half_description.startswith('H'):
        return np.linalg.norm(Q - ideal_classical_unitary['H'])
    else:
        return np.linalg.norm(Q - ideal_classical_unitary['X'])


def get_Q_at_wl_fit(Q, wl):
    Q_res = np.zeros(Q.shape[1:])
    for i in range(Q.shape[1]):
        for j in range(Q.shape[2]):
            p = np.polyfit(wls, Q[:, i, j], 2)
            Q_res[i, j] = np.polyval(p, wl)
    return Q_res

recon_matrices_df = pd.read_pickle('sinkhorn_reconstructed_matrices.pkl')
print(recon_matrices_df.head())
wls = np.load('sinkhorn_reconstruction_wls.npy')
Qs = recon_matrices_df['reconstructed_matrices'].to_list()
cell = 15
Q_1550 = get_Q_at_wl_fit(Qs[cell], 1550e-9)
i = 2
j = 4
p = np.polyfit(wls, Qs[cell][:, i, j], 2)
plt.plot(wls*1e9, Qs[cell][:, i, j], label='Measured Data')
plt.plot(wls*1e9, np.polyval(p, wls), '--k', label='Fitted Polynomial')
plt.plot(1550, Q_1550[i, j], 'rx', label='Interpolated Value @ degenerate wavelength')
plt.xlabel('Wavelength [nm]')
plt.ylabel(f'P_{i}{j} matrix element spectrum')
plt.legend()
plt.savefig(f'P_{i}{j}_spectrum.png')
plt.close()


norms = []
Qs_1550 = []
for idx, row in recon_matrices_df.iterrows():
    # Each row's 'reconstructed_matrices' is an ndarray (shape: [wavelengths, 6, 6])
    Q_wls = row['reconstructed_matrices']
    Q_1550 = get_Q_at_wl_fit(Q_wls, 1550e-9)
    norm = get_norm(Q_1550, row['half_description'])
    Qs_1550.append(Q_1550)
    norms.append(norm)


# Add the new column to the dataframe
recon_matrices_df['norm'] = norms
recon_matrices_df['Qs_1550'] = Qs_1550

for design in recon_matrices_df['two_thirds_description'].unique():
    print(design)
    norms = recon_matrices_df.loc[recon_matrices_df.two_thirds_description == design].norm.to_numpy()
    print(f'Mean: {norms.mean():.4f}. Std: {norms.std():.4f}')


stats_df = pd.DataFrame(columns=['Design', 'Mean', 'Std', 'Count'])
designs = [x for x in recon_matrices_df['two_thirds_description'].unique() if 'H' not in x]
designs = [designs[i] for i in [3, 4]]
labels = []
norms_list = []
norms_scatter = []
x_scatter = []
fig, ax = plt.subplots(figsize=(10, 6))

for i, design in enumerate(designs):
    des_name_words = design.split(' ')
    des_name = 'Uniform' if '2sg' in des_name_words else 'CSDC'
    norms = recon_matrices_df.loc[recon_matrices_df.two_thirds_description == design].norm.to_numpy()
    stats_df.loc[len(stats_df)] = [des_name, norms.mean(), norms.std(), norms.size]
    labels.append(des_name)
    norms_list.append(norms)
    for norm in norms:
        norms_scatter.append(norm)
        x_scatter.append([i+1])
ax.boxplot(norms_list, whis=10, tick_labels=labels)
ax.scatter(x_scatter, norms_scatter)
ax.set_xlabel('CNOT Design')
ax.set_ylabel('||P - P$_{ideal}$||')
ax.grid(axis='y')
plt.show()



for i, design in enumerate(designs):
    des_name_words = design.split(' ')
    des_name = 'Uniform' if '2sg' in des_name_words else 'CSDC'
    des_df = recon_matrices_df.loc[recon_matrices_df.two_thirds_description == design]
    norms = []
    for wl_idx, wl in enumerate(wls):
        wl_norms = []
        for idx, row in des_df.iterrows():
            Q = row['reconstructed_matrices'][wl_idx, :, :]
            norm = get_norm(Q, row['half_description'])
            wl_norms.append(norm)
        norms.append(np.mean(wl_norms))
    min_wl = wls[np.argmin(norms)]*1e9
    plt.plot(wls*1e9-min_wl, norms, label=des_name)
plt.xlabel('$\Delta \lambda$ [nm]')
plt.ylabel('||P - P$_{ideal}$||')
plt.legend()
plt.xlim(-5, 5)
plt.title(f'CNOT Classical Measurement Wavelength Sensitivity')
plt.show()