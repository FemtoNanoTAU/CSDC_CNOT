import numpy as np
import pandas as pd
import json

def getB(A, P):
    B = 1 / np.matvec(P.transpose(0, 2, 1), A)
    return B
     
def getA(P, B):
    return 1 / np.matvec(P, B)

def my_fixed_point_solver(A0, P, max_n=2**13, tol=np.finfo(float).eps, pr=False):
    Ac = np.copy(A0) # current
    Ap = np.copy(A0) # previous
    for n in range(max_n): # iterate until fixed point
        Ac = getA(P, getB(Ap, P))
        Ac = np.abs(Ac) / np.linalg.norm(Ac)
        if np.linalg.norm(Ap - Ac) < tol: # converged
            if pr:
                print(f"converged after {n} iterations, with error {np.linalg.norm(Ap - Ac)}")
            return Ac
        Ap = np.copy(Ac)
    return Ac

def solve_measurements(P):
    A = 10 * np.random.randn(*P.shape[:-1])
    A = my_fixed_point_solver(A, P)
    B = getB(A, P)
    dA = np.zeros(P.shape)
    dB = np.zeros(P.shape)
    for i in range(P.shape[0]):
        dA[i, :, :] = np.diag(A[i, :])
        dB[i, :, :] = np.diag(B[i, :])
    Q = dA @ P @ dB
    return Q, A, B


meas_df = pd.read_pickle('cnot_classical_measurements.pkl')
print(f'meas_df: {meas_df.head()}')

with open('coupler_characterization4_27-08-24.json', 'r') as f:
    coupler_char_data = json.load(f)
with open('coupler_characterization4_cnots_witheulers_27-08-24.json') as f:
    cnot_char_data = json.load(f)

port_map = [{"in": 5, "out": 0}, # A0
            {"in": 4, "out": 1}, # C1
            {"in": 3, "out": 2}, # C0
            {"in": 2, "out": 3}, # T0
            {"in": 1, "out": 4}, # T1
            {"in": 0, "out": 5}] # A1

reconstructed_matrices = []
etas_A = []
etas_B = []
half_descriptions = []
two_thirds_descriptions = []
cnot_families = []
types = []
cells = []

for cell in meas_df['cell'].unique():
    cnot_circuit = cnot_char_data[cell-1]
    half_gate = cnot_circuit[1][0]['Rails']['2']
    two_thirds_gate = cnot_circuit[1][1]['Rails']['2']
    found_coupler_half = None
    found_coupler_two_thirds = None
    for coupler_json in coupler_char_data:
        coupler = coupler_json[1][0]['Rails']['2']
        coupler_description = coupler_json[1][0].get('description', '')
        if coupler == half_gate:
            found_coupler_half = coupler_description
        if coupler == two_thirds_gate:
            found_coupler_two_thirds = coupler_description
    if found_coupler_half is None or found_coupler_two_thirds is None:
        print(f'Could not find couplers for cell {cell}')
        continue

    half_descriptions.append(found_coupler_half)
    two_thirds_descriptions.append(found_coupler_two_thirds)
    cells.append(cell)
    if two_thirds_descriptions[-1].startswith('H'):
        cnot_families.append('H')
    elif two_thirds_descriptions[-1].startswith('X'):
        cnot_families.append('X')
    else:
        print('Not an H and not an X? Who Am I??')
    if '4sg' in two_thirds_descriptions[-1]:
        types.append('Composite')
    else:
        types.append('Uniform')
    
    cell_df = meas_df.loc[meas_df.cell == cell]
    wls = np.array(cell_df.iloc[0].wl)
    # idx_1550 = np.argmin(np.abs(wls - 1550e-9))
    P = np.zeros((wls.shape[0], 6, 6))    
    for i in range(6):
        for j in range(6):
            outport = port_map[i]['in']
            inport =  port_map[j]['out']
            element_df = cell_df.loc[cell_df.inport == inport].loc[cell_df.outport == outport]            
            Ilambs = np.array(element_df.iloc[0].Ilamb)
            P[:, i, j] = Ilambs
    Q, A, B = solve_measurements(P)
    reconstructed_matrices.append(Q)
    etas_A.append(A)
    etas_B.append(B)

print(wls)
print(wls.shape)
pd.DataFrame({'cells': cells, 'reconstructed_matrices': reconstructed_matrices,'etas_A': etas_A, 'etas_B': etas_B, 'types': types,
              'half_description': half_descriptions, 'two_thirds_description': two_thirds_descriptions, 'cnot_families': cnot_families}).to_pickle('sinkhorn_reconstructed_matrices.pkl')
np.save('sinkhorn_reconstruction_wls.npy', wls)