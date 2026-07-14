import numpy as np
import matplotlib.pyplot as plt

eps = {"Uniform": {"s_x": 0.022, "s_z": 0.125}, "Composite": {"s_x": 0.027, "s_z": 0.066}}
legnth_ums = {"X_half": {"Uniform": 10.03, "Composite": 10.32}, "X_two_thirds": {"Uniform": 13.24, "Composite": 13.60}}
num_of_points = 10000
types = ["Uniform", "Composite"]

def get_fidelity(u, v, length_um):
    return 0.25 * np.abs(np.trace(np.conj(u).T @ v))**2 * np.exp(-length_um*0.15 * 1e-4)

def get_gate(angle, s_x, s_z):
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    a = angle * np.sqrt((1+s_x)**2 + s_z**2)
    ns = angle / a * ((1+s_x)*X + s_z*Z)
    return np.cos(a)*I + 1j*np.sin(a) * ns

ideal_gates = {"X_half": np.pi/4,
               "X_two_thirds": np.arccos(1/np.sqrt(3))}


for gate in ideal_gates.keys():
    fids = {"Uniform": np.zeros(num_of_points), "Composite": np.zeros(num_of_points)}
    for type in types:
        s_xs = np.random.normal(loc=0, scale=eps[type]["s_x"], size=num_of_points)
        s_zs = np.random.normal(loc=0, scale=eps[type]["s_z"], size=num_of_points)
        for point in range(num_of_points):
            fids[type][point] = get_fidelity(get_gate(ideal_gates[gate], 0, 0), 
                                             get_gate(ideal_gates[gate], s_xs[point], s_zs[point]),
                                             legnth_ums[gate][type])
        print(f"Mean Fidelity for {gate} Gate of type {type}: {np.mean(fids[type])*100:.2f} % $\pm$ {np.std(fids[type])*100:.2f} %")
    plt.hist(fids["Uniform"], bins=100, alpha=0.5)
    plt.hist(fids["Composite"], bins=100, alpha=0.5)
    plt.xlabel("Fidelity")
    plt.ylabel("Count")
    plt.title(f"Fidelity Distribution for {gate} Gate")
    plt.legend()
    plt.show()