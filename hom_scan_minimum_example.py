import scipy
import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian(x, amp, mean, stddev, const):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + const

def get_minimum_fit_from_hom(filename, rate_index, exclude_rate_list, plot=False):
    rates = np.loadtxt(filename, skiprows=6)
    steps = [x[1] / 256 + x[0] for x in rates[:, 0:2]]
    unique_steps = np.unique(steps)
    x_data = unique_steps
    y_data = []
    for s in unique_steps:
        indices = np.where(steps == s)
        y_curr = []
        for ind in indices[0]:
            exclude_sum = 0
            for exclude_rate in exclude_rate_list:
                exclude_sum += rates[ind, exclude_rate]
            if exclude_sum == 0:
                y_curr.append(rates[ind, rate_index])
        if len(y_curr) == 0:
            y_data.append(0)
            print("?")
        else:
            y_data.append(np.mean(y_curr))
    popt = [-2.5, 51650, 10, 2.5]
    popt, _ = scipy.optimize.curve_fit(gaussian, x_data, y_data, p0=popt)
    print("Optimal parameters (amp, mean, stddev, const):", popt)
    if plot:    
        plt.scatter(x_data, y_data, label='Data')
        x_data = np.linspace(min(x_data), max(x_data), 100)
        plt.plot(x_data, gaussian(x_data, *popt), color='red', label='Fit')
        plt.ylabel('Coincidence')
        plt.xlabel('Stage location [$\mu m$]')
        plt.plot(popt[1], gaussian(popt[1], *popt), 'xk', markersize=10)
        plt.legend()
        print(np.abs(popt[0])/ np.abs(popt[-1]))
        # plt.show()
    return popt[1]

dirname = os.path.join(os.getcwd(), 'cnot_quantum_results')
filename = 'Coincidence_Matrix_log_17-03-2025_08-53_CC4_CNOT12_C0T1_coarse_scan.txt'
# filename = 'Coincidence_Matrix_log_19-03-2025_11-07_CC4_CNOT37_C0T1_coarse_scan.txt'

min_loc = get_minimum_fit_from_hom(os.path.join(dirname, filename), rate_index=11, exclude_rate_list=[10, 13, 14, 15], plot=True)
plt.savefig('hom_scan_minimum_example.png')
plt.show()
print(min_loc)
# plt.xlim([51645, 51655])