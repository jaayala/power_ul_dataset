import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import curve_fit



data_all = pd.read_csv("dataset_ul.csv") 

#We filter the CPU platform (NUC1), the type of experiment and the BW = 10 MHz
selected_cpu_platform = np.unique(data_all['cpu_platform'])[1]
type_of_experiment = 0
BW = np.unique(data_all['BW'])[2]
idx_bool = (data_all['cpu_platform'] == selected_cpu_platform) & (data_all['fixed_mcs_flag'] == type_of_experiment) & (data_all['BW'] == BW)
data = data_all[idx_bool]

power_measure_name = 'rapl_power'

a = np.array(data['selected_airtime'])
c = np.array(data['mean_snr'])
vals = np.array(data[power_measure_name])

def func(X, g0, g1, g2, g3, cth):
    a,c = X
    c = np.minimum(c, cth)
    return g0 + g1*a - (g2 + g3*a) * (cth - c)

p0 = 0, 0, 0, 0, 20
Z1, _ = curve_fit(func, (a,c), vals, p0)
print(Z1)


#PLOTTING
a_unique = np.unique(a)
a_unique = np.array([1, .8, .6, .4, .2])

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
lw_val = 4
scatter_size = 25
file_format = 'pdf'

plt.figure(0, figsize=(10,7.5))
for i in range(len(a_unique)):
    
    idx_bool = (a == a_unique[i])
    idx = np.where(idx_bool)[0]
    plt.scatter(c[idx], vals[idx], c=colors[i], s=scatter_size, alpha=0.4)
    
    X = (a_unique[i]*np.ones_like(idx), c[idx])
    plt.plot(c[idx], func(X, *Z1),  c=colors[i], label=r'$a$ = {}'.format( a_unique[i]), lw=lw_val)
plt.ylabel('Power (W)', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(loc='best',fontsize=23)

plt.savefig('./figure1.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')


# CONTOUR FIGURE
a = np.array(data['selected_airtime'])
a_unique = np.unique(a)
a_unique = np.round(a_unique, 2)
n_a = len(a_unique)

txgain = np.array(data['txgain'])
txgain_unique = np.unique(txgain)
n_txg = len(txgain_unique)

snr_vals = np.zeros((n_a, n_txg))


snr_ax = np.array(data.loc[data['selected_airtime'] == a_unique[0]]['mean_snr'])

thr_matrix = np.zeros((n_a, n_txg)) 
power_matrix = np.zeros((n_a, n_txg)) 

for i1 in range(len(a_unique)):
    for i2 in range(len(txgain_unique)):
        idx = (np.round(data['selected_airtime'], 2) == a_unique[i1]) & (data['txgain'] == txgain_unique[i2])
        failed_experiment = data.loc[idx]['failed_experiment'].item()
        thr_matrix[i1, i2] = data.loc[idx]['thr'] #if len(data.loc[idx]['thr']) > 0 and failed_experiment == 0 else np.nan
        power_matrix[i1, i2] = data.loc[idx][power_measure_name] #if len(data.loc[idx]['thr']) > 0 and failed_experiment == 0 else np.nan

N=30
cmap_val = 'viridis'

snr_ax = np.sort(snr_ax)
X, Y = np.meshgrid(snr_ax, a_unique)

fig = plt.figure(1, figsize=(10,7.5))
cax = plt.contourf(X, Y, thr_matrix / np.max(thr_matrix), N, cmap=cmap_val)
cax1 = plt.contour(X, Y, thr_matrix / np.max(thr_matrix), N, cmap=cmap_val)

plt.title('Throughput (%)', fontsize=28)
plt.ylabel('Airtime', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

cbar = fig.colorbar(cax, ticks=[0, .25, .5, .75, 1])
cbar.ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=24)

plt.annotate("",size=24, xy=(23.972, .9), xytext=(34.59, .9),  
             arrowprops=dict(arrowstyle='<->,head_length=0.18,head_width=0.2', facecolor='black',color='black',lw='2'))
plt.text(26, .85, 'Max. MCS', fontsize=20)
plt.savefig('./figure2.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
plt.show()

