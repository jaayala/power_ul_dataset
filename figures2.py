import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 



data_all = pd.read_csv("dataset_ul.csv") 

#We filter the CPU platform (NUC1), the type of experiment and the BW = 10 MHz
selected_cpu_platform = np.unique(data_all['cpu_platform'])[1]
type_of_experiment = 1
BW = np.unique(data_all['BW'])[2]
idx_bool = (data_all['cpu_platform'] == selected_cpu_platform) & (data_all['fixed_mcs_flag'] == type_of_experiment) & (data_all['BW'] == BW)
data = data_all[idx_bool]

a = np.array(data['selected_airtime'])
c = np.array(data['mean_snr'])
m = np.array(data['selected_mcs'])

turbiters = np.array(data['turbodec_it'])
dectime = np.array(data['dec_time'])

failed = np.array(data['failed_experiment'])
idx = np.where(failed == 1)[0]
a = np.delete(a, idx)
c = np.delete(c, idx)
m = np.delete(m, idx)
turbiters = np.delete(turbiters, idx)
dectime = np.delete(dectime, idx)


file_format = 'pdf'
colors_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
lw_val = 4
scatter_size = 15

m_unique = np.array([0, 6, 9, 12, 15, 18, 23])

plt.figure(0, figsize=(10,7.5))
air = 1
for i in range(len(m_unique)):
    
    idx_bool = (m == m_unique[i]) & (a == air)
    idx = np.where(idx_bool)[0]
    idx_sorted_c = np.argsort(c[idx])
    sorted_c = c[idx]
    sorted_c = sorted_c[idx_sorted_c]
    sorted_turbiters = turbiters[idx]
    sorted_turbiters = sorted_turbiters[idx_sorted_c]
    
    plt.plot(sorted_c, sorted_turbiters, c=colors_vals[i], label=r'MCS = {}'.format(m_unique[i]), lw=lw_val)

   
plt.ylabel('Turbo decoder iterations', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(loc='best', fontsize=24)

plt.savefig('./figure3.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')


plt.figure(1, figsize=(10,7.5))
air = 1
for i in range(len(m_unique)):
    
    idx_bool = (m == m_unique[i]) & (a == air)
    idx = np.where(idx_bool)[0]
    
    idx_sorted_c = np.argsort(c[idx])
    sorted_c = c[idx]
    sorted_c = sorted_c[idx_sorted_c]
    sorted_dectime = dectime[idx]
    sorted_dectime = sorted_dectime[idx_sorted_c]
    
    plt.plot(sorted_c, sorted_dectime, c=colors_vals[i], label=r'MCS = {}'.format(m_unique[i]), lw=lw_val)

   
plt.ylabel('Decoding time (ms)', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(loc='best', fontsize=23)

plt.savefig('./figure4.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')

plt.show()
