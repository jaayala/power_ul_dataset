import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import curve_fit
from matplotlib import colors


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

vals = np.array(data['rapl_power'])

failed = np.array(data['failed_experiment'])
idx = np.where(failed == 1)[0]
a = np.delete(a, idx)
c = np.delete(c, idx)
m = np.delete(m, idx)
vals = np.delete(vals, idx)

a_unique = np.unique(a)
m_unique = np.unique(m)

total_points = len(a_unique) * len(m_unique)
a1 = []
m1 = []
p1 = []

for i in range(len(a_unique)):
    for j in range(len(m_unique)):

        idx_bool = (a == a_unique[i]) & (m == m_unique[j]) & (c > 22)
        idx = np.where(idx_bool)[0]
        if len(idx) == 0:
            continue
        a1.append(a_unique[i])
        m1.append(m_unique[j])

        p1.append(np.mean(vals[idx]))

a1 = np.array(a1)
m1 = np.array(m1)
p1 = np.array(p1)

def func_const(X, g1, g2, g3, g4, g5, g6):
    a,m = X
    return g1 + g2*a + g3*m + g4*a**2 + g5*m**2 + g6*a*m


G1, _ = curve_fit(func_const, (a1,m1), p1, method='trf', maxfev=50000)
print(G1)


def func_linear(X, b0, b1, b2, b3, b4):
    a,c,m = X 
    P0 = func_const((a,m), *G1)  
    cth = b0 + b1*m
    m1 = b2 + b3*a + b4*m    
    c = np.minimum(c, cth)    
    P = P0 + m1*(cth-c)
    return P
    
G2, _ = curve_fit(func_linear, (a,c,m), vals, method='trf', maxfev=50000)
print(G2)




def func_model1(X, g0, g1, g2, g3, cth):
    a,c = X
    c = np.minimum(c, cth)
    return g0 + g1*a + (g2 + g3*a)*c

# The vector Z1 is obtained in the file figures1.py
Z1 = np.array([ 3.35146707e+00,  5.23408651e-01, -1.26247435e-03,  4.19028413e-02,  2.74738439e+01])

#PLOTTING 1
file_format = 'pdf'
colors_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
lw_val = 4
scatter_size = 15

m_unique = np.array([0, 6, 12, 18, 23])

plt.figure(1, figsize=(10,7.5))
air = 1
for i in range(len(m_unique)):
    
    idx_bool = (m == m_unique[i]) & (a == air)
    idx = np.where(idx_bool)[0]
    plt.scatter(c[idx], vals[idx], c=colors_vals[i], s=scatter_size, alpha=0.3)
    
    X = (air*np.ones_like(c[idx]), c[idx], m_unique[i]*np.ones_like(c[idx]))
    P = func_linear(X, *G2)
    
    plt.plot(c[idx],P, c=colors_vals[i], label=r'$m$ = {}'.format(m_unique[i]), lw=lw_val)

idx_bool = (m == m_unique[0]) & (a == air)
idx = np.where(idx_bool)[0]
X = (air * np.ones_like(idx), c[idx])
P_m1 = func_model1(X, *Z1)
plt.plot(c[idx],P_m1, c='k',  lw=lw_val, label='Default\nscheduler')   
    
plt.ylabel('Power (W)', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

_, top = plt.ylim()  # return the current ylim
plt.ylim(top=(top+0))   # set the ylim to bottom, top

plt.legend(loc='best', ncol=3,fontsize=20)

plt.annotate("",size=24, xy=(18, 4.6), xytext=(18, 5.117),  arrowprops=dict(arrowstyle='<->,head_length=0.18,head_width=0.2', facecolor='black',color='black',lw='2'))
plt.annotate("Increase of 11.2%",size=20, xy=(18, 4.855), xytext=(5, 5.15),  arrowprops=dict(arrowstyle='<-', facecolor='black',color='black',lw='2', connectionstyle='angle3,angleA=90,angleB=0'))
plt.savefig('./figure5.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')



#PLOTTING 2
mcs_unique = np.unique(data['selected_mcs'])
n_mcs = len(mcs_unique)
txgain_unique = np.unique(data['txgain'])
n_txgain = len(txgain_unique)

snr_vals = np.zeros((n_mcs, n_txgain))

for i in range(n_mcs):
    idx = (data['selected_mcs'] == mcs_unique[i]) & (data['selected_airtime'] == air)
    snr_vals[i,:] = data.loc[idx]['mean_snr']
    failed_experiment = np.array(data.loc[idx]['failed_experiment'])
    snr_vals[i,failed_experiment == 1] = np.nan
    

snr_ax = np.nanmean(snr_vals, axis=0)
txgain_ax = txgain_unique
mcs_ax = mcs_unique

thr_matrix = np.zeros((len(mcs_ax), len(txgain_ax))) 
power_matrix = np.zeros((len(mcs_ax), len(txgain_ax))) 



for i1 in range(len(mcs_ax)):
    for i2 in range(len(txgain_ax)):

        idx = (data['selected_mcs'] == mcs_ax[i1]) & (data['txgain'] == txgain_ax[i2]) & (data['selected_airtime'] == air)
        failed_experiment = data.loc[idx]['failed_experiment'].item()
        thr_matrix[i1, i2] = data.loc[idx]['thr'] if failed_experiment == 0 else 0
        power_matrix[i1, i2] = data.loc[idx]['rapl_power'] if failed_experiment == 0 else 0
        
    
type_of_experiment_2 = 0
idx_bool = (data_all['cpu_platform'] == selected_cpu_platform) & (data_all['fixed_mcs_flag'] == type_of_experiment_2) & (data_all['BW'] == BW)
data2 = data_all[idx_bool]
    
txgain_unique2 = np.unique(data2['txgain'])
n_txgain2 = len(txgain_unique)

snr_ax2 = np.zeros(n_txgain2)
thr2 = np.zeros(n_txgain2)
power2 = np.zeros(n_txgain2)
used_mcs2 = np.zeros(n_txgain2)

for i in range(n_txgain2):
    idx = np.logical_and(data2['selected_airtime'] == 1. , data2['txgain'] == txgain_ax[i])
    thr2[i] = data2.loc[idx]['thr'] 
    power2[i] = data2.loc[idx]['rapl_power'] 
    snr_ax2[i] = data2.loc[idx]['mean_snr'] 
    used_mcs2[i] = data2.loc[idx]['mean_used_mcs'] 


N=20
cmap_val = 'viridis'
snr_ax = np.sort(snr_ax)
X, Y = np.meshgrid(snr_ax, mcs_ax)

fig = plt.figure(3, figsize=(10,7.5))
cax = plt.contourf(X, Y, thr_matrix / np.nanmax(thr_matrix), N, cmap=cmap_val)
cax1 = plt.contour(X, Y, thr_matrix / np.nanmax(thr_matrix), N, cmap=cmap_val)
plt.title('Throughput (%)', fontsize=28)
plt.ylabel('MCS', fontsize=28)
plt.xlabel('SNR', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(right=34.5)
plt.plot(snr_ax2, used_mcs2, lw=lw_val, c='saddlebrown', label='Default Scheduler')
plt.legend(loc="best", fontsize=24)
cbar = fig.colorbar(cax, ticks=[0, .25, .5, .75, 1])
cbar.ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=20)

plt.annotate("",size=24, xy=(18, 15.657), xytext=(18, 23),  arrowprops=dict(arrowstyle='<->,head_length=0.18,head_width=0.2', facecolor='k',color='k',lw='2'))
plt.text(12, 18, 'Increase\nof 76.1%', fontsize=20)

plt.savefig('./figure6.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')




# PLOTTING 3 

colors_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
ls_val = ['-', '--', ':', '-.']
lw_val = 4

m_unique = np.array([23, 18])
air_unique = np.array([0.8, 0.6, 0.2])

plt.figure(4, figsize=(10,7.5))
plist = []
air = 1
for i in range(len(m_unique)):
    for j in range(len(air_unique)):    
        idx_bool = (m == m_unique[i]) & (a == air_unique[j])
        idx = np.where(idx_bool)[0]
        plt.scatter(c[idx], vals[idx], c=colors_vals[i], s=scatter_size, alpha=0.3)
        
        sorted_c = np.sort(c[idx])[::-1]
        
        if i == 0: # plot only the range of interest
            idx_del = np.where(sorted_c < 14.3723)[0]
            sorted_c = np.delete(sorted_c, idx_del)

        X = (air_unique[j]*np.ones_like(sorted_c), sorted_c, m_unique[i]*np.ones_like(sorted_c))
        P = func_linear(X, *G2)

        p, = plt.plot(sorted_c, P, c=colors_vals[i], ls=ls_val[j], label=r'$m$ = {}, $a$ = {}'.format(m_unique[i], air_unique[j]), lw=lw_val)
        plist.append(p)



plt.legend([plist[0],plist[1],plist[2],plist[0],plist[3]], [r"$a$ = 0.8", r"$a$ = 0.6", r"$a$ = 0.2", r"$m$ = 23", r"$m$ = 18"], loc='best', ncol=2,fontsize=24)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('k')
leg.legendHandles[2].set_color('k')


x1, x2 = plt.xlim()
plt.xlim(10, x2)

#Markers
XX1 = (0.6, 25, 23)
XX2 = (0.8, 25, 18)
YY1 = func_linear(XX1, *G2)
YY2 = func_linear(XX2, *G2)
ms = 15
plt.plot(25, YY1, marker='o', c='maroon', markersize=ms)
plt.plot(25, YY2, marker='s', c='maroon', markersize=ms)

plt.ylabel('Power (W)', fontsize=28)
plt.xlabel('SNR (dB)', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

plt.savefig('./figure7.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')



# PLOTTING 4

a_unique =  np.unique(a)
selected_mcs = np.array([18, 23])

thr_matrix0 = np.zeros((len(a_unique), len(txgain_ax))) 
thr_matrix1 = np.zeros((len(a_unique), len(txgain_ax))) 

for i1 in range(len(a_unique)):
    for i2 in range(len(txgain_ax)):

        idx0 = (data['selected_mcs'] == selected_mcs[0]) & (data['txgain'] == txgain_ax[i2]) & (data['selected_airtime'] == a_unique[i1])
        idx1 = (data['selected_mcs'] == selected_mcs[1]) & (data['txgain'] == txgain_ax[i2]) & (data['selected_airtime'] == a_unique[i1])
        failed_experiment0 = data.loc[idx0]['failed_experiment'].item()
        failed_experiment1 = data.loc[idx1]['failed_experiment'].item()
        thr_matrix0[i1, i2] = data.loc[idx0]['thr'] if failed_experiment0 == 0 else 0
        thr_matrix1[i1, i2] = data.loc[idx1]['thr'] if failed_experiment1 == 0 else 0
        
maxthr = np.maximum(np.nanmax(thr_matrix0), np.nanmax(thr_matrix1))
X, Y = np.meshgrid(snr_ax, a_unique)

fig, axs = plt.subplots(2, 1, figsize=(10,7.5))
images = []
axs[0].set_title('$m$ = {}'.format(selected_mcs[1]), fontsize=28)
axs[1].set_title('$m$ = {}'.format(selected_mcs[0]), fontsize=28)

axs[0].set_ylabel('Airtime', fontsize=28)
axs[1].set_ylabel('Airtime', fontsize=28)
axs[1].set_xlabel('SNR (dB))', fontsize=28)

axs[0].tick_params(axis='both', labelsize=24)
axs[1].tick_params(axis='both', labelsize=24)
        
images.append(axs[0].contourf(X, Y, thr_matrix1 / maxthr, N, cmap=cmap_val))
images.append(axs[1].contourf(X, Y, thr_matrix0 / maxthr, N, cmap=cmap_val))
images.append(axs[0].contour(X, Y, thr_matrix1 / maxthr, N, cmap=cmap_val))
images.append(axs[1].contour(X, Y, thr_matrix0 / maxthr, N, cmap=cmap_val))

#Markers
point0 = [25, 0.6] #59.8%
point1 = [25, 0.8]  #57.2%

ms = 15
axs[0].plot(point0[0], point0[1], marker='o', c='maroon', markersize=ms)
axs[1].plot(point1[0], point1[1], marker='s', c='maroon', markersize=ms)

axs[0].text(6.5, .45, 'Unfeasible\n   region', fontsize=19, c='w')

norm = colors.Normalize(vmin=0, vmax=1)
for im in images:
    im.set_norm(norm)
    
fig.tight_layout()

cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0, .25, .5, .75, 1])
cbar.ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=24)
cbar.ax.set_ylabel('Throughput (%)', fontsize=24)

axs[0].annotate("      Airtime = 0.6\nThroughput = 59.8%",size=24, xy=(25, .57), xytext=(20, .25),  arrowprops=dict(arrowstyle='->,head_length=0.18,head_width=0.2', facecolor='black',color='black',lw='2'))
axs[1].annotate("      Airtime = 0.8\nThroughput = 57.2%",size=24, xy=(25, .77), xytext=(20, .4),  arrowprops=dict(arrowstyle='->,head_length=0.18,head_width=0.2', facecolor='black',color='black',lw='2'))

plt.savefig('./figure8.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')

plt.show()





