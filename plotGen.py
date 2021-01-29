import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 
import numpy as np
import scipy as sp 

import os.path
from os import path

import json 

data = pd.DataFrame()
data_array = []

plt.style.use('seaborn') # pretty matplotlib plots
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)

matplotlib.rc('xtick', labelsize=32) 
matplotlib.rc('ytick', labelsize=32) 

upsilons = ["u0", "u1000"]
betas = ["b1", "b4" "b10", "b25",  "b100", "b250", "b500", "b1000"]
agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]

data = pd.DataFrame()

for upsilon in upsilons: 
    for beta in betas: 
        genData = pd.DataFrame()
        for agent in agents:
            gen_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent  + "/" + "BigGen.pkl"
            recon_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent + "/" + "test_losses.log"
            if(path.exists(gen_file) and path.exists(recon_file)):
                mses = np.abs(np.mean(pd.read_pickle(gen_file).to_numpy(), axis=0))
                f = open(recon_file, "r")
                recon_loss = f.read().replace('\n', '')

                
                d = json.loads(recon_loss) 
                d["upsilon"] = float(upsilon[1:])
                d["beta"] = np.log(float(beta[1:]))
                #d["beta"] = float(beta[1:])

                d["common_mse"]     = mses[0]
                d["hud_mse"]        = mses[1]
                d["lud_mse"]        = mses[2]
                d["uncommon_mse"]   = mses[3]
                d["max"]            = mses[4] 
                d["min"]            = mses[5] 


                genData = genData.append(d, ignore_index=True)
        
        if(not genData.empty):
            genData['common_mean'] = np.mean(genData["common_mse"])
            genData['uncommon_mean'] = np.mean(genData["uncommon_mse"])
            genData['hud_mean'] = np.mean(genData["hud_mse"])
            genData['lud_mean'] = np.mean(genData["lud_mse"])
            genData['max_mean'] = np.mean(genData["max"])
            genData['min_mean'] = np.mean(genData["min"])

            genData['common_diff']  = genData['common_mean'] / genData['uncommon_mean']
            genData['utility_diff'] = genData['hud_mean'] / genData['lud_mean']

            data = data.append(genData, ignore_index=True)

#  recon_diff = recon_diff.append({"uncommon_mse":uncommon_mse, "common_mse":common_mse, "hud_mse":hud_mse, "lud_mse":lud_mse, "max_pile_error":pile_eu_error[max_eu_pile_index], "min_pile_error":pile_eu_error[min_eu_pile_index]}, ignore_index=True)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.suptitle("Reconstruction Accuracy Difference by Die Utility", size=42)

u0_data = data.loc[data["upsilon"] == 0]
u1000_data = data.loc[data["upsilon"] == 1000]

x = u0_data["beta"] 
y = u0_data["hud_mean"]

trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y -.18,  p0=(0.1, 0.1))[0]

#trend = [1.77980882e-04, 8.55227880e-01]

xs = np.linspace(0,7) 
ys = trend[0] * np.exp(trend[1] * xs) + .18
line = ax1.plot(x, y, 'b.', xs, ys, 'b-')
line[1].set_label(r'$\upsilon = 0$')

x = u1000_data["beta"] 
y = u1000_data["hud_mean"]
trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y -.17,  p0=(0.1, 0.1))[0]

xs = np.linspace(0,7)
ys = trend[0] * np.exp(trend[1] * xs) +.17
line = ax1.plot(x, y, 'r.', xs, ys, 'r-')
line[1].set_label(r'$\upsilon = 1000$')

x = u0_data["beta"] 
y = u0_data["lud_mean"]

trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y -.21 ,  p0=(0.1, 0.1))[0]

xs = np.linspace(0,7)
ys = trend[0] * np.exp(trend[1] * xs) + .21
line = ax2.plot(x, y, 'b.', xs, ys, 'b-')
line[1].set_label(r'$\upsilon = 0$')

x = u1000_data["beta"] 
y = u1000_data["lud_mean"]
trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y -.23,  p0=(0.1, 0.1))[0]

xs = np.linspace(0,7)
ys = trend[0] * np.exp(trend[1] * xs) +.23
line = ax2.plot(x, y, 'r.', xs, ys, 'r-')
line[1].set_label(r'$\upsilon = 1000$')

ax1.set_title("Highest Utility Die", size = 32)
ax2.set_title("Lowest Utility Die", size = 32)

ax1.set_ylabel("Reconstruction Error", size = 32)

plt.legend(fontsize=32) # using a size in points
ax1.set_xlabel("Information Bottleneck", size = 32)
ax2.set_xlabel("Information Bottleneck", size = 32)

plt.show()
assert(False)

assert(False)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
#fig, ax1 = plt.subplots(1, 1)
plt.suptitle("Reconstruction Accuracy Difference by Die Utility", size=42)

sns.lineplot(data=data, x="beta", y="max_mean", hue="upsilon", ax=ax1)
sns.lineplot(data=data, x="beta", y="min_mean", hue="upsilon", ax=ax2)

ax1.set_title("Max Utility Pile", size=32)
ax2.set_title("Min Utility Pile", size=32)

ax1.set_ylabel("Reconstruction Error", size=32)
ax2.set_ylabel("", size=32)

handles, labels = ax1.get_legend_handles_labels()
#fig.legend(handles, labels, prop={'size': 28}, ncol=2)

ax1.get_legend().remove()
#ax2.get_legend().remove()

ax1.set_xlabel("Information Constraint", size=32)
ax2.set_xlabel("Information Constraint", size=32)


plt.show()