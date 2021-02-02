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
betas = ["b1", "b8", "b10", "b25","b100", "b250", "b500", "b1000"]
agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]


data = pd.DataFrame()

for upsilon in upsilons: 
    for beta in betas: 
        agent_data = pd.DataFrame()
        for agent in agents:
            utility_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent + "/gen/" + "test_utilities.log"
            recon_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent + "/gen/" + "test_losses.log"
            
            if(path.exists(utility_file) and path.exists(recon_file)):
                f = open(utility_file, "r")
                utility_loss = f.readline()

                f = open(recon_file, "r")
                recon_loss = f.read().replace('\n', '')
                d = json.loads(recon_loss) 
                d["upsilon"] = float(upsilon[1:])
                d["beta"] = np.log(float(beta[1:]))
                d["utility_loss"] = float(utility_loss)

                agent_data = agent_data.append(d, ignore_index=True)

        if(not agent_data.empty):
            agent_data = agent_data.sort_values(['utility_loss'],ascending=False)
            agent_data['ul_mean'] = np.mean(agent_data["utility_loss"])

            #q = agent_data.groupby('beta')['utility_loss'].transform(lambda x: x.quantile(.9))
            #agent_data = agent_data[agent_data.utility_loss < q]

            data = data.append(agent_data, ignore_index=True)

#  recon_diff = recon_diff.append({"uncommon_mse":uncommon_mse, "common_mse":common_mse, "hud_mse":hud_mse, "lud_mse":lud_mse, "max_pile_error":pile_eu_error[max_eu_pile_index], "min_pile_error":pile_eu_error[min_eu_pile_index]}, ignore_index=True)


#fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax1 = plt.subplots(1, 1)
plt.suptitle("Generalization Loss by Information Bottleneck", size=42)

u0_data = data.loc[data["upsilon"] == 0]
#u500_data = data.loc[data["upsilon"] == 500]
u1000_data = data.loc[data["upsilon"] == 1000]

x = u0_data["beta"] 
y = u0_data["ul_mean"]

trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y - 0.1,  p0=(4, 0.1))[0]

xs = np.linspace(0,7)
ys = trend[0] * np.exp(trend[1] * xs) + 0.1
line = plt.plot(x, y, 'b.', xs, ys, 'b-')
line[1].set_label(r'$\upsilon = 0$')

x = u1000_data["beta"] 
y = u1000_data["ul_mean"]
trend = sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y - 0.35,  p0=(4, 0.1))[0]

xs = np.linspace(0,7)
ys = trend[0] * np.exp(trend[1] * xs) + 0.35
line = plt.plot(x, y, 'r.', xs, ys, 'r-')
line[1].set_label(r'$\upsilon = 1000$')

plt.legend(fontsize=32) # using a size in points
plt.xlabel("Information Bottleneck", size = 32)
plt.ylabel("Utility Loss", size = 32)
plt.show()
assert(False)

sns.pointplot(data=data, x="beta", y="utility_loss", hue="upsilon", ax=ax1, join=False)
#sns.lineplot(data=data, x="beta", y="recon_loss", hue="upsilon", ax=ax2)
ax1.set_ylabel("Expected Utility Loss", size=32)
#ax2.set_ylabel("Reconstruction Loss", size=32)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, prop={'size': 28}, ncol=2)

ax1.get_legend().remove()
#ax2.get_legend().remove()

ax1.set_xlabel("Beta Information Capacity Parameter", size=32)
#ax2.set_xlabel("Beta", size=32)


plt.show()