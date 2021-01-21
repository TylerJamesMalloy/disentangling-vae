import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 

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


upsilons = ["u0", "u10", "u100", "u1000"]
betas = ["b1", "b10", "b25", "b50", "b100", "b500", "b1000"]
agents = ["a1", "a3", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]

data = pd.DataFrame()

for upsilon in upsilons: 
    for beta in betas: 
        for agent in agents:
            utility_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent + "/" + "test_utilities.log"
            recon_file = "./results/" + upsilon + "/" + "dice_" + beta + "_" + agent + "/" + "test_losses.log"
            if(path.exists(utility_file) and path.exists(recon_file)):
                f = open(utility_file, "r")
                utility_loss = f.readline()

                f = open(recon_file, "r")
                recon_loss = f.read().replace('\n', '')
                d = json.loads(recon_loss) 
                d["upsilon"] = upsilon
                d["beta"] = beta[1:]
                d["utility_loss"] = float(utility_loss)

                data = data.append(d, ignore_index=True)

#fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax1 = plt.subplots(1, 1)
plt.suptitle("Loss Rate-Distortion Curve", size=42)

sns.lineplot(data=data, x="beta", y="utility_loss", hue="upsilon", ax=ax1)
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