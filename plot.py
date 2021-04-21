import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

recon_data = pd.DataFrame()
loss_data = pd.DataFrame()
two_marbles = pd.read_csv("./results/10K/two_marbles_u0p1_10K/train_losses.log")
three_marbles = pd.read_csv("./results/10K/three_marbles_u0p1_10K/train_losses.log")
four_marbles = pd.read_csv("./results/10K/four_marbles_u0p1_10K/train_losses.log")

two_marbles['pile_num'] = 2
three_marbles['pile_num'] = 3
four_marbles['pile_num'] = 4

recon_data = recon_data.append(two_marbles.loc[two_marbles['Loss'] == "recon_loss"], ignore_index=True)
recon_data = recon_data.append(three_marbles.loc[three_marbles['Loss'] == "recon_loss"], ignore_index=True)
recon_data = recon_data.append(four_marbles.loc[four_marbles['Loss'] == "recon_loss"], ignore_index=True)

loss_data = loss_data.append(two_marbles.loc[two_marbles['Loss'] == "loss"], ignore_index=True)
loss_data = loss_data.append(three_marbles.loc[three_marbles['Loss'] == "loss"], ignore_index=True)
loss_data = loss_data.append(four_marbles.loc[four_marbles['Loss'] == "loss"], ignore_index=True)

recon_data = recon_data[recon_data.Epoch > 10]
recon_data = recon_data[recon_data.Epoch < 2500]

loss_data = loss_data[loss_data.Epoch > 10]
loss_data = loss_data[loss_data.Epoch < 2500]

fig, axes = plt.subplots(2)

sns.lineplot(x="Epoch", y="Value", data=recon_data, hue="pile_num", ax=axes[0])
sns.lineplot(x="Epoch", y="Value", data=loss_data, hue="pile_num", ax=axes[1])
plt.show()
