import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


betaB_10_utility_dice = pd.read_pickle('results/betaB_10_utility_dice/utility.pkl')
betaB_50_utility_dice = pd.read_pickle('results/betaB_50_utility_dice/utility.pkl')

betaB_10_utility_dice["Capacity"] = 10
betaB_50_utility_dice["Capacity"] = 50

betaB_utility = pd.DataFrame()
betaB_utility = betaB_utility.append(betaB_10_utility_dice, ignore_index = True)
betaB_utility = betaB_utility.append(betaB_50_utility_dice, ignore_index = True)

betaB_10_bernouli_dice = pd.read_pickle('results/betaB_10_bernouli_dice/utility.pkl')
betaB_50_bernouli_dice = pd.read_pickle('results/betaB_50_bernouli_dice/utility.pkl')

betaB_10_bernouli_dice["Capacity"] = 10
betaB_50_bernouli_dice["Capacity"] = 50

betaB_bernouli = pd.DataFrame()
betaB_bernouli = betaB_bernouli.append(betaB_10_bernouli_dice, ignore_index = True)
betaB_bernouli = betaB_bernouli.append(betaB_50_bernouli_dice, ignore_index = True)

utility_10_dice = pd.read_pickle('results/utility_10_dice/utility.pkl')
utility_50_dice = pd.read_pickle('results/utility_50_dice/utility.pkl')

utility_10_dice["Capacity"] = 10
utility_50_dice["Capacity"] = 50

print(betaB_10_bernouli_dice)

utility = pd.DataFrame()
utility = utility.append(utility_10_dice, ignore_index = True)
utility = utility.append(utility_50_dice, ignore_index = True)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax = sns.lineplot(x="epoch", y="loss", hue="Capacity", data=betaB_utility, ax=ax1, ci=99)  
ax = sns.lineplot(x="epoch", y="loss", hue="Capacity", data=betaB_bernouli, ax=ax2, ci=99) 
ax = sns.lineplot(x="epoch", y="loss", hue="Capacity", data=utility, ax=ax3, ci=99) 
plt.show()
