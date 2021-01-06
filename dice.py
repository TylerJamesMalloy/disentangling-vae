import numpy as np
from scipy.stats import uniform
from bvae import BetaVAE

numTrials = 1000 # make a parameter 
beta = 0.01

die_p = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
die_0 = [0,0,0,0,0,0]
die_1 = [1,0,3,0,5,0]
die_2 = [0,2,0,4,0,6]
die_3 = [1,2,3,0,0,0]
die_4 = [0,0,0,4,5,6]
die_5 = [1,2,3,4,5,6]
dice = [die_0, die_1, die_2, die_3, die_4, die_5]

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def l2_loss(x,y):
    return (np.sum((x-y)**2))

losses = []
klds = []

# how to incorporate bias into faster learning
# BetaVAEModel = BetaVAE(input_shape=[2,len(die_p),2])

for _ in range(numTrials):
    p = uniform.rvs(size=len(dice))
    p = p / np.sum(p)

    dice_eu = []
    for die in dice: 
        dice_eu.append(np.sum(die * die_p))

        inputs = [p, np.array(die)]
        print(BetaVAEModel.predict(inputs, mode="encode"))

    eu = (np.sum(dice_eu * p))
    
    # What if I represent the probability with a uniform distribution? 
    uni_p  = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    uni_eu = (np.sum(dice_eu * uni_p))
    
    loss = l2_loss(eu, uni_eu)
    losses.append(loss)

    kld = kl_divergence(uni_eu, eu)
    klds.append(kld)

    

print(np.mean(losses))
print(np.std(losses))
print(np.mean(klds))
print(np.std(klds))
    
