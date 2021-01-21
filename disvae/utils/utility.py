import numpy as np

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def getUtilityLoss(data=None, recon_data=None, flag=False):
    if(data is None or recon_data is None):
        print(" Warning: Got None value for data or reconstruction")
        print(" Data: ", data)
        print(" Recon: ", recon_data)
        return None 
    
    if(len(data.shape) > 3):
        data = np.squeeze(data)
    if(len(recon_data.shape) > 3):
        recon_data = np.squeeze(recon_data)

    eu_losses = []
    for batch_index, (recon, datum) in enumerate(zip(recon_data, data)): 
        # Probability of drawing each of the 32 dice within each of the 12 piles 
        recon_piles = recon[0:12]
        datum_piles = datum[0:12]

        # probabilities of each of the 10 sides for each of the 32 dice 
        recon_probabilities = recon[22:32]
        datum_probabilities = datum[22:32] 

        # normalized outcomes of each of the 10 sides for each of the 32 dice
        recon_outcomes = recon[12:22]
        datum_outcomes = datum[12:22]

        data_dice_EUs = []
        recon_dice_EUs = []
        for die_index in range(32):
            die_datum_probabilities =  datum_probabilities[:,die_index]
            if(np.sum(recon_probabilities[:,die_index]) > 0): # If the sum of the reconstructed probabiltiies is too small to matter, act randomly 
                die_recon_probabilities =  recon_probabilities[:,die_index] / np.sum(recon_probabilities[:,die_index])
            else:
                die_recon_probabilities = np.ones(10) / 10
                die_recon_probabilities = die_recon_probabilities / np.sum(die_recon_probabilities)

            die_datum_outcomes = datum_outcomes[:,die_index]
            die_recon_outcomes = recon_outcomes[:,die_index]

            data_dice_EUs.append(np.sum(die_datum_probabilities * die_datum_outcomes))
            recon_dice_EUs.append(np.sum(die_recon_probabilities * die_recon_outcomes))
        
        data_dice_EUs = np.asarray(data_dice_EUs)
        recon_dice_EUs = np.asarray(recon_dice_EUs)

        data_pile_EUs = []
        recon_pile_EUs = []
        for index, (recon_pile, datum_pile) in enumerate(zip(recon_piles, datum_piles)): 
            if(np.sum(recon_pile) > 0):
                recon_pile_probability = recon_pile / np.sum(recon_pile)
            else:
                recon_pile_probability = np.ones(32) / 32
                recon_pile_probability = recon_pile_probability / np.sum(recon_pile_probability)

            datum_pile_probability = datum_pile / np.sum(datum_pile)

            recon_pile_EUs.append(np.sum(recon_pile_probability * (100 * recon_dice_EUs))) # ecale eus 
            data_pile_EUs.append(np.sum(datum_pile_probability * (100 * data_dice_EUs)))   # scale eus 
        
        data_pile_EUs = np.asarray(data_pile_EUs)
        recon_pile_EUs = np.asarray(recon_pile_EUs)

        data_eu = np.max(data_pile_EUs)
        softmax_inverse_temp = 10
        denominator = np.sum(exp_normalize(recon_pile_EUs * softmax_inverse_temp))
        if(denominator > 0):
            recon_policy = exp_normalize(recon_pile_EUs * softmax_inverse_temp) / denominator
        else: 
            recon_policy = np.ones(12) / 12
            recon_policy = recon_policy / np.sum(recon_policy)

        recon_eu = np.sum(recon_policy * data_pile_EUs) 
        eu_losses.append((data_eu - recon_eu))

        if(eu_losses != eu_losses):
            print(" Got NAN expected utility loss")
            print(" Loss ", data_eu - recon_eu)
            print(" Data EU ", data_eu)
            print(" Recon EU ", recon_eu)
            print(" Recon Policy", recon_policy)
            print(" Data Pile EUs", data_pile_EUs)
            print(" ", recon_eu)
            assert(False)

    batch_mean_loss = np.mean(eu_losses)
    return batch_mean_loss