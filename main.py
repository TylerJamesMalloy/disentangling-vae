import argparse
import logging
import sys
import os
from configparser import ConfigParser

import numpy as np
import pandas as pd 

import torch 
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

from torch import optim

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, get_gendata, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining


CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')
                         
    general.add_argument('-is-eu', type=str2bool, nargs='?', const=True, default=True,
                         help='Determine if the EU loss will be calculated and saved.')
    general.add_argument('-u', type=float, default=default_config['u'],
                         help='Utility weight for loss calculation')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood to use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-gen-only', action='store_true',
                            default=default_config['is_gen_only'],
                            help='Whether to only evaluate generalization using precomputed model `name`.')
    evaluation.add_argument('--model-num',  type=str,
                            default=default_config['model_num'],
                            help='Dimension of the latent variable.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    if not args.is_eval_only and not args.is_gen_only:

        create_safe_directory(exp_dir, logger=logger)

        if args.loss == "factor":
            logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
            args.batch_size *= 2
            args.epochs *= 2

        # PREPARES DATA
        train_loader = get_dataloaders(args.dataset,
                                       batch_size=args.batch_size,
                                       logger=logger)
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        
        # PREPARES MODEL
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
        logger.info('Num parameters in model: {}'.format(get_n_param(model)))

        # TRAINS
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)  # make sure trainer and viz on same device
        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          gif_visualizer=gif_visualizer,
                          is_utility=args.is_eu)
        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if (args.is_metrics or not args.no_test) and not args.is_gen_only:
        #model = load_model(exp_dir, is_gpu=not args.no_cuda, filename="model-" + args.model_num + ".pt")
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        # TO-DO: currently uses train datatset
        test_loader = get_dataloaders(metadata["dataset"],
                                      batch_size=args.eval_batchsize,
                                      shuffle=False,
                                      logger=logger)
        loss_f = get_loss_f(args.loss,
                            n_data=len(test_loader.dataset),
                            device=device,
                            **vars(args))
        evaluator = Evaluator(model, loss_f,
                              device=device,
                              logger=logger,
                              save_dir=exp_dir,
                              is_progress_bar=not args.no_progress_bar)

        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)

    if args.is_gen_only: 
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        # TO-DO: currently uses train datatset

        gen_datasets = ["train.npy"]

        for gen_dataset in gen_datasets: 
            test_loader = get_gendata(  genData = gen_dataset,
                                        batch_size=args.eval_batchsize,
                                        shuffle=False,
                                        logger=logger)
            
            data_index = 0
            recon_diff = pd.DataFrame()
            for data, _ in test_loader:
                data_index += 1

                datum = data.to(device)#.cpu().numpy()
                recon_datum, latent_dist, latent_sample = model(datum)

                recon_data = np.squeeze(recon_datum.detach().cpu().numpy())
                stimulus_data = np.squeeze(datum.detach().cpu().numpy())

                all_data_piles = stimulus_data[:,0:12]
                die_sums = np.sum(np.sum(all_data_piles, axis=0), axis=0)

                common_die_index = np.argmax(die_sums)
                uncommon_die_min = np.argmin(die_sums)
                uncommon_die_indicies = [0,1,2,3,4,5,6,27,28,29,30,31]

                for recon_datum, stimulus_datum in zip(recon_data, stimulus_data):
                    # Probability of drawing each of the 32 dice within each of the 12 piles 
                    recon_piles = recon_datum[0:12]
                    datum_piles = stimulus_datum[0:12]

                    # probabilities of each of the 10 sides for each of the 32 dice 
                    recon_probabilities = recon_datum[22:32]
                    datum_probabilities = stimulus_datum[22:32] 

                    # normalized outcomes of each of the 10 sides for each of the 32 dice
                    recon_outcomes = recon_datum[12:22]
                    datum_outcomes = stimulus_datum[12:22]

                    # most common die outcome and utility reconstruction accuracy: 
                    
                    common_prob_recon = recon_probabilities[:,common_die_index] / np.sum(recon_probabilities[:,common_die_index])
                    common_prob_datum = datum_probabilities[:,common_die_index] / np.sum(datum_probabilities[:,common_die_index])

                    common_outcome_recon = recon_outcomes[:,common_die_index] / np.sum(recon_outcomes[:,common_die_index])
                    common_outcome_datum = datum_outcomes[:,common_die_index] / np.sum(datum_outcomes[:,common_die_index])

                    common_error  = np.mean(np.abs(common_prob_datum - common_prob_recon)) + np.mean(np.abs(common_outcome_recon - common_outcome_datum))
                    # Least common die outcome and utility reconstruction accuracy: 

                    uncommon_errors = []
                    for uncommon_die_index in uncommon_die_indicies: 
                        uncommon_prob_recon = recon_probabilities[:,uncommon_die_index] / np.sum(recon_probabilities[:,uncommon_die_index])
                        uncommon_prob_datum = datum_probabilities[:,uncommon_die_index] / np.sum(datum_probabilities[:,uncommon_die_index])

                        uncommon_outcome_recon = recon_outcomes[:,uncommon_die_index] / np.sum(recon_outcomes[:,uncommon_die_index])
                        uncommon_outcome_datum = datum_outcomes[:,uncommon_die_index] / np.sum(datum_outcomes[:,uncommon_die_index])

                        uncommon_errors.append(np.mean(np.abs(uncommon_prob_datum - uncommon_prob_recon)) + np.mean(np.abs(uncommon_outcome_recon - uncommon_outcome_datum)))

                    uncommon_error_mean = np.mean(uncommon_errors)

                    uncommon_prob_recon = recon_probabilities[:,uncommon_die_min] / np.sum(recon_probabilities[:,uncommon_die_min])
                    uncommon_prob_datum = datum_probabilities[:,uncommon_die_min] / np.sum(datum_probabilities[:,uncommon_die_min])

                    uncommon_outcome_recon = recon_outcomes[:,uncommon_die_min] / np.sum(recon_outcomes[:,uncommon_die_min])
                    uncommon_outcome_datum = datum_outcomes[:,uncommon_die_min] / np.sum(datum_outcomes[:,uncommon_die_min])

                    uncommon_error  = np.mean(np.abs(uncommon_prob_datum - uncommon_prob_recon)) + np.mean(np.abs(uncommon_outcome_recon - uncommon_outcome_datum))

                    
                    if(common_error > uncommon_error ):

                        print("uncommon_error ",uncommon_error, " common_error ",common_error )

                        print("uncommon prob ", np.mean(np.abs(uncommon_prob_datum - uncommon_prob_recon)))
                        print("uncommon out ", np.mean(np.abs(uncommon_outcome_recon - uncommon_outcome_datum)))

                        print("common prob ", np.mean(np.abs(common_prob_datum - common_prob_recon)))
                        print("common out ", np.mean(np.abs(common_outcome_recon - common_outcome_datum)))

                        assert(False)
                    
                    
                    recon_diff = recon_diff.append({"uncommon_error":uncommon_error, "common_error":common_error, "uncommon_error_mean":uncommon_error_mean}, ignore_index=True)


            #recon_diff.to_pickle(exp_dir + "\FrequencyError.pkl")
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)



"""
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

#print(data_eu - recon_eu)
pile_eu_error = (data_pile_EUs / np.sum(data_pile_EUs)) - (recon_pile_EUs / np.sum(recon_pile_EUs)) 


max_eu_mse_pile_index = np.argmax(pile_eu_error)
min_eu_mse_pile_index = np.argmin(pile_eu_error)

max_eu_pile_index = np.argmax(data_pile_EUs)
min_eu_pile_index = np.argmin(data_pile_EUs)

min_max_mse_diff = pile_eu_error[max_eu_pile_index] - pile_eu_error[min_eu_pile_index]

# most common die outcome and utility reconstruction accuracy: 
common_die_index = np.argmax(np.sum(datum_piles, axis=0))
common_prob_recon = recon_probabilities[:,common_die_index] / np.sum(recon_probabilities[:,common_die_index])
common_prob_datum = datum_probabilities[:,common_die_index] / np.sum(datum_probabilities[:,common_die_index])

common_outcome_recon = recon_outcomes[:,common_die_index] / np.sum(recon_outcomes[:,common_die_index])
common_outcome_datum = datum_outcomes[:,common_die_index] / np.sum(datum_outcomes[:,common_die_index])

common_mse  = np.mean((common_prob_datum - common_prob_recon)**2) + np.mean((common_outcome_recon - common_outcome_datum)**2)
# Least common die outcome and utility reconstruction accuracy: 
uncommon_die_index = np.argmin(np.sum(datum_piles, axis=0))
uncommon_prob_recon = recon_probabilities[:,uncommon_die_index] / np.sum(recon_probabilities[:,uncommon_die_index])
uncommon_prob_datum = datum_probabilities[:,uncommon_die_index] / np.sum(datum_probabilities[:,uncommon_die_index])

uncommon_outcome_recon = recon_outcomes[:,uncommon_die_index] / np.sum(recon_outcomes[:,uncommon_die_index])
uncommon_outcome_datum = datum_outcomes[:,uncommon_die_index] / np.sum(datum_outcomes[:,uncommon_die_index])

uncommon_mse  = np.mean(np.abs(uncommon_prob_datum - uncommon_prob_recon)) + np.mean(np.abs(uncommon_outcome_recon - uncommon_outcome_datum))

# highest utility die reconstruction accuracy
hud_index  = np.argmax(data_dice_EUs)
hud_prob_recon = recon_probabilities[:,hud_index] / np.sum(recon_probabilities[:,hud_index])
hud_prob_datum = datum_probabilities[:,hud_index] / np.sum(datum_probabilities[:,hud_index])

hud_out_recon = recon_outcomes[:,hud_index] / np.sum(recon_outcomes[:,hud_index])
hud_out_datum = datum_outcomes[:,hud_index] / np.sum(datum_outcomes[:,hud_index])

hud_mse  = np.mean(np.abs(hud_prob_datum - hud_prob_recon))  + np.mean(np.abs(hud_out_recon - hud_out_datum))

# lowest utility die reconstruction accuracy 
lud_index  = np.argmin(data_dice_EUs)
lud_prob_recon = recon_probabilities[:,lud_index] / np.sum(recon_probabilities[:,lud_index])
lud_prob_datum = datum_probabilities[:,lud_index] / np.sum(datum_probabilities[:,lud_index])

lud_out_recon = recon_outcomes[:,lud_index] / np.sum(recon_outcomes[:,lud_index])
lud_out_datum = datum_outcomes[:,lud_index] / np.sum(datum_outcomes[:,lud_index])

lud_mse  = np.mean(np.abs(lud_prob_recon - lud_prob_datum))  + np.mean(np.abs(lud_out_recon - lud_out_datum))

common_prob_recon = recon_probabilities[:,common_die_index] / np.sum(recon_probabilities[:,common_die_index])
common_prob_datum = datum_probabilities[:,common_die_index] / np.sum(datum_probabilities[:,common_die_index])

common_outcome_recon = recon_outcomes[:,common_die_index] / np.sum(recon_outcomes[:,common_die_index])
common_outcome_datum = datum_outcomes[:,common_die_index] / np.sum(datum_outcomes[:,common_die_index])

common_error  = np.mean(np.abs(common_prob_datum - common_prob_recon)) + np.mean(np.abs(common_outcome_recon - common_outcome_datum))
# Least common die outcome and utility reconstruction accuracy: 

uncommon_errors = []
for uncommon_die_index in uncommon_die_indicies: 
    uncommon_prob_recon = recon_probabilities[:,uncommon_die_index] / np.sum(recon_probabilities[:,uncommon_die_index])
    uncommon_prob_datum = datum_probabilities[:,uncommon_die_index] / np.sum(datum_probabilities[:,uncommon_die_index])

    uncommon_outcome_recon = recon_outcomes[:,uncommon_die_index] / np.sum(recon_outcomes[:,uncommon_die_index])
    uncommon_outcome_datum = datum_outcomes[:,uncommon_die_index] / np.sum(datum_outcomes[:,uncommon_die_index])

    uncommon_errors.append(np.mean(np.abs(uncommon_prob_datum - uncommon_prob_recon)) + np.mean(np.abs(uncommon_outcome_recon - uncommon_outcome_datum)))

uncommon_error  = np.mean(uncommon_errors)

mse_diff = (uncommon_mse - common_mse) 
recon_diff = recon_diff.append({"uncommon_mse":uncommon_mse, "common_mse":common_mse, "hud_mse":hud_mse, "lud_mse":lud_mse, "max_pile_error":pile_eu_error[max_eu_pile_index], "min_pile_error":pile_eu_error[min_eu_pile_index]}, ignore_index=True)
"""