import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms 

from disvae.utils.modelIO import load_model
from PIL import Image
import numpy as np 
import torch
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # this was necessary on my machine, may not be on yours 


MODEL_PATH = "./results/marbles_b40_u0/"

bvae_model = load_model(MODEL_PATH, filename="model-480.pt")
imgs = np.load("./data/marbles/raw/10k_marbles.npy")
utils = np.load("./data/marbles/raw/10k_marbles_utilities.npy")

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    model = model.to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-5
    batch_size = 1024
    epochs = 100

    def recon_to_image(recon_batch):
        recon = recon_batch.cpu().detach().numpy()
        recon_img = np.transpose(recon[0]) * 255
        recon_img = recon_img.astype('uint8')

        im = Image.fromarray(recon_img)
        im.save("./recon.jpeg")

    for epoch in range(epochs):

        mini_batch = np.random.choice(10000, batch_size)

        epoch_loss = []

        for batch_index, index in enumerate(mini_batch):
            data = np.array([np.transpose(imgs[index])])
            tensor = torch.from_numpy(data).float().to(device)
            recon_batch, latent_dist, latent_sample = bvae_model(tensor)
            concat_dims = torch.cat((latent_dist[0],latent_dist[1]), dim=1)

            utility_truth = torch.from_numpy(np.array([[utils[index]]])).float().to(device)
            utility_prediction = model(concat_dims)
            
            loss = loss_fn(utility_prediction, utility_truth)

            epoch_loss.append(loss.item())

            if(batch_index == len(mini_batch) - 1):
                print("epoch: ", epoch, " has mean loss ", np.mean(epoch_loss))
            
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

    torch.save(model.state_dict(), MODEL_PATH + "utility_model.pt")

def load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1))

    model_paths = ["marbles_b4_u0/", "marbles_b4_u1e-1/", "marbles_b40_u0/", "marbles_b40_u1e-2/", "marbles_b40_u1e-1/", "marbles_b100_u0/", "marbles_b100_u1e-2/", "marbles_b100_u1e-1/"]
    
    img_index = 5000

    for model_path in model_paths:
        model.load_state_dict(torch.load("./results/" + model_path + "/" + "utility_model.pt"))
        model = model.to(device)

        data = np.array([np.transpose(imgs[img_index])])
        tensor = torch.from_numpy(data).float().to(device)

        recon_batch, latent_dist, latent_sample = bvae_model(tensor)
        concat_dims = torch.cat((latent_dist[0],latent_dist[1]), dim=1)

        print(model(concat_dims).item())
    
    print(utils[img_index])


load()