import numpy as np
from PIL import Image
import torch 
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # this was necessary on my machine, may not be on yours 


class utilityModel():
    def __init__(self, input_dim=20, layers=[64,64], args=None):

        self.output_dim = args.output_dim
        self.dataset = args.dataset 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layers[0], layers[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(layers[1], self.output_dim))
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.learning_rate = 5e-4
        self.batch_size = 1024
        self.epochs = 100
        self.utilities = np.load("./data/" + self.dataset + "/raw/1k_marbles_utilities.npy")
        self.epoch = 0

    def experience():
        # add experience to memory
        return 

    def memory():
        return 

    def getUtilityLoss(self, latent_dist, latent_sample, idxs):

        utils = self.utilities[idxs]
        utils = torch.from_numpy(utils).float().to(self.device)
        concat_dims = torch.cat((latent_dist[0],latent_dist[1]), dim=1)
        prediction = self.model(concat_dims)

        loss = torch.nn.MSELoss(reduction='none')
        loss_result = torch.sum(loss(utils,prediction),dim=0) 

        return np.mean(loss_result.cpu().detach().numpy())

    def trainUtility(self, data, latent_dist, latent_sample, idxs):
        concat_dims = torch.cat((latent_dist[0],latent_dist[1]), dim=1)

        self.epoch += 1

        utility_truth = self.utilities[idxs]
        utility_truth = torch.from_numpy(utility_truth).float().to(self.device)
        utility_prediction = self.model(concat_dims)

        loss = self.loss_fn(utility_prediction, utility_truth)

        self.model.zero_grad()
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.learning_rate * param.grad
