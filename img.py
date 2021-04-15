
from disvae.utils.modelIO import load_model
from PIL import Image
import numpy as np 
import torch
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # this was necessary on my machine, may not be on yours 

    

img_index = 5000

model_paths = ["marbles_b4_u0", "marbles_b4_u1e-2", "marbles_b4_u1e-1", "marbles_b40_u0", "marbles_b40_u1e-2", "marbles_b40_u1e-1", "marbles_b100_u0", "marbles_b100_u1e-2", "marbles_b100_u1e-1"]

for model_path in model_paths:

    bvae_model = load_model("./results/" + model_path + "/", filename="model-480.pt")
    imgs = np.load("./data/marbles/raw/10k_marbles.npy")
    utils = np.load("./data/marbles/raw/10k_marbles_utilities.npy")

    im = Image.fromarray(imgs[img_index])
    im.save("./original.jpeg")

    data = np.array([np.transpose(imgs[img_index])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(data).float().to(device)
    recon_batch, latent_dist, latent_sample = bvae_model(tensor)

    recon = recon_batch.cpu().detach().numpy()
    recon_img = np.transpose(recon[0]) * 255
    recon_img = recon_img.astype('uint8')

    im = Image.fromarray(recon_img)
    im.save("./recon_ " + model_path + ".jpeg")

assert(False)


def getProbabilities(data=None, recon_data=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.cpu().detach().numpy()
    recon_data = recon_data.cpu().detach().numpy()

    circle_indicies = [[1, 11], [14, 24], [27, 37], [40, 50], [53, 63]]
    color_values = [20, 12, 8, 4, 2, 1]

    util_mse = 0

    for datum, recon_datum in zip(data, recon_data):
        recon_utilities = []
        data_utilities = []

        for circle_x in circle_indicies:
            for circle_y in circle_indicies:

                #circle = datum[:,1:11,1:11]
                datum_circle = np.transpose(datum[:,circle_x[0]:circle_x[1],circle_y[0]:circle_y[1]]).astype("uint8") * 255
                recon_circle = np.transpose(recon_datum[:,circle_x[0]:circle_x[1],circle_y[0]:circle_y[1]]).astype("uint8") * 255
                recon_circle_colors = np.zeros(6)
                datum_circle_colors = np.zeros(6)

                for x in np.random.choice(10, 3):
                    for y in np.random.choice(10, 3):
                        datum_pixel = datum_circle[x,y,:]
                        recon_pixel = recon_circle[x,y,:]
                        if(np.array_equal(datum_pixel, [255, 255, 255]) or np.array_equal(datum_pixel, [0, 0, 0])):
                            # skip white and black pixles
                            continue

                        recon_color = get_closest_color(recon_pixel)
                        recon_circle_colors[recon_color] += 1

                recon_circle_colors
                    

                recon_utilities.append(color_values[recon_color_guess])
                data_utilities.append(color_values[datum_color_guess])

        util_mse += np.square(np.mean(recon_utilities) - np.mean(data_utilities))

    return util_mse

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
dqn = DQN()



