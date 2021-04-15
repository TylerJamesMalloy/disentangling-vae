import numpy as np
from PIL import Image
import torch 
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # this was necessary on my machine, may not be on yours 


def fancy_argmin(input_list):
    # return the index of the lowest argument, if any are tied for lowest pick randomly 
    list_min = np.min(input_list)
    mins = []
    for val_index, val in enumerate(input_list):
        if(val == list_min):
            mins.append(val_index)
    
    if(len(mins) > 1):
        return np.random.choice(mins, 1)[0]
    else:
        return int(list_min)

def get_closest_color(pixel):
    red_mse = np.mean(np.square(pixel - [255, 0, 0]))
    green_mse = np.mean(np.square(pixel - [0, 255, 0]))
    blue_mse = np.mean(np.square(pixel - [0, 0, 255]))
    yellow_mse = np.mean(np.square(pixel - [255, 255, 0]))
    magenta_mse = np.mean(np.square(pixel - [255, 0, 255]))
    cyan_mse = np.mean(np.square(pixel - [0, 255, 255]))

    return fancy_argmin([magenta_mse, blue_mse, red_mse, cyan_mse, green_mse, yellow_mse])

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def getUtilityLoss(data=None, recon_data=None):
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
                        
                        datum_color = get_closest_color(datum_pixel)
                        datum_circle_colors[datum_color] += 1

                        recon_color = get_closest_color(recon_pixel)
                        recon_circle_colors[recon_color] += 1

                if(np.sum(recon_circle_colors) > 0):
                    recon_circle_colors = recon_circle_colors / np.sum(recon_circle_colors)
                    recon_color_guess = np.random.choice(range(6), p=recon_circle_colors)
                else:
                    recon_color_guess = np.random.choice(range(6))
                    

                if(np.sum(datum_circle_colors) > 0):
                    datum_circle_colors = datum_circle_colors / np.sum(datum_circle_colors)
                    datum_color_guess = np.random.choice(range(6), p=datum_circle_colors)
                else:
                    datum_color_guess = np.random.choice(range(6))

                recon_utilities.append(color_values[recon_color_guess])
                data_utilities.append(color_values[datum_color_guess])

        util_mse += np.square(np.mean(recon_utilities) - np.mean(data_utilities))

    return util_mse