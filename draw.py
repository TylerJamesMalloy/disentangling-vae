from graphics import *
import numpy as np 
from PIL import Image as PILImage
import os

num_images = 1000
# s = np.random.dirichlet((4, 8, 12, 14, 16, 20), num_images)
s = np.random.dirichlet((2, 4, 6, 8), num_images)

def old_get_color_rgb(color):
    if(color == 0):
        return (255, 0, 255)    # Magenta
    if(color == 1):
        return (0, 255, 0)      # Blue 
    if(color == 2):
        return (255, 0, 0)      # Red 
    if(color == 3):
        return (0, 255, 255)    # Cyan
    if(color == 4):
        return (0, 0, 255)      # Green 
    if(color == 5):
        return (255, 255, 0)    # Yellow

def get_color_rgb(color):
    if(color == 0):
        return (128, 128, 128)  # Brown 
    if(color == 1):
        return (0, 255, 0)      # Blue 
    if(color == 2):
        return (255, 0, 0)      # Red 
    if(color == 3):
        return (0, 0, 255)      # Green 
    
data = []
utils = []

def draw_one():
    circle_xs = [12, 32, 52, 72] 
    circle_ys = circle_xs

    # 2, 2, 4, 8, 12, 20 
    utilities = [8, 4, 2, 1]
    stimulus_utility = 0

    for circle_x in circle_xs:
        for circle_y in circle_ys:
            #color = np.random.choice(6, 1, p=distribution)
            color = np.random.choice(4, 1, p=distribution)
            rgb = get_color_rgb(color)

            stimulus_utility += utilities[int(color)]

            pt = Point(circle_x, circle_y)
            cir = Circle(pt, 9)
            color = color_rgb(rgb[0], rgb[1], rgb[2])
            cir.setFill(color)
            cir.draw(win)
    utils.append(stimulus_utility)

def draw_two():
    #circle_xs = [5, 13, 21, 29,   38, 46, 54, 62]  
    circle_xs = [6, 16, 26, 36,   49, 59, 69, 79]  
    circle_ys = [26, 37, 48, 59] 

    # 2, 2, 4, 8, 12, 20 
    utilities = [8, 4, 2, 1]
    pile_utils = [0,0]
    for circle_x in circle_xs:
        for circle_y in circle_ys:
            #color = np.random.choice(6, 1, p=distribution)
            color = np.random.choice(4, 1, p=distribution)
            rgb = get_color_rgb(color)
            
            if(circle_x > 37):
                pile_utils[0] += utilities[int(color)]
            else:
                pile_utils[1] += utilities[int(color)]

            pt = Point(circle_x, circle_y)
            cir = Circle(pt, 5)
            color = color_rgb(rgb[0], rgb[1], rgb[2])
            cir.setFill(color)
            cir.draw(win)
    
    
    utils.append(pile_utils)
    print(pile_utils)

    rect1 = Rectangle(Point(-1, 20), Point(43,65))
    rect1.draw(win)
    rect2 = Rectangle(Point(43, 20), Point(85,65))
    rect2.draw(win)

def draw_three():
    # first two piles: 
    circle_xs = [6, 16, 26, 36,   49, 59, 69, 79]  
    circle_ys = [5, 15, 25, 35] 

    # 2, 2, 4, 8, 12, 20 
    utilities = [8, 4, 2, 1]
    pile_utils = [0, 0, 0]
    for circle_x in circle_xs:
        for circle_y in circle_ys:
            #color = np.random.choice(6, 1, p=distribution)
            color = np.random.choice(4, 1, p=distribution)
            rgb = get_color_rgb(color)

            if(circle_x > 37):
                pile_utils[0] += utilities[int(color)]
            else:
                pile_utils[1] += utilities[int(color)]

            pt = Point(circle_x, circle_y)
            cir = Circle(pt, 5)
            color = color_rgb(rgb[0], rgb[1], rgb[2])
            cir.setFill(color)
            cir.draw(win)
    
    circle_xs = [26, 36, 49, 59]  
    circle_ys = [49, 59, 69, 79] 
    for circle_x in circle_xs:
        for circle_y in circle_ys:
            #color = np.random.choice(6, 1, p=distribution)
            color = np.random.choice(4, 1, p=distribution)
            rgb = get_color_rgb(color)

            pile_utils[2] += utilities[int(color)]

            pt = Point(circle_x, circle_y)
            cir = Circle(pt, 5)
            color = color_rgb(rgb[0], rgb[1], rgb[2])
            cir.setFill(color)
            cir.draw(win)

    utils.append(pile_utils)
    print(pile_utils)

    rect1 = Rectangle(Point(-1, -1), Point(43,43))
    rect1.draw(win)
    rect2 = Rectangle(Point(43, -1), Point(84,43))
    rect2.draw(win)
    rect2 = Rectangle(Point(21, 42), Point(64,84))
    rect2.draw(win)

def draw_four():
    # first two piles: 
    circle_xs = [6, 16, 26, 36,  49, 59, 69, 79]  
    circle_ys = [5, 15, 25, 35,  49, 59, 69, 79] 

    utilities = [8, 4, 2, 1]
    pile_utils = [0, 0, 0, 0]
    count = [0, 0, 0, 0]
    for circle_x in circle_xs:
        for circle_y in circle_ys:
            #color = np.random.choice(6, 1, p=distribution)
            color = np.random.choice(4, 1, p=distribution)
            rgb = get_color_rgb(color)

            if(circle_x > 37 and circle_y > 37):
                pile_utils[0] += utilities[int(color)]
                count[0] += 1
            elif(circle_x > 37 and circle_y < 37):
                pile_utils[1] += utilities[int(color)]
                count[1] += 1
            elif(circle_x < 37 and circle_y < 37):
                pile_utils[2] += utilities[int(color)]
                count[2] += 1
            else:
                pile_utils[3] += utilities[int(color)]
                count[3] += 1

            pt = Point(circle_x, circle_y)
            cir = Circle(pt, 5)
            color = color_rgb(rgb[0], rgb[1], rgb[2])
            cir.setFill(color)
            cir.draw(win)
    print(pile_utils)
    utils.append(pile_utils)

    rect1 = Rectangle(Point(-1, -1), Point(43,43))
    rect1.draw(win)
    rect2 = Rectangle(Point(43, -1), Point(85,43))
    rect2.draw(win)

    rect1 = Rectangle(Point(-1, 43), Point(85,85))
    rect1.draw(win)
    rect2 = Rectangle(Point(43, 43), Point(85,85))
    rect2.draw(win)


folder = "four_marbles"
for image_number, distribution in enumerate(s):
    #print(distribution)
    image=Image(Point(42,42), 84, 84)
    win = GraphWin(width = 84, height = 84) # create a window
    win.setCoords(0, 0, 84, 84) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
    mySquare = Rectangle(Point(-1, -1), Point(85, 85)) # create a rectangle from (1, 1) to (9, 9)
    mySquare.draw(win) # draw it to the window

    # draw_one()
    # draw_two()
    # draw_three()
    draw_four()

    filename = "./data/" + folder + "/raw/imgs/marbles_" + str(image_number) + '.eps'
    win.postscript(file = filename, width=84, height=84) 
    win.close()
    # use PIL to convert to PNG 
    img = PILImage.open(filename) 
    numpy_img = np.array(img)
    img.save("./data/" + folder + "/raw/imgs/marbles_" + str(image_number) +".png")
    img.close()
    os.remove("./data/" + folder + "/raw/imgs/marbles_" + str(image_number) +".eps") 

    data.append(numpy_img)
    

np.save(file="./data/" + folder + "/raw/1k_marbles.npy", arr=data)
np.save(file="./data/" + folder + "/raw/1k_marbles_utilities.npy", arr=utils)