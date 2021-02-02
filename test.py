import os 

upsilons = ["u0", "u1000"]
betas = ["b1", "b4", "b10", "b25","b100", "b250", "b500", "b1000"]
agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]

for upsilon in upsilons: 
    for beta in betas: 
        for agent in agents:

            # python main.py u0/test -d dice --is-gen-only  
            folder = upsilon + "/dice_" + beta + "_" + agent 
            if(os.path.exists("./results/" + folder)):
                print("python main.py " + folder + " -d dice --is-gen-only")
                os.system("python main.py " + folder + " -d dice --is-gen-only")