import os 


upsilons = ["500"]
betas = ["1000"] 
agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]

for upsilon in upsilons: 
    for beta in betas: 
        for agent in agents:
            print("python main.py u" + upsilon + "/dice_b" + beta + "_" + agent + " -d dice -l betaH -e 10 -u " +  upsilon + " --betaH-B " + beta + " --checkpoint-every 1 ")
            os.system("python main.py u" + upsilon + "/dice_b" + beta + "_" + agent + " -d dice -l betaH -e 10 -u " +  upsilon + " --betaH-B " + beta + " --checkpoint-every 1 ")

