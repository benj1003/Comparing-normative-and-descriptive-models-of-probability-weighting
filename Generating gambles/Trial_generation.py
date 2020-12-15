import numpy as np 
import pandas as pd 
import os, scipy.io
import random
import matplotlib.pyplot as plt 
import sys

def trial_generation(gamble_df, g, nTrial, nAgent, threshold=1):
    gamble_nr = [(g+1)]*nTrial
    choice = [np.nan]*nTrial
    a_max = [gambles_df["A1_x"][g]]*nTrial
    a_min = [gambles_df["A2_x"][g]]*nTrial
    b_max = [gambles_df["B1_x"][g]]*nTrial
    b_min = [gambles_df["B2_x"][g]]*nTrial

    metadata = [gamble_nr[0], a_max[0], a_min[0], b_max[0], nAgent, nTrial]

    data = []
    for jj in range(nAgent):
        accepted = False #bool check if frequency probabilities are sufficiently close to underlying probability
        while not accepted:
            data_for_agent = []
            trial_count = []
            choice_a = []      #outcome in trial for choice A
            prob_max_a = []    #probability of a_max according to frequency 
            choice_b = []      #outcome in trial for choice B
            prob_max_b = []    #probability of b_max according to frequency 
            
            max_a_counter = 0
            max_b_counter = 0
            for ii in range(nTrial):
                trial_count.append(ii)

                #Choice A
                rnd = np.random.uniform(0,1,1)
                choice_a.append(gambles_df["A1_x"][g] if rnd < gambles_df["A1_p"][g] else gambles_df["A2_x"][g]) 

                if choice_a[ii] == gambles_df["A1_x"][g]:
                    max_a_counter += 1
                p_a_tmp = max_a_counter/(ii+1)
                prob_max_a.append(p_a_tmp)

                #Choice B
                choice_b.append(gambles_df["B1_x"][g] if np.random.uniform(0,1,1) < gambles_df["B1_p"][g] else gambles_df["B2_x"][g])

                if choice_b[ii] == gambles_df["B1_x"][g]:
                    max_b_counter += 1
                p_b_tmp = max_b_counter/(ii+1)
                prob_max_b.append(p_b_tmp)


            prob_a_within_threshold = abs(prob_max_a[-1] - gambles_df["A1_p"][g]) < threshold
            prob_b_within_threshold = abs(prob_max_b[-1] - gambles_df["B1_p"][g]) < threshold


            if prob_a_within_threshold and prob_b_within_threshold:
                accepted = True
                data_for_agent = {'Choice':choice,'maxA':a_max, 'minA': a_min,'xA': choice_a, 'p_maxA': prob_max_a, 'maxB': b_max, 'minB': b_min, 'xB': choice_b, 'p_maxB': prob_max_b}
                data.append(data_for_agent)
                
    return data, metadata

#Import gambles
gambles_df = pd.read_csv('Generating gambles\Gambles.csv', sep=";")
nGambles = gambles_df.shape[0] 

#Generate empty dataframe
cols = ['Gamble_nr','maxA','minA','maxB','nAgents','nTrials']
df_metadata_all_gambles = pd.DataFrame(columns=cols, index=range(nGambles))

#Generate trials
data_all_gamles = []
for g in range(nGambles):
    print(f"Completed Gamble {g+1}")
    data_one_gamble, metadata_one_gamble = trial_generation(gambles_df, g, nTrial=200, nAgent=10)
    df_metadata_all_gambles.loc[g] = metadata_one_gamble
    data_all_gamles.append(data_one_gamble)

#Save Trials
df_metadata_all_gambles.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..', 'Recovery','data',f"metadata.txt")),index=False)
scipy.io.savemat(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..', 'Recovery','data','all_gambles.mat')),{'Data':data_all_gamles},oned_as='row') 