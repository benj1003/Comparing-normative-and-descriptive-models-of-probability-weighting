import numpy as np 
import pandas as pd 
import os, scipy.io
import random
import matplotlib.pyplot as plt 
import sys

def trial_generation(gamble_df, g, nTrial=100, threshold=0.01):
    gamble_nr = [(g+1)]*nTrial
    choice = [np.nan]*nTrial
    a_max = [gambles_df["A1_x"][g]]*nTrial
    a_min = [gambles_df["A2_x"][g]]*nTrial
    b_max = [gambles_df["B1_x"][g]]*nTrial
    b_min = [gambles_df["B2_x"][g]]*nTrial

    accepted = False #bool check if frequency probabilities are sufficiently close to underlying probability

    while not accepted:
        data = []
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

        if abs(prob_max_a[-1] - gambles_df["A1_p"][g]) < threshold and abs(prob_max_b[-1] - gambles_df["B1_p"][g]) < threshold:
            acceepted = True
            data.append([gamble_nr,trial_count,choice, a_max,a_min,choice_a,prob_max_a,b_max,b_min,choice_b,prob_max_b])
            df = pd.DataFrame(data =data[0]) 
            df = df.transpose()
            df.columns = ['Gamble_nr','Trial_count','Choice','maxA','minA','xA', 'pMaxA','maxB','minB', 'xB', 'pMaxB']

            return df

def duplicate_trials_to_sessions(data, nSessions):
    gamble_nr = list(data['Gamble_nr'])*nSessions
    trial_count = list(data['Trial_count'])*nSessions
    choice = list(data['Choice'])*nSessions
    maxA = list(data['maxA'])*nSessions
    minA = list(data['minA'])*nSessions
    xA = list(data['xA'])*nSessions
    pMaxA = list(data['pMaxA'])*nSessions
    maxB = list(data['maxB'])*nSessions
    minB = list(data['minB'])*nSessions
    xB = list(data['xB'])*nSessions
    pMaxB = list(data['pMaxB'])*nSessions

    data = [gamble_nr,trial_count,choice, maxA,minA,xA,pMaxA,maxB,minB,xB,pMaxB]
    df = pd.DataFrame(data =data) 
    df = df.transpose()
    df.columns = ['Gamble_nr','Trial_count','Choice','maxA','minA','xA', 'pMaxA','maxB','minB', 'xB', 'pMaxB']

    metadata = [gamble_nr[0], maxA[0], minA[0], maxB[0], nSessions, len(gamble_nr)/nSessions]

    datadict = {'Choice':choice,'maxA':maxA, 'minA': minA,'xA': xA, 'p_maxA': pMaxA, 'maxB': maxB, 'minB': minB, 'xB': xB, 'p_maxB': pMaxB}

    return datadict, metadata


#making sure the path is correct
if not os.getcwd() == r'c:\Users\benja\OneDrive\Dokumenter\GitHub\Master-thesis\Python scripts':
    os.chdir(r'c:\Users\benja\OneDrive\Dokumenter\GitHub\Master-thesis\Python scripts')

#Import gambles
gambles_df = pd.read_csv('Gambles.csv', sep=";")
nGambles = gambles_df.shape[0] 

cols = ['Gamble_nr','maxA','minA','maxB','nTrials','nSessions']
df_metadata = pd.DataFrame(columns=cols, index=range(nGambles))
datadict_full = []

for g in range(nGambles):
    data = trial_generation(gambles_df, g)
    datadict, metadata = duplicate_trials_to_sessions(data, 50)
    df_metadata.loc[g] = metadata
    datadict_full.append(datadict)

df_metadata.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Parameter recovery','data',f"metadata.txt")),index=False)
scipy.io.savemat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Parameter recovery','data','all_gambles.mat')),datadict,oned_as='row')