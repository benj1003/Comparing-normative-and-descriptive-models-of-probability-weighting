import numpy as np 
import pandas as pd 
import os, scipy.io
import random
import matplotlib.pyplot as plt 
import sys

def trial_generation(gambles_df, g, n_TRIAL, n_SESSIONS, permute=False, save=False):
    gamble_nr = [(g+1)]*n_TRIAL
    gamble_nr_s = gamble_nr*n_SESSIONS
    data_one_session = []
    data_multiple_sessions = []
    session_count = []
    trial_count = []
    trial_count_s = []
    a_max = [gambles_df["A1_x"][g]]*n_TRIAL
    a_min = [gambles_df["A2_x"][g]]*n_TRIAL
    b_max = [gambles_df["B1_x"][g]]*n_TRIAL
    b_min = [gambles_df["B2_x"][g]]*n_TRIAL
    a_max_s = a_max*n_SESSIONS
    a_min_s = a_min*n_SESSIONS
    b_max_s = b_max*n_SESSIONS
    b_min_s = b_min*n_SESSIONS
    tmp_a = []
    tmp_a_s = []
    tmp_a_p = []
    tmp_a_p_s = []
    tmp_b = []
    tmp_b_s = []
    tmp_b_p = []
    tmp_b_p_s = []
    

    counter_a = 0
    counter_b = 0
    for ii in range(n_SESSIONS):
        session_count += [ii+1]*n_TRIAL
        if ii == 0:
            for jj in range(n_TRIAL):
                trial_count.append(jj+1)
                tmp_a.append(gambles_df["A1_x"][g] if np.random.uniform(0,1,1) < gambles_df["A1_p"][g] else gambles_df["A2_x"][g]) 
                
                if tmp_a[jj] == gambles_df["A1_x"][g]:
                    counter_a += 1
                tmp_a_p.append(counter_a/(jj+1))

                tmp_b.append(gambles_df["B1_x"][g] if np.random.uniform(0,1,1) < gambles_df["B1_p"][g] else gambles_df["B2_x"][g])
                if tmp_b[jj] == gambles_df["B1_x"][g]:
                    counter_b += 1
                tmp_b_p.append(counter_b/(jj+1))
        if permute:
            tmp_a = random.sample(tmp_a,len(tmp_a))
            tmp_a_p = random.sample(tmp_a_p,len(tmp_a_p))
            tmp_b = random.sample(tmp_b,len(tmp_b))
            tmp_b_p = random.sample(tmp_b_p,len(tmp_b_p))
        tmp_a_s += tmp_a 
        tmp_a_p_s += tmp_a_p 
        tmp_b_s += tmp_b
        tmp_b_p_s += tmp_b_p

        trial_count_s += trial_count

        data_multiple_sessions.append([gamble_nr_s,session_count,trial_count_s, a_max_s,a_min_s,tmp_a_s,tmp_a_p_s,b_max_s,b_min_s,tmp_b_s,tmp_b_p_s])
        df_all_session = pd.DataFrame(data =data_multiple_sessions[0]) 
        df_all_session = df_all_session.transpose()
        df_all_session.columns = ['Gamble_nr','Session_count','Trial_count','maxA','minA','xA', 'pA','maxB','minB', 'xB', 'pB']

        data_one_session.append([gamble_nr,[1]*n_TRIAL,trial_count, a_max,a_min,tmp_a,tmp_a_p,b_max,b_min,tmp_b,tmp_b_p])
        df_one_session = pd.DataFrame(data =data_one_session[0]) 
        df_one_session = df_one_session.transpose()
        df_one_session.columns = ['Gamble_nr','Session_count','Trial_count','maxA','minA','xA', 'pA','maxB','minB', 'xB', 'pB']

        datadict = {'maxA':a_max_s, 'minA': a_min_s,'xA': tmp_a_s, 'p_maxA': tmp_a_p_s, 'maxB': b_max_s, 'minB': b_min_s, 'xB': tmp_b_s, 'p_maxB': tmp_b_p_s}

        df_metadata = pd.DataFrame(data={"Gamble": (g+1), "N_sessions": n_SESSIONS, "N_Trials": n_TRIAL, "Permute": permute}, index=[0])

        if save:
            name = f'gamble_{g}_all_sessions_permuted={permute}.mat'
            scipy.io.savemat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Simulations','data',name)),datadict,oned_as='row')
            df_metadata.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Simulations','data',f"Gamble_{g}_0_metadata.txt")))
            df_one_session.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Simulations','data',f"Gamble_{g}_1_One_session.txt")))
            df_all_session.to_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Simulations','data',f"Gamble_{g}_1_all_sessions.txt")))
    
    return df_metadata, df_one_session, df_all_session

#making sure the path is correct
if not os.getcwd() == "c:/Users/benja/OneDrive/Dokumenter/DTU\Master Thesis/Python scripts":
    os.chdir("c:/Users/benja/OneDrive/Dokumenter/DTU\Master Thesis/Python scripts")

#Init variables
n_TRIAL = 5
n_SESSIONS = 3
g = 0
permute = False
save = True

#read the file with the gambles
gambles_df = pd.read_csv('Gambles.csv', sep=";")

#run function
Meta,df_one,df_multi = trial_generation(gambles_df, g, n_TRIAL, n_SESSIONS, permute=permute, save=save)

