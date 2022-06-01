import math
import os

import h5py
import numpy as np
import scipy.stats as sc


def read_output(file_name,analysis):
    with h5py.File(os.path.join(os.path.dirname( __file__ ),'..','samples_stats',file_name), 'r') as file:
        if analysis == "model_comparisson":
            z = file['samples'].get('z')[:]
            return z

        if analysis == "parameter_recovery":
            alpha = file['samples'].get('alpha_pt')[:]
            beta = file['samples'].get('beta_pt')[:]
            delta = file['samples'].get('delta_pt')[:]
            gamma = file['samples'].get('gamma_pt')[:]
            return alpha, beta, delta, gamma

def process_params(param, n_agents, n_chains, n_samples, output="map"):
    dist_marginal = []
    dist_agents = []
    for i in range(n_agents):
        tmp = []
        for s in range(n_samples):
            for j in range(n_chains):
                dist_marginal.append(param[i,s,j])
                tmp.append(param[i,s,j])
        dist_agents.append(tmp)
    if output == "dist":
        return dist_marginal, dist_agents

    map_marginal = sc.mode(dist_marginal)[0][0]
    map_agent = []
    for i in range(n_agents):
        map_agent.append(sc.mode(dist_agents[i])[0][0])
    return map_marginal, map_agent

def cpt_weighting_function(x, delta, gamma):
    w = []
    for i in x:
        denumerator = delta*i**gamma + (1-i)**gamma
        w.append(delta*i**gamma/denumerator)
    return w

def lml_weighting_function(x,T):
    w = []
    for i in x:
        xn = 1-i
        n = i + math.sqrt(i/T)
        d = n + (xn+math.sqrt(xn/T))
        w.append(n/d)
    return w

def model_select(z, n_subjects, n_chains, n_samples): #NOT DONE!
    lml = 0
    cpt = 0
    z_choices_subject = []
    for i in range(n_subjects):
        lml_subject = 0
        cpt_subject = 0
        for j in range(n_chains):
            for s in range(n_samples):
                if z[i,s,j] in [1,3,5,7]:
                    cpt += 1
                    cpt_subject += 1
                else:
                    lml += 1
                    lml_subject += 1
        z_choices_subject.append([cpt_subject/(n_samples*n_chains),lml_subject/(n_samples*n_chains)])
    z_choices = [cpt/(n_subjects*n_samples*n_chains),lml/(n_subjects*n_samples*n_chains)]
    return z_choices, z_choices_subject
