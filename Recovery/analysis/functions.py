import h5py
import os
import numpy as np

def read_output(file_name,analysis):
    with h5py.File(os.path.join(os.path.dirname( __file__ ),'..','samples_stats',file_name), 'r') as file:
        if analysis == "model_comparisson":
            z = file['samples'].get('z').value
            return z

        if analysis == "parameter_recovery":
            alpha_cpt = file['samples'].get('alpha_pt').value
            alpha_lml = file['samples'].get('alpha_lml').value
            beta_cpt = file['samples'].get('beta_pt').value
            beta_lml = file['samples'].get('beta_lml').value
            delta = file['samples'].get('delta_pt').value
            gamma = file['samples'].get('gamma_pt').value
            return alpha_cpt, alpha_lml, beta_cpt, beta_lml, delta, gamma

def process_params(param, n_chunks, n_subjects, n_chains, n_samples, output="map"):
    dist_marginal = [] #marginal distribution for delta over all subjects and chunks
    dist_chunks = [] #marginal distribution for delta over all subjects (within each chunk)
    dist_subjects = [] #distribution for delta within each chunk and subject
    for c in range(n_chunks):
        tmp_chunk = []
        for i in range(n_subjects):
            tmp_subject = []
            for s in range(n_samples):
                for j in range(n_chains):
                    dist_marginal.append(param[c,i,s,j])

                    tmp_subject.append(param[c,i,s,j])
            tmp_chunk.append(tmp_subject)
        dist_subjects.append(tmp_chunk)
        dist_chunks.append(tmp_subject)

        if output=="distributions":
            return dist_marginal, dist_subjects, dist_chunks
    
    map_marginal = []
    map_chunks = []
    map_subjects = []

    hist, bin_edges = np.histogram(dist_marginal, bins=500)
    map_marginal.append(bin_edges[np.argmax(hist)])
    for c in range(n_chunks):
        tmp = []
        hist, bin_edges = np.histogram(dist_chunks[c], bins=500)
        map_chunks.append(bin_edges[np.argmax(hist)])
        for i in range(n_subjects):
            hist, bin_edges = np.histogram(dist_subjects[c][i], bins=500)
            tmp.append(bin_edges[np.argmax(hist)])
        map_subjects.append(tmp)

    return map_marginal, map_chunks, map_subjects

def cpt_weighting_function(x, delta, gamma):
    w = []
    for i in x:
        denumerator = delta*i**gamma + delta*(1-i)**gamma
        w.append(delta*i**gamma/denumerator)
    return w


def model_select(z, n_subjects, n_chains, n_samples):
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