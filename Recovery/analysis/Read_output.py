import h5py
import os

def read_output(file_name,analysis):
    with h5py.File(os.path.join(os.path.dirname( __file__ ),'samples_stats',file_name), 'r') as file:
        # if analysis == "model_comparisson":
        #     z = file['samples'].get('z').value
        #     return z

        if analysis == "parameter_recovery":
            alpha_cpt = file['samples'].get('alpha_pt').value
            alpha_lml = file['samples'].get('alpha_lml').value
            beta_cpt = file['samples'].get('beta_pt').value
            beta_lml = file['samples'].get('beta_lml').value
            delta = file['samples'].get('delta_pt').value
            gamma = file['samples'].get('gamma_pt').value
            return alpha_cpt, alpha_lml, beta_cpt, beta_lml, delta, gamma