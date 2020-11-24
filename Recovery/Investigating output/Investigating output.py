import h5py
import os
import sys
import numpy as np

file_name = 'Choices_simulated_from_CPT_Gamble_1'

with h5py.File(f'Recovery/samples_stats/{file_name}.mat', 'r') as file:
    print(list(file.keys()))

    print("\n SAMPLES:")
    ##SAMPLES
    print(list(file['samples'].keys()))

    y = file['samples'].get('y').value

    t = file['samples']


print("\nDone with:")
print("- nAgents = 2")
print("- nChunks = 1")
print("- nTrials = 3")
print("- nSamples = 50")
print("- nTrials = 4")


print(f"Shape of choice variable: {np.shape(y)}")
print()
y = t.get('y').value
print(y)
