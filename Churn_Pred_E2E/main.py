import subprocess
import sys 
import os

def install_requirements():
    flag_file = os.path.join(os.getcwd(),'.first_run')
    #print(flag_file)
    if not os.path.exists(flag_file):
        print("First run detected. Installing requirements...")
        subprocess.check_call('pip install -r requirement.txt')
        
        with open(flag_file, 'w') as f:
            f.write('Installation completed')

#install_requirements()
'''import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)