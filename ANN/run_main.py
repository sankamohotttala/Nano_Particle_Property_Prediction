import subprocess
import time
import os
from utility_FFNN import  getTimeString
import json


def save_args_to_file(diction, time, folder_name,final =False):
    if final:
        dic_final ={'Time_for_hyper_tuning':time}
    else:
        dic_final ={**diction,'Time_for_training':time}
    
    file_path = os.path.join(folder_name, 'hyper_param_tuing_details.json')
    with open(file_path, 'a') as f:
        json.dump(dic_final, f, indent = 4)

# new sub folder for this 
# output_folder = r'E:\Other Projects\Nano particles - Dr. Harinda\nanoproject-visuolization-sliit\Model_training\FFNN_hyper_param_tuning' #original path
output_folder = r'F:\Codes\joint attention\Nano-particle\test_folder'
random_foldername = getTimeString()
# random_foldername = 'tmp'
new_out_path = os.path.join(output_folder, 'hyper_tuning_'+random_foldername)
if not os.path.exists(new_out_path):
    os.makedirs(new_out_path)

#device_manual = 0 when it is selected auto, 1 when assiging cpu, 2 when setting gpu
configurations_ = [
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':10,'device_manual':0},
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':10,'device_manual':1},
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':11,'device_manual':2}
    # {'batch_size': 8, 'lr_val': 0.01}
    # {'batch_size': 8, 'lr_val': 0.01},
    # {'batch_size': 8, 'lr_val': 0.01}
]
# configurations = [
#     {'param1': 'value1a', 'param2': 'value2a'},
#     {'param1': 'value1b', 'param2': 'value2b'},
#     {'param1': 'value1c', 'param2': 'value2c'},
# ]

configurations = [{**config,'hyperparam_tuning':True,'path_outputs':new_out_path} for config in configurations_]


script_path = 'FFNN_non_linear_save_csv.py'

s_t = time.time()
trial_time= []
for config in configurations:
    command = ['python39', script_path]
    for key, value in config.items():
        if type(value) != str:
            value = str(value)  
        command.extend([f'--{key}', value])
    # print('cat')
    p_s_t = time.time() 

    subprocess.run(command)

    trial_time.append(time.time() - p_s_t)

all_time = time.time() - s_t

for i ,time_ in zip(configurations,trial_time):
    save_args_to_file(i, time_, new_out_path)
save_args_to_file(i, all_time, new_out_path, True)