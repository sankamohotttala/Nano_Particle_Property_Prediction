import subprocess
import concurrent.futures
import time
import os
from utility_FFNN import  getTimeString
import json
import pandas as pd

def save_args_to_file(diction, time, folder_name,final =False):
    if final:
        dic_final ={'Time_for_hyper_tuning':time}
    else:
        dic_final ={**diction,'Time_for_training':time}
    
    file_path = os.path.join(folder_name, 'hyper_param_tuing_details.json')
    with open(file_path, 'a') as f:
        json.dump(dic_final, f, indent = 4)

def process_folders(root_path):
    all_data = []  
    column_names = None

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == 'outputs.csv':
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path)

                if column_names is None:
                    column_names = df.columns.tolist()

                all_data.append(df)
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(os.path.join(root_path, 'final_outputs.csv'), index=False)

def run_subprocess(command):
    subprocess.run(command)

# new sub folder for this 
# output_folder = r'F:\Codes\joint attention\Nano-particle\FFNN_hyper_param_tuning'
output_folder = r'F:\Codes\joint attention\Nano-particle\test_folder\run_main_parallel'
random_foldername = getTimeString()
# random_foldername = 'tmp'
new_out_path = os.path.join(output_folder, 'hyper_tuning_' + random_foldername)
if not os.path.exists(new_out_path):
    os.makedirs(new_out_path)

#device_manual = 0 when it is selected auto, 1 when assiging cpu, 2 when setting gpu
configurations_ = [
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1}
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
    # {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1}
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


script_path = 'FFNN_non_linear_multi_thread.py'


trial_time= []

commands =[]
for config in configurations:
    command = ['python39', script_path]
    for key, value in config.items():
        if type(value) != str:
            value = str(value)  
        command.extend([f'--{key}', value])
    commands.append(command)
    trial_time.append(-1) #dummy values

s_t = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_subprocess, commands)
    print('cat1')
print('cat2')
e_t = time.time() 
all_time = e_t - s_t

for i ,time_ in zip(configurations,trial_time):
    save_args_to_file(i, time_, new_out_path)
save_args_to_file(i, all_time, new_out_path, True)

#single final_output.csv at the root  by combining all output.csv files
process_folders(new_out_path)