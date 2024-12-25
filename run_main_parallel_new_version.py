import subprocess
import concurrent.futures
import time
import os
from utility_FFNN import  getTimeString
import json
import pandas as pd

from itertools import permutations, combinations
import random
import copy
import re


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

def lr_vs_batch_size():
    l_rate = [0.5,0.1,0.05,0.01,0.005, 0.001,0.0005, 0.0001,0.00005, 0.00001]
    batch_size = [1,2,4,8,16,32,64,128,256,512] 

    tmp_dic_list = []
    for l_ra in l_rate:
        for batch_ in batch_size:
            tmp_dic_list.append({'batch_size':batch_ , 'lr_val':l_ra})

    return tmp_dic_list

def architecture_search():

    required_number = 10 # number of architectures we select at the end
    layer_node_config = [256,128,64,32,16,8,4,2]
    tmp_combin = list(combinations(layer_node_config, 4))

    all_permu = []
    for combin in tmp_combin:
        all_permu.extend(list(permutations(combin)))

    # pyramid architecture small to large
    # inverted pyramid architecture
    # other architectures
    pyramid_archi = []
    inverted_archi = []
    other_archi = []

    for perm_ in all_permu:
        tmp_changing = perm_[:]
        perm = list(perm_[:])
        asending_sort = list(tmp_changing[:])
        asending_sort.sort()

        descending_sort = list(tmp_changing[:])
        descending_sort.sort(reverse = True)

        if perm == asending_sort: # pyramid archi
            pyramid_archi.append([perm,'pyramid'])
        elif perm == descending_sort:
            inverted_archi.append([perm,'inv_pyramid'])
        else: # other mixed structures
            other_archi.append([perm,'other'])

    all_architectures = {'0':pyramid_archi, '1':inverted_archi,'2':other_archi}
    all_architectures_copy_2 = copy.deepcopy(all_architectures) # we keep this without doing any changes for later usage

    numbers = [0, 1, 2] #pyramdi, inverted_pyramid , other
    weights = [0.4, 0.4, 0.2]

    #method 1 - for all permutations giving equal weight - currently usinh method 2
    # randomlist = random.sample(range(len(all_permu)), required_number)
    # layer_architectures = [all_permu[val] for val in randomlist]

    #method 2 - with the above weight list values
    selecte_architetures = []
    for i in range(required_number):
        tmp_random_val = random.choices(numbers, weights)[0]
        tmp_permutations = all_architectures[str(tmp_random_val)]
        random_index = random.sample(range(len(tmp_permutations)), 1)[0]
        selected_archi =  tmp_permutations.pop(random_index)
        selecte_architetures.append(selected_archi)

    activation_fuc =['leaky_relu','relu','gelu','sigmoid','tanh']

    tmp_dic_list = []
    for archite in selecte_architetures:
        for activ_f in activation_fuc:
            tmp_dic_list.append({'batch_size':16 , 'lr_val':0.001, 'nodes_per_layer':archite[0], 'activation_f':activ_f,'architecture_type':archite[1]})
            
    return tmp_dic_list



# new sub folder for this 
# output_folder = r'F:\Codes\joint attention\Nano-particle\FFNN_hyper_param_tuning'
output_folder = r'F:\Codes\joint attention\Nano-particle\FFNN_hyper_param_tuning'

random_foldername = getTimeString()
# random_foldername = 'tmp'
new_out_path = os.path.join(output_folder, 'hyper_tuning_'+random_foldername)
if not os.path.exists(new_out_path):
    os.makedirs(new_out_path)




# #device_manual = 0 when it is selected auto, 1 when assiging cpu, 2 when setting gpu
# configurations_ = [
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1},
#     {'batch_size': 8, 'lr_val': 0.01,'num_epochs':2,'device_manual':1}
#     # {'batch_size': 8, 'lr_val': 0.01}
#     # {'batch_size': 8, 'lr_val': 0.01},
#     # {'batch_size': 8, 'lr_val': 0.01}
# ]




# configurations = [
#     {'param1': 'value1a', 'param2': 'value2a'},
#     {'param1': 'value1b', 'param2': 'value2b'},
#     {'param1': 'value1c', 'param2': 'value2c'},
# ]

# configurations_ = lr_vs_batch_size()
configurations_ = architecture_search()

configurations = [{**config,'hyperparam_tuning':True,'path_outputs':new_out_path} for config in configurations_]


script_path = 'FFNN_non_linear_multi_thread.py'

pattern = r'\b\d+\b'

trial_time= []

commands =[]
for config in configurations:
    command = ['python', script_path]
    for key, value in config.items():
        if type(value) != str and type(value) !=list:
            value = str(value)
            command.extend([f'--{key}', value])
        elif type(value) == list: # this conditionn atm only useful with architecture nodes per layer
            tmp =str(value)[1:-1]
            digits = re.findall(pattern, tmp)
            digits.insert(0,f'--{key}')
            command.extend(digits)
        else:
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