import shutil
import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import torch
# import portalocker


def saveFile(directory = '',  file=None ):
    destination_directory = directory

    current_script_path = os.path.realpath(file)

    destination_file_path = os.path.join(destination_directory, os.path.basename(current_script_path))

    shutil.copy(current_script_path, destination_file_path)

    print(f'Script copied to {destination_file_path}')



def getTimeString():
    current_datetime = datetime.now()

    datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

    return datetime_string

def MAPD_percentage(actual, predicted):
    tmp = 0
    for act_val , pred_val in zip(actual,predicted):
        tmp += (np.abs(act_val - pred_val)/np.abs(act_val))
    mapd_percentage = (tmp/len(actual))*100
    return mapd_percentage


def create_CSV_for_outputs(output_path, arguments = None): # original function
    
    metrics = ['train','train_mse', 'train_rmse', 'train_mae', 'train_mapd', 'train_r2_squared', 'validation','val_mse', 'val_rmse',
                'val_mae', 'val_mapd', 'val_r2_squared','test', 'test_mse', 'test_rmse', 'test_mae', 'test_mapd', 'test_r2_squared',
                'time complexity','training time','validation time', 'test time','hyper-parameters','batch_size','lr_val','num_epochs','init_weights','num_hidden_layers',
                'nodes_per_layer','activation','other parameters','feature_processing','label_type','input_feature_size','output_size',
                'input_path','output_path','device']
    column_names = ['diretory_path']
    column_names.extend(metrics)

    file_path = os.path.join(output_path,'outputs.csv')
    if arguments is None or arguments.architecture_type is None: # we keep argument None situation for backward compatibility
        df = pd.DataFrame(columns=column_names)
        if os.path.exists(file_path):
            print(f"The file '{file_path}' already exists.")
            df = pd.read_csv(file_path)  
            return df, file_path
        else:
            df.to_csv(file_path, index=False)
            return df, file_path
    
    elif arguments.architecture_type in ['others pyramid', 'inverted_pyramid', 'other']:
        column_names.append('architecture_type')
        df = pd.DataFrame(columns=column_names)
        if os.path.exists(file_path):
            print(f"The file '{file_path}' already exists.")
            df = pd.read_csv(file_path)  
            return df, file_path
        else:
            df.to_csv(file_path, index=False)
            return df, file_path
    else:
        ValueError('architecture type is not valid!!')

def device_select(number):
    if number == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif number == 1:
        device = 'cpu'
    elif number ==2:
        device = 'cuda'
    else:
        ValueError

    return device