import shutil
import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import random

from utility_FFNN import saveFile, getTimeString

def check_output_exists(path_f):
    output_path = path_f
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
def create_project_folder(path_f):
    random_foldername = getTimeString()
    random_number = random.randint(1, 100000)
    output_folder = os.path.join(path_f, random_foldername+str(random_number))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print('same folder existed !!!!!')
    return output_folder

def save_args_to_file(args, folder_name):
    # Convert the args namespace to a dictionary
    args_dict = vars(args)
    file_path = os.path.join(folder_name, 'args_.json')
    # Save the dictionary to a file in JSON format
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

def arg_parser():

    parser = argparse.ArgumentParser(description='Params and hyper-params for Nano research')

    #paths 
    output_path_ = r'F:\Codes\joint attention\Nano-particle\FFNN_outputs_tmp_new'
    input_paths_ = r'F:\Codes\joint attention\Nano-particle\output_new_new'
    
    #args added
    parser.add_argument('--path_inputs', type=str, help='input dataset path',default = input_paths_)
    parser.add_argument('--path_outputs', type=str, help='output directory', default = output_path_)
    parser.add_argument('--input_num_features', type=int, help='number of features', default = 35)
    parser.add_argument('--output_num_features', type=int, help='number of labels', default = 1)
    parser.add_argument('--processed_features', type=str, default = 'normalized_35' ,help='normalization/standardiation of featrues') ## 'normalized_35' , 'raw_35'
    parser.add_argument('--label_type', type=str, default = 'r_avg_raw'  ,help='label used')
    parser.add_argument('--batch_size', type=int, default = 16  ,help='batch size')
    parser.add_argument('--lr_val', type=float, default = 0.001  ,help='learning rate')
    parser.add_argument('--num_epochs', type=int, default = 300  ,help='number of epochs')
    parser.add_argument('--init_weights', type=str, default = 'x_n'  ,help='weight initialization') # 
    parser.add_argument('--num_hidden_layers', type=int, default = 4  ,help='number of hidden layers without output and input layers')
    parser.add_argument('--nodes_per_layer', type=int, nargs='+',default = [256, 128, 64, 16]   ,help='nodes per layer')    # [256, 128, 64, 16]
    parser.add_argument('--activation_f', type=str, default = 'relu'  ,help='activation function used')
    parser.add_argument('--hyperparam_tuning', type=bool, default = False  ,help='hyper-param tuning creates a sub folder auto')
    parser.add_argument('--device_manual', type=int, default = 1  ,help='auto setting = 0, cpu =1, gpu 2')
    parser.add_argument('--architecture_type', type=str, default = None  ,help='None if we are not interested in this: others pyramid, inverted_pyramid, other')
    # Parse the arguments
    args = parser.parse_args()
    check_output_exists(args.path_outputs)
    folder_path = create_project_folder(args.path_outputs)


    saveFile(folder_path , __file__)
    save_args_to_file(args, folder_path)

    return args , parser, folder_path