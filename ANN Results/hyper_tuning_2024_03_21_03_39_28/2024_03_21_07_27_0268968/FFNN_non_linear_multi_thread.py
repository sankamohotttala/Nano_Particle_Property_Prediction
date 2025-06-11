#we attempt to develop the original model here
# we add the training , test and validation times as well - Done
# we addded the csv file saving part as well for hyper-parameter tuning -Done
# we add argument based hyperparams + params and then the optionn to run multiple times - Done
# multiple processors / threads  - Done
# GPU training - Done

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
from math import ceil
import logging
import random 

from utility_FFNN import saveFile , MAPD_percentage, create_CSV_for_outputs, device_select
from setup_ import arg_parser

print(torch.__version__)

# initialize with a random value
torch.manual_seed(seed = 42)
args, parser, folder_path = arg_parser()

device = device_select(args.device_manual)
print(f"Training on {device}")



# ----setup other parameters here
show_figures = False
histogram_visualize = False #display and/or save the feature and label histograms

#output folder initialize
# output_path = r'E:\Other Projects\Nano particles - Dr. Harinda\nanoproject-visuolization-sliit\Model_training\FFNN_outputs_tmp'
output_path = args.path_outputs


# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# random_foldername = getTimeString()
# output_folder = os.path.join(output_path, random_foldername)
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

output_folder = folder_path
hist_folder = os.path.join(output_folder, 'weight_hists')
if not os.path.exists(hist_folder):
    os.makedirs(hist_folder)

# logger initialization
logging.basicConfig(filename = os.path.join(output_folder, "log_file.log"), 
					format='%(asctime)s %(message)s', 
					filemode='w') 
logger=logging.getLogger('FFNN_non_linear_save_csv') 
logger.setLevel(logging.DEBUG) 

csv_df, csv_path = create_CSV_for_outputs(output_folder) # iitializing the dataframe for csv or loading the previously used csv as a dataframe

# logger.debug("This is just a harmless debug message") 
# logger.info("This is just an information for you") 
# logger.warning("OOPS!!!Its a Warning") 
# logger.error("Have you try to divide a number by zero") 
# logger.critical("The Internet is not working....") 

# input and output dim
input_num_features = args.input_num_features #35
output_size = args.output_num_features #1

#define hyper-parameters
batch_size_ = args.batch_size #original 8
init_weights = args.init_weights # 'x_n'
lr_val = args.lr_val # 0.01
num_epochs = args.num_epochs #original 300 --- use this value

#architecture hyper-parameters
# num_hidden_layers = 8 # altogether there are num_hidden_layers + 1 since {innput and  output }
# nodes_per_layer = [512, 256, 128 , 64, 32 ,16 ,8 ,4] #if no hidden layers  then leave this as a empty list

num_hidden_layers = args.num_hidden_layers  # 4
nodes_per_layer = args.nodes_per_layer #[256, 128, 64, 16] 
activation_f = args.activation_f #'relu'  # possible values => {'leaky_relu', 'relu', 'gelu', 'sigmoid', 'tanh'}

assert len(nodes_per_layer) == num_hidden_layers


features__ = args.processed_features        #'normalized_35' # 'normalized_35' , 'raw_35'
labels__ = args.label_type      #'r_avg_raw' # 'r_avg_raw' , 'n_bulk_raw', 'n_surface_raw'



#save this file for later usage
saveFile(output_folder , __file__)


# ## Load dataset
# path_datasets = r'E:\Other Projects\Nano particles - Dr. Harinda\nanoproject-visuolization-sliit\Model_training\output_new'
# path_datasets = r'E:\Other Projects\Nano particles - Dr. Harinda\nanoproject-visuolization-sliit\output_new_new'
path_datasets = args.path_inputs
file_path_features = os.path.join(path_datasets , 'features_raw_35.npy')
file_path_normalized_features = os.path.join(path_datasets , 'features_normalized_35.npy')
file_path_standardize_features = os.path.join(path_datasets , 'features_standardized_35.npy')


file_path_label_r_avg = os.path.join(path_datasets , 'label_r_avg_raw.npy')
file_path_label_n_bulk = os.path.join(path_datasets , 'label_n_bulk_raw.npy')
file_path_label_n_surface = os.path.join(path_datasets , 'label_n_surface_raw.npy')
# file_path_features_stand = os.path.join(path_datasets , 'features_selected_35_standardized.npy')
# file_path_features_normalize = os.path.join(path_datasets , 'normalized_features_zero_one_range.npy')

with open( file_path_features , 'rb') as f:
    features = np.load(f)
with open( file_path_label_r_avg , 'rb') as f:
    labels_r_avg = np.load(f)
with open( file_path_label_n_bulk , 'rb') as f:
    labels_n_bulk = np.load(f)
with open( file_path_label_n_surface , 'rb') as f:
    labels_n_surface = np.load(f)
with open( file_path_standardize_features , 'rb') as f:
    features_stand = np.load(f)
with open( file_path_normalized_features , 'rb') as f:
    features_normal = np.load(f)

print(features.shape)
print(labels_r_avg.shape)
print(labels_n_bulk.shape)
# print(features_stand.shape)
print(features_normal.shape)


# ## Visualize data
if histogram_visualize:
    data = features
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(nrows=1, ncols=35, figsize=(70, 10))
    for i in range(35):
        axs[i].hist(data[:, i], bins=30, color='blue', alpha=0.7)
        axs[i].set_title(f'Feature {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'raw_features_35.png'))
    if show_figures:
        plt.show()

    data = features_stand
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(nrows=1, ncols=35, figsize=(70, 10))
    for i in range(35):
        axs[i].hist(data[:, i], bins=30, color='blue', alpha=0.7)
        axs[i].set_title(f'Feature {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'srandardized_features_35.png'))
    if show_figures:
        plt.show()


    data = features_normal
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(nrows=1, ncols=35, figsize=(70, 10))
    for i in range(35):
        axs[i].hist(data[:, i], bins=30, color='blue', alpha=0.7)
        axs[i].set_title(f'Feature {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'normalized_features_35.png'))
    if show_figures:
        plt.show()


    data = labels_r_avg
    plt.figure(figsize=(10, 6))
    plt.hist(data[:, 0], bins=30, color='blue', alpha=0.7)
    plt.title('Feature 1')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_folder, 'r_avg_raw_label.png'))
    if show_figures:
        plt.show()


    data = labels_n_bulk
    plt.figure(figsize=(10, 6))
    plt.hist(data[:, 0], bins=30, color='blue', alpha=0.7)
    plt.title('Feature 1')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_folder, 'n_bulk_raw_label.png'))
    if show_figures:
        plt.show()

    data = labels_n_surface
    plt.figure(figsize=(10, 6))
    plt.hist(data[:, 0], bins=30, color='blue', alpha=0.7)
    plt.title('Feature 1')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_folder, 'n_surface_raw_label.png'))
    if show_figures:
        plt.show()


# ## Split the dataset into train , test and validation splits
 
features = features_normal

if features__ == 'normalized_35':
    features = features_normal
elif features__ == 'raw_35':
    features = features # this doesn't work! Exploding gradient issue i think 
else:
    ValueError('not a valid feature set!')

if labels__ == 'r_avg_raw':
    labels = labels_r_avg
elif labels__ == 'n_bulk_raw':
    labels = labels_n_bulk
elif labels__ == 'n_surface_raw':
    labels = labels_n_surface 
else:
    ValueError('not a valid label set!')



features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(f"features train shape: {features_train.shape}")
print(f"features test shape: {features_test.shape}") # tmp 
print(f"labels train shape: {labels_train.shape}")
print(f"labes test shape: {labels_test.shape}") # tmp

print(f"features train type: {type(features_train)}")
print(f"features test type: {type(features_test)}")
print(f"labels train type: {type(labels_train)}")
print(f"labes test type: {type(labels_test)}")

#test set again split into actual test and validation sets
features_validation, features_test, labels_validation, labels_test = train_test_split(features_test, labels_test, test_size=0.5, random_state=42)

print(f"features validation shape: {features_validation.shape}")
print(f"features test shape: {features_test.shape}")
print(f"labels validation shape: {labels_validation.shape}")
print(f"labes test shape: {labels_test.shape}")

print(f"features validation type: {type(features_validation)}")
print(f"features test type: {type(features_test)}")
print(f"labels validation type: {type(labels_validation)}")
print(f"labes test type: {type(labels_test)}")

# Convert data to PyTorch tensors
features_train_t = torch.tensor(features_train, dtype=torch.float32)
labels_train_t = torch.tensor(labels_train, dtype=torch.float32)  
features_val_t = torch.tensor(features_validation, dtype=torch.float32)
labels_val_t = torch.tensor(labels_validation, dtype=torch.float32)  
features_test_t = torch.tensor(features_test, dtype=torch.float32)
labels_test_t = torch.tensor(labels_test, dtype=torch.float32) 

# Create Dataset objects
train_data_t = TensorDataset(features_train_t, labels_train_t)
val_data_t = TensorDataset(features_val_t, labels_val_t)
test_data_t = TensorDataset(features_test_t, labels_test_t)

#creating dataloader objects
train_loader = DataLoader(dataset=train_data_t, batch_size=batch_size_, shuffle=True)
val_loader = DataLoader(dataset=val_data_t, batch_size=batch_size_, shuffle=False)
test_loader = DataLoader(dataset=test_data_t, batch_size=batch_size_, shuffle=False)

print('train batches in dataset: ', len(train_loader))
print('val batches in dataset: ', len(val_loader))
print('test batches in dataset: ', len(test_loader))

logger.info(f'train batches in dataset: {len(train_loader)}' ) 
logger.info(f'val batches in dataset: {len(val_loader)}' ) 
logger.info(f'test batches in dataset: {len(test_loader)}' ) 

# Model Training
class FFNN_non_linear(nn.Module):
    def __init__(self, input_size = 35, output_size =1, init_weights = 'x_u', n_hidd_layers = None, nodes = None, activation = 'relu'):
        super(FFNN_non_linear, self).__init__()
        # self.layer_names = []
#         self.fc1 = nn.Linear(input_size, 64)
        self.num_hidden_layers = n_hidd_layers
        self.nodes = nodes
        self.init_w =init_weights
        self.in_size = input_size
        self.out_size = output_size

        self.channels_tuple = self.layer_utility()

        self.net_core = nn.ModuleList([])

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.activation_dic = {'leaky_relu':nn.LeakyReLU(), 'relu': nn.ReLU(), 'gelu':nn.GELU(), 'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh()}
        self.final_act = self.activation_dic[activation]

        #create model architecture
        for in_c, out_c in self.channels_tuple:
            self.net_core.append(nn.Linear(in_c, out_c))

        self.net_core = nn.Sequential(*self.net_core)

        self.initialize_weights()


    def forward(self, x):
        #more than 1 layer
        for i in range(len(self.net_core)-1):        
            x = self.net_core[i](x)
            x = self.final_act(x)
        x = self.net_core[-1](x) #last layer has no activations
        return x
    
    def initialize_weights(self):
        if self.init_w == 'x_u':
            for layer in self.net_core:
                init.xavier_uniform_(layer.weight)
                init.uniform_(layer.bias)  
        elif self.init_w == 'x_n':
            for layer in self.net_core:
                init.xavier_normal_(layer.weight)
                init.normal_(layer.bias)            
        elif self.init_w == 'ones':
            for layer in self.net_core:
                init.ones_(layer.weight)
                init.ones_(layer.bias)
        elif self.init_w == 'zeros':
            for layer in self.net_core:
                init.zeros_(layer.weight)
                init.zeros_(layer.bias)                                              
        else:
            # do not initialize here ; do it in using a different function
            pass
        
    def layer_utility(self):
        layer_nodes = self.nodes[:] # nodes in the hidden layers
        layer_nodes.append(self.out_size)
        layer_nodes.insert(0, self.in_size)

        layers_channels =[]
        for ind in range( self.num_hidden_layers+1):
            layers_channels.append((layer_nodes[:-1][ind] , layer_nodes[1:][ind]))

        return layers_channels

# Initialize the model
input_size = features_train.shape[1]  
assert input_size == input_num_features
model = FFNN_non_linear(input_size, output_size, init_weights,num_hidden_layers,nodes_per_layer,activation_f)

#pass model to the device (gpu/cpu)
model = model.to(device)

# initialize loss function
criterion = nn.MSELoss() 
# criterion = nn.L1Loss()

#initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=lr_val)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_val, momentum=0.9)


print(vars(model))


# Model parameters are calculated here
trainable_params = []
non_trainable_params = []
name_trainable_list = []
name_non_trainable_list = []

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name} - size: {param.size()}")
        logger.info(f"Trainable parameter: {name} - size: {param.size()}" ) 
        trainable_params.append(param)
        name_trainable_list.append(name)
    else:
        print(f"Non-trainable parameter: {name} - size: {param.size()}")
        logger.info(f"Non-trainable parameter: {name} - size: {param.size()}" )
        non_trainable_params.append(param)
        name_non_trainable_list.append(name)

# Calculate the total number of trainable and non-trainable parameters
num_trainable_params = sum(p.numel() for p in trainable_params) # numel()  returns the total number of elements in a tensor
num_non_trainable_params = sum(p.numel() for p in non_trainable_params)

print(f"Number of trainable parameters: {num_trainable_params}")
print(f"Number of non-trainable parameters: {num_non_trainable_params}")

logger.info(f"Number of trainable parameters: {num_trainable_params}")
logger.info(f"Number of non-trainable parameters: {num_non_trainable_params}")


def getParameters(model):
    trainable_params = []
    name_trainable_list = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            name_trainable_list.append(name)

    return trainable_params , name_trainable_list

# use this only if plots are less than 3 
# num_plots = len(trainable_params)
# col = 3
# row =ceil(num_plots/3)

# fig, axs = plt.subplots(row, col)

# for i in range(num_plots):
#     c_val = i % 3
#     r_val = i // 3
#     axs[ c_val].hist(trainable_params[i].detach().numpy().flatten(), bins='auto')
#     axs[ c_val].set_title(f' {name_trainable_list[i]}')
#     axs[ c_val].set_xlabel('Value')
#     axs[ c_val].set_ylabel('Frequency')

# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, 'weight_initialization.png'))
# if show_figures:
#     plt.show()




#histogram visualization of the trainable parameters after initialization
# use this if num of plots are higher than 3

def weight_plot(weights, weight_names, initial =False, epoch = -1):
    num_plots = len(weights)
    col = 3
    row =ceil(num_plots/3)

    fig, axs = plt.subplots(row, col)

    for i in range(num_plots):
        c_val = i % 3
        r_val = i // 3
        axs[r_val, c_val].hist(weights[i].cpu().detach().numpy().flatten(), bins='auto')
        axs[r_val, c_val].set_title(f' {weight_names[i]}')
        axs[r_val, c_val].set_xlabel('Value')
        axs[r_val, c_val].set_ylabel('Frequency')

    plt.tight_layout()
    if initial:
        plt.savefig(os.path.join(output_folder, 'weight_initialization.png'))
        if show_figures:
            plt.show()
    else:
        plt.savefig(os.path.join(hist_folder, f'epoch_{epoch}.png'))


weight_plot(trainable_params , name_trainable_list, True)

print(input_size)
print(output_size)
print(num_trainable_params)


# Training loop
train_losses = []  # to store training loss each epoch
val_losses = []    # to store validation loss each epoch
learning_rates = []  # to store learning rate each epoch

train_start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    predicted_val_train =[]
    actual_val_train =[]
    for inputs, labels in train_loader:
        #pass data to device (cpu/gpu)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # pdb.set_trace()
        outputs = model(inputs)
        # loss = criterion(outputs.view(-1), labels)
        loss = criterion(outputs, labels)
        #if we go for multi-class labeling, then add them here as well ex: loss = loss_1 + loss_2 + loss_3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        actual_val_train.extend([val.cpu().numpy()[0] for val in labels])
        predicted_val_train.extend([val.cpu().detach().numpy()[0] for val in outputs])

    
    mse_train_WL = mean_squared_error(actual_val_train, predicted_val_train) #sanity check for MSE - check if this is the same as running loss
    mae_train_WL = mean_absolute_error(actual_val_train, predicted_val_train)
    r2_squared_train_WL = r2_score(actual_val_train, predicted_val_train)
    rmse_train_WL = np.sqrt(mse_train_WL)
    mapd_train_WL = MAPD_percentage(actual_val_train , predicted_val_train)


    #validation
    model.eval()
    val_loss = 0.0
    predicted_val_valida =[]
    actual_val_valida =[]
    with torch.no_grad():
        for inputs, labels in val_loader:

            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            # loss = criterion(outputs.view(-1), labels)
            loss = criterion(outputs, labels)
            #if we go for multi-class labeling, then add them here as well ex: loss = loss_1 + loss_2 + loss_3
            val_loss += loss.item()

            actual_val_valida.extend([val.cpu().numpy()[0] for val in labels])
            predicted_val_valida.extend([val.cpu().numpy()[0] for val in outputs])
    
    # Store losses
    train_losses.append(running_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
#     learning_rates.append(scheduler.get_last_lr()[0])  # store the learning rate
    
    # Step the scheduler
#     scheduler.step()
    
    mse_valid_WL = mean_squared_error(actual_val_valida, predicted_val_valida) #sanity check for MSE - check if this is the same as running loss
    mae_valid_WL = mean_absolute_error(actual_val_valida, predicted_val_valida)
    r2_squared_valid_WL = r2_score(actual_val_valida, predicted_val_valida)
    rmse_valid_WL = np.sqrt(mse_valid_WL)
    mapd_valid_WL = MAPD_percentage(actual_val_valida , predicted_val_valida)
    

    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f} ------ Train = MSE:{mse_train_WL:.4f} , RMSE:{rmse_train_WL:.4f} ," 
            f"MAE:{mae_train_WL:.4f} , MAPD:{mapd_train_WL:.4f}, R_squared:{r2_squared_train_WL:.4f} ------ Validation = MSE:{mse_valid_WL:.4f} , RMSE:{rmse_valid_WL:.4f} ," 
            f"MAE:{mae_valid_WL:.4f} , MAPD:{mapd_valid_WL:.4f}, R_squared:{r2_squared_valid_WL:.4f}")
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}  \n\tTrain = MSE:{mse_train_WL:.6f} , RMSE:{rmse_train_WL:.6f} , "
            f"MAE:{mae_train_WL:.6f} , MAPD:{mapd_train_WL:.6f}, R_squared:{r2_squared_train_WL:.6f}  \n\tValidation = MSE:{mse_valid_WL:.6f} , RMSE:{rmse_valid_WL:.6f} , "
            f"MAE:{mae_valid_WL:.6f} , MAPD:{mapd_valid_WL:.6f}, R_squared:{r2_squared_valid_WL:.6f}\n" )
    if epoch % 10 == 0:
        params_ , p_names_ = getParameters(model)
        weight_plot(params_, p_names_, False, epoch+1)

train_end_time = time.time()
train_time = train_end_time - train_start_time

print(f"Time for training the model: {train_time:.4f}\n")
logger.info(f"Time for training the model: {train_time:.6f}" )

# Plotting the training and validation loss
plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'train_val_loss.png'))
if show_figures:
    plt.show()



# Plotting the training loss
# plt.plot(train_losses, label='Training loss')
plt.figure()
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'val_loss.png'))
if show_figures:
    plt.show()



# Plotting the validation loss
plt.figure()
plt.plot(train_losses, label='Training loss')
# plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'train_loss.png'))
if show_figures:
    plt.show()



#-------------------evaluation of the Model---------------------------

model.eval()

trial_dict = {'train':train_loader, 'val':val_loader, 'test':test_loader}
metrics = ['mse', 'rmse', 'mae', 'mapd', 'r2-squared']
time_calc = {'train':train_time, 'val':0, 'test':0}

all_csv_row_vals =[]

for trial , dataset_loader in trial_dict.items():

    start_time = time.time()
    actual_values = []
    predicted_values = []

    with torch.no_grad():
        for inputs, labels in dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            actual_values.append(labels.cpu().numpy())
            predicted_values.append(outputs.cpu().numpy())

    actual_values = np.concatenate(actual_values).flatten()
    predicted_values = np.concatenate(predicted_values).flatten()

    # Calculate R^2 and RMSE
    r2 = r2_score(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mapd = MAPD_percentage(actual_values, predicted_values)

    print(f"MSE for {trial}: {mse:.4f}")
    print(f"RMSE for {trial}: {rmse:.4f}")
    print(f"MAE for {trial}: {mae:.4f}")
    print(f"MAPD for {trial}: {mapd:.4f}")
    print(f"R^2 Score for {trial}: {r2:.4f}")

    logger.info(f"MSE for {trial}: {mse:.6f}" )
    logger.info(f"RMSE for {trial}: {rmse:.6f}" )
    logger.info(f"MAE for {trial}: {mae:.6f}" )
    logger.info(f"MAPD for {trial}: {mapd:.6f}" )
    logger.info(f"R^2 Score for {trial}: {r2:.6f}" )

    end_time = time.time()
    time_cal = end_time - start_time
    if trial !='train':
        time_calc[trial] = time_cal 

    print(f"Inference time for {trial} dataset: {time_cal:.4f}\n")
    logger.info(f"Inference time for {trial} dataset: {time_cal:.6f}\n" )

    metrics_list = [f'{trial}',f'{mse:.8f}',f'{rmse:.8f}', f'{mae:.8f}', f'{mapd:.8f}', f'{r2:.8f}']
    all_csv_row_vals.extend(metrics_list)
# ,f'{time_calc['train']:.8f}',f'{time_calc['val']:.8f}',f'{time_calc['test']:.8f}
all_csv_row_vals.append('time complexity')    
all_csv_row_vals.append(time_calc['train'])
all_csv_row_vals.append(time_calc['val'])
all_csv_row_vals.append(time_calc['test'])
# print('\n')

all_csv_row_vals.append('hyper-parameters')
hyper_params = [batch_size_, lr_val,num_epochs,init_weights,num_hidden_layers,nodes_per_layer,activation_f] 
all_csv_row_vals.extend(hyper_params)

all_csv_row_vals.append('other-parameters')
other_params = [features__ ,labels__ , input_num_features,output_size, path_datasets , output_path]
all_csv_row_vals.extend(other_params)

all_csv_row_vals.append(device)

all_csv_row_vals.insert(0,output_folder)


csv_df = csv_df.append(pd.Series(all_csv_row_vals, index=csv_df.columns), ignore_index=True)
csv_df.to_csv(csv_path, index=False)


params_ , p_names_ = getParameters(model)
weight_plot(params_, p_names_, False, 'final')

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(actual_values, predicted_values, alpha=0.5)
plt.title("Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

min_val = min(np.min(actual_values), np.min(predicted_values))
max_val = max(np.max(actual_values), np.max(predicted_values))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.grid(True)
plt.savefig(os.path.join(output_folder, 'actual_vs_predicted.png'))
if show_figures:
    plt.show()



#residual plot
residuals = actual_values - predicted_values

plt.figure(figsize=(8, 6))
plt.scatter(predicted_values, residuals, color='blue', marker='.', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'residual_plot.png'))
if show_figures:
    plt.show()



print('train label average: ',labels_train.sum(0)/ labels_train.shape[0])
print('test label average: ',labels_test.sum(0)/ labels_test.shape[0])
print('validation label average: ',labels_validation.sum(0)/ labels_validation.shape[0])
print('predicted (on validation dataset) values average: ',predicted_values.sum(0)/ predicted_values.shape[0])

logger.info(f"Train label average: , {labels_train.sum(0)/ labels_train.shape[0]}" )
logger.info(f"Test label average: , {labels_test.sum(0)/ labels_test.shape[0]}" )
logger.info(f"validation label average: , {labels_validation.sum(0)/ labels_validation.shape[0]}" )
logger.info(f"predicted (on validation dataset) values average: , {predicted_values.sum(0)/ predicted_values.shape[0]}" )

