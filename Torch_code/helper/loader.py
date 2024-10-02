import torch
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, TensorDataset


import config
import utils
import utils_GAN



def load_data(outer_file_path, rows, fc, device, snr):
    H_true = np.empty((0, 2, 612, 14)) # true channel

    H_equal = np.empty((0, 2, 612, 14)) # noisy channel # LS channel
    H_linear = np.empty((0, 2, 612, 14)) # noisy channel # LS+Linear Interpolated channel
    H_practical = np.empty((0, 2, 612, 14)) # noisy channel # Practical Estimated channel

    # read data from ifferent .mat file, then concatenate them
    for i in range(len(rows)):
        file_path_partial = 'Gan_'+ str(snr)+'_dBOutdoor1_' + fc +'_1ant_612subcs_Row_' + rows[i][0] +'_' + rows[i][1] + '.mat'

        file_path = os.path.join(outer_file_path, file_path_partial)
        file_path = os.path.normpath(file_path)
        file = h5py.File(file_path, 'r')
        
        H_true = np.concatenate((H_true, np.array(file['H_data'])), axis = 0) # N_samples x channel(2) x height(614) x width(14)
        H_equal = np.concatenate((H_equal, np.array(file['H_equalized_data'])), axis = 0)
        H_linear = np.concatenate((H_linear, np.array(file['H_linear_data'])), axis=0)
        if 'H_practical_data' in file:
            H_practical = np.concatenate((H_practical, np.array(file['H_practical_data'])), axis=0)
        
    shuffle_order = np.random.permutation(H_true.shape[0]);
    H_true = torch.tensor(H_true[shuffle_order])
    H_equal = torch.tensor(H_equal[shuffle_order])
    H_linear = torch.tensor(H_linear[shuffle_order])
    if 'H_practical_data' in file:
        H_practical = torch.tensor(H_practical[shuffle_order])
    
    train_size = np.floor(H_linear.shape[0]*0.9) //config.BATCH_SIZE *config.BATCH_SIZE
    train_size = int(train_size)
    
    # [samples, 2, 612, 14]
    # Split into training and validation sets for H_NN training
    trainLabels = H_true[0:train_size,:,:,:].to(device, dtype=torch.float)
    valLabels = H_true[train_size:,:,:,:].to(device, dtype=torch.float)
    
    H_equal_train   = H_equal[0:train_size,:,:,:].to(device, dtype=torch.float)         # after LS
    H_linear_train   = H_linear[0:train_size,:,:,:].to(device, dtype=torch.float)       # after LS + linear interpolation
    if 'H_practical_data' in file:
        H_practical_train = H_practical[0:train_size,:,:,:].to(device, dtype=torch.float)   # using Matlab practical estimation
    
    # Split H_equal, H_linear, H_practical for validation later
    H_equal_val = H_equal[train_size:,:,:,:].to(device, dtype=torch.float)
    H_linear_val = H_linear[train_size:,:,:,:].to(device, dtype=torch.float)
    if 'H_practical_data' in file:
        H_practical_val = H_practical[train_size:,:,:,:].to(device, dtype=torch.float)
    
    return [trainLabels, valLabels], [H_equal_train, H_linear_train, H_practical_train], [H_equal_val, H_linear_val, H_practical_val]

def load_map_data(outer_file_path, device, snr, train_rate=0.9):
    H_true = np.empty((0, 2, 612, 14)) # true channel

    H_equal = np.empty((0, 2, 612, 14)) # noisy channel # LS channel
    H_linear = np.empty((0, 2, 612, 14)) # noisy channel # LS+Linear Interpolated channel
    H_practical = np.empty((0, 2, 612, 14)) # noisy channel # Practical Estimated channel

    # read data from .mat file
    file_path_partial = str(snr)+'dB/1_mapBaseData.mat'

    file_path = os.path.join(outer_file_path, file_path_partial)
    file_path = os.path.normpath(file_path)
    file = h5py.File(file_path, 'r')
    
    H_true = np.concatenate((H_true, np.array(file['H_data'])), axis = 0) # N_samples x channel(2) x height(614) x width(14)
    H_equal = np.concatenate((H_equal, np.array(file['H_equalized_data'])), axis = 0)
    H_linear = np.concatenate((H_linear, np.array(file['H_linear_data'])), axis=0)
    if 'H_practical_data' in file:
        H_practical = np.concatenate((H_practical, np.array(file['H_practical_data'])), axis=0)
        
    shuffle_order = np.random.permutation(H_true.shape[0]);
    H_true = torch.tensor(H_true[shuffle_order])
    H_equal = torch.tensor(H_equal[shuffle_order])
    H_linear = torch.tensor(H_linear[shuffle_order])
    if 'H_practical_data' in file:
        H_practical = torch.tensor(H_practical[shuffle_order])
    
    train_size = np.floor(H_linear.shape[0]* train_rate) //config.BATCH_SIZE *config.BATCH_SIZE
    train_size = int(train_size)
    
    # [samples, 2, 612, 14]
    # Split into training and validation sets for H_NN training
    trainLabels = H_true[0:train_size,:,:,:].to(device, dtype=torch.float)
    valLabels = H_true[train_size:,:,:,:].to(device, dtype=torch.float)
    
    H_equal_train   = H_equal[0:train_size,:,:,:].to(device, dtype=torch.float)         # after LS
    H_linear_train   = H_linear[0:train_size,:,:,:].to(device, dtype=torch.float)       # after LS + linear interpolation
    H_practical_train = torch.empty_like(H_linear_train)
    if 'H_practical_data' in file:
        H_practical_train = H_practical[0:train_size,:,:,:].to(device, dtype=torch.float)   # using Matlab practical estimation
    
    # Split H_equal, H_linear, H_practical for validation later
    H_equal_val = H_equal[train_size:,:,:,:].to(device, dtype=torch.float)
    H_linear_val = H_linear[train_size:,:,:,:].to(device, dtype=torch.float)
    H_practical_val = torch.empty_like(H_linear_val)
    if 'H_practical_data' in file:
        H_practical_val = H_practical[train_size:,:,:,:].to(device, dtype=torch.float)
    
    return [trainLabels, valLabels], [H_equal_train, H_linear_train, H_practical_train], [H_equal_val, H_linear_val, H_practical_val]


def loader_dataset(H_linear_train_normd, H_true_train_nomrd, H_linear_val_normd, H_true_val_normd):
    # Split real and imaginary grids into 2 image sets, then concatenate
    trainData_normd   = torch.cat((H_linear_train_normd[:,0,:,:], H_linear_train_normd[:,1,:,:]), dim=0).unsqueeze(1)  # 612 x 14 x (Nsamples*2)
    trainLabels_normd = torch.cat((H_true_train_nomrd[:,0,:,:], H_true_train_nomrd[:,1,:,:]), dim=0).unsqueeze(1)  # 612 x 14 x (Nsamples*2)

    trainData_normd.shape

    # Create a DataLoader for dataset
    dataset = TensorDataset(trainData_normd, trainLabels_normd)  # [6144, 1, 612, 14]
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(H_linear_val_normd, H_true_val_normd)  # [367, 2, 612, 14]
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def load_model(params):
    generator_li = utils_GAN.Generator(in_channel=params['in_channel'])   # in_channel=1 to estimate real and imag parts separately,
                                                                # default: in_channel=2 to estimate real and imag parts at the same time
    discriminator_li = utils_GAN.Discriminator(params['in_channel'])
    generator_ls = utils_GAN.Generator(in_channel=params['in_channel'])   # in_channel=1 to estimate real and imag parts separately,
                                                                # default: in_channel=2 to estimate real and imag parts at the same time
    discriminator_ls = utils_GAN.Discriminator(params['in_channel'])
    if params['load_saved_model']:
        # modify the directory
        params['epoc'] = params['epoc_saved_model']
        
        variable_load_path = os.path.join(config.FILE_PATH, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_variable_'+params['rowss']+'.pth')
        var_state = torch.load(variable_load_path)
                
        # load for (LS+LI) model
        generator_li_load_path = os.path.join(config.FILE_PATH, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_generator_'+params['rowss']+'.pth')
        discriminator_li_load_path = os.path.join(config.FILE_PATH, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_discriminator_'+params['rowss']+'.pth')
        # load the models
        generator_li.load_state_dict(torch.load(generator_li_load_path))
        discriminator_li.load_state_dict(torch.load(discriminator_li_load_path))
        
        
        # load for LS model
        generator_ls_load_path = os.path.join(config.FILE_PATH, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_generator_'+params['rowss']+'.pth')
        discriminator_ls_load_path = os.path.join(config.FILE_PATH, 'model/static', 'GAN_'+str(params['snr'])+'dB_epoch_'+str(params['epoc'])+'_discriminator_'+params['rowss']+'.pth')
        # load the models
        generator_ls.load_state_dict(torch.load(generator_ls_load_path))
        discriminator_ls.load_state_dict(torch.load(discriminator_ls_load_path))

    if params['load_saved_model']:
        gen_li_loss_track = var_state['gen_li_loss_track']
        disc_li_loss_track = var_state['disc_li_loss_track']
        gen_li_val_loss_track = var_state['gen_li_val_loss_track']
        gen_ls_loss_track = var_state['gen_ls_loss_track']
        disc_ls_loss_track = var_state['disc_ls_loss_track']
        gen_ls_val_loss_track = var_state['gen_ls_val_loss_track']
    else: 
        gen_li_loss_track  = []    # BCE loss in training
        disc_li_loss_track = []    # BCE loss in training
        gen_li_val_loss_track = [] # MSE _ compare estimated and true channels
        gen_ls_loss_track  = []    # BCE loss in training
        disc_ls_loss_track = []    # BCE loss in training
        gen_ls_val_loss_track = [] # MSE _ compare estimated and true channels
    return [generator_li, discriminator_li], [generator_ls, discriminator_ls], [gen_li_loss_track, disc_li_loss_track, gen_li_val_loss_track], [gen_ls_loss_track, disc_ls_loss_track, gen_ls_val_loss_track]

# save to .mat file
def find_incremental_filename(directory, prefix_name, postfix_name, extension='.mat'):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out files that match the pattern prefix_name + number + postfix_name + extension
    existing_files = [f for f in files if f.startswith(prefix_name) and f.endswith(postfix_name + extension)]
    
    # Extract the numbers from the filenames
    numbers = []
    for f in existing_files:
        # Strip the prefix and postfix, then extract the number in between
        try:
            number_part = f[len(prefix_name):-len(postfix_name + extension)]
            if number_part.isdigit():
                numbers.append(int(number_part))
        except ValueError:
            pass  # Skip any files that don't match the expected pattern
    
    # Determine the next number
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1  # Start numbering from 1 if no existing files  
    return next_number

def genLoader(data, target, BATCH_SIZE, device, mode, shuff, approach, lower_range=-1):
        # mode = 'train' or 'valid'
        # approach = 'minmax', 'std', or 'no'
        #   in 'minmax' case: x-min;  y-max
        #                       lower_range = -1  -- scale to [-1 1] range
        #                       lower_range =  0  -- scale to  [0 1] range
        #   in    'std' case: x-mean; y-var 

    # 1.2 Normalization Min-Max scaler
    if approach == 'minmax':
        data_normd,   data_x, data_y  = utils.minmaxScaler(data, lower_range)
        label_normd, label_x, label_y = utils.minmaxScaler(target, lower_range)
                    # x: min
                    # y: max
    elif approach == 'std':
        data_normd,   data_x, data_y  = utils.standardize(data)
        label_normd, label_x, label_y = utils.standardize(target)
                    # x: mean
                    # y: var
        # data_normd  == torch.tensor N_samples(data) x 2 x 612 x 14
        # label_normd == torch.tensor N_samples(label) x 2 x 612 x 14
        # label/data x, y == torch.tensor  N_samples (data/target) x 2 - min/max/mean/var of real/imag of each sample
    elif approach == 'no':
        data_normd  = data
        label_normd = target
        label_x = torch.zeros(target.shape[0])
        label_y = torch.zeros(target.shape[0])
    
    if mode == 'train':
        # Split real and imaginary grids into 2 image sets, then concatenate
        data_normd  = torch.cat((data_normd[:,0,:,:], data_normd[:,1,:,:]), dim=0).unsqueeze(1)  # 612 x 14 x (Nsamples*2)
        label_normd = torch.cat((label_normd[:,0,:,:], label_normd[:,1,:,:]), dim=0).unsqueeze(1)  # 612 x 14 x (Nsamples*2)
        label_x     = torch.cat((label_x[:,0], label_x[:,1]), dim=0)
        label_y     = torch.cat((label_y[:,0], label_y[:,1]), dim=0)
        # label x, y == torch.tensor  N_samples (data/target)*2 x 1 - min/max/mean/var of concatenated real-imag of each sample
        
    data_normd  = data_normd.to(device, dtype=torch.float)
    label_normd = label_normd.to(device, dtype=torch.float)
    label_x     = label_x.to(device, dtype=torch.float)
    label_y     = label_y.to(device, dtype=torch.float)

    # 1.3 Create a DataLoader for dataset
    dataset = TensorDataset(data_normd, label_normd, label_x, label_y)  # [4224, 1, 612, 14]
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuff)

    return loader, label_x, label_y
