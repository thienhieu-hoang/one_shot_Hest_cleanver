import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata

import sys
import os
# Get the parent directory of the current folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add the parent directory to sys.path so Python can find the helper folder
sys.path.append(parent_dir)

# to load models
from train.Pix2Pix.helperPix2Pix.discriminator_model import Discriminator
from train.Pix2Pix.helperPix2Pix.generator_model import Generator, Generator_FineTune
from helper.utils import CNN_Est2, CNN_Est, deMinMax
from helper.utils_transfer import FineTuneModel2

def linear_extrapolation_matrix(matrix):
    # Extract known columns (2 to 11)
    known_columns = matrix[:, 2:12]  # Columns 2 to 11
    
    # Extrapolate for columns 0 and 1 based on columns 2 and 3
    # Linear extrapolation for each row in columns 0 and 1
    slope_left = (known_columns[:, 1] - known_columns[:, 0])  # Slope between columns 2 and 3
    matrix[:, 0] = known_columns[:, 0] - slope_left  # Extrapolate for column 0
    matrix[:, 1] = known_columns[:, 0] - slope_left * 2  # Extrapolate for column 1

    # Extrapolate for columns 12 and 13 based on columns 10 and 11
    # Linear extrapolation for each row in columns 12 and 13
    slope_right = (known_columns[:, 9] - known_columns[:, 8])  # Slope between columns 10 and 11
    matrix[:, 12] = known_columns[:, 9] + slope_right  # Extrapolate for column 12
    matrix[:, 13] = known_columns[:, 9] + slope_right * 2  # Extrapolate for column 13

    a = matrix[:,:]
    
    return a

def helperLinearInterp(Y_noise, pilot_Indices, pilot_Symbols):
    dmrsRx = torch.empty_like(Y_noise)  # Target tensor
    row_idx = pilot_Indices[:,0] - 1 # Convert to 0-based indexing
    col_idx = pilot_Indices[:,1] - 1 # Convert to 0-based indexing
    dmrsRx[:,row_idx, col_idx] = Y_noise[:,row_idx, col_idx]  # Extract pilot symbols
    dmrsEsts = dmrsRx * torch.conj(pilot_Symbols)  # Element-wise multiplication with conjugated pilots

    # Output tensors like in MATLAB
    H_equalized = dmrsEsts.clone()
    H_linear = torch.zeros_like(Y_noise, dtype=torch.complex64)
    
    known_values = H_equalized[:, row_idx, col_idx].cpu().numpy()  # Convert to NumPy for interpolation
    row_idx = row_idx.cpu().numpy()
    col_idx = col_idx.cpu().numpy()
    
    # Generate mesh grid for interpolation
    grid_x, grid_y = np.meshgrid(np.arange(Y_noise.shape[2]), np.arange(Y_noise.shape[1]))  # Full (x, y) grid

    
    # Perform grid interpolation
    for i in range(Y_noise.shape[0]):
        interpolated = griddata(
            (col_idx, row_idx),  # Known (x, y) positions
            known_values[i,:],  # Known values
            (grid_x, grid_y),  # Target grid
            method='linear',  # Linear interpolation
            fill_value=0  # Optional: fill NaNs with 0
        )
        H_linear_temp = torch.tensor(interpolated, dtype=torch.complex64).to(H_equalized.device)
        H_linear[i,:,:] = linear_extrapolation_matrix(H_linear_temp) # extrapolation
        
    H_linear = torch.tensor(H_linear, dtype=torch.complex64).to(H_equalized.device)

    return H_equalized, H_linear

def helperLoadModels(device, snr):
    # Loading trained-only models
    model_path = '../model/static/CNN/BS16/3500_3516/ver27_/'
    if snr<0:
        LS_CNN_trained = CNN_Est().to(device)
    else:
        LS_CNN_trained = CNN_Est2().to(device)
    checkpoint = (torch.load(model_path + str(snr)+'dB/CNN_1_LS_CNN_model.pth'))
    LS_CNN_trained.load_state_dict(checkpoint["model_state_dict"])
    LS_CNN_trained.eval()
    #--------------------------------------------------
    if snr<0:
        LI_CNN_trained = CNN_Est().to(device)
    else:
        LI_CNN_trained = CNN_Est2().to(device)
    checkpoint = (torch.load(model_path + str(snr)+'dB/CNN_1_LS_LI_CNN_model.pth'))
    LS_CNN_trained.load_state_dict(checkpoint["model_state_dict"])
    LI_CNN_trained.eval()
    #--------------------------------------------------
    epoc_load = 40 # epoch to load saved models
    model_path = '../model/static/GAN/BS16/3500_3516/ver17_/'
    LS_GAN_trained = Generator().to(device)
    checkpoint = (torch.load(model_path + str(snr) + 'dB/' +str(epoc_load) + 'epoc_G_1_LS_GAN_model.pth'))
    LS_GAN_trained.load_state_dict(checkpoint["model_state_dict"])
    LS_GAN_trained.eval()
    #--------------------------------------------------
    LI_GAN_trained = Generator().to(device) 
    checkpoint = (torch.load(model_path + str(snr) + 'dB/' +str(epoc_load) + 'epoc_G_1_LS_LI_GAN_model.pth'))
    LS_GAN_trained.load_state_dict(checkpoint["model_state_dict"])
    LI_GAN_trained.eval()

    ############################################################
    # Loading transferred models
    model_path = '../transfer/transferd_model/static/'
    temp_model = CNN_Est()
    LS_CNN_transfer = FineTuneModel2(temp_model).to(device)
    if snr < 0:
        checkpoint = (torch.load(model_path + 'CNN/ver11_/' + str(snr)+'dB/LS_CNN_model.pth'))
    else:
        checkpoint = (torch.load(model_path + 'CNN/ver10_/' + str(snr)+'dB/LS_CNN_model.pth'))
    LS_CNN_transfer.load_state_dict(checkpoint["model_state_dict"])
    LS_CNN_transfer.eval()
    #--------------------------------------------------
    LI_CNN_transfer = FineTuneModel2(temp_model).to(device)
    if snr < 0:
        checkpoint = (torch.load(model_path + 'CNN/ver11_/' + str(snr)+'dB/LS_LI_CNN_model.pth'))
    else:
        checkpoint = (torch.load(model_path + 'CNN/ver10_/' + str(snr)+'dB/LS_LI_CNN_model.pth'))
    LI_CNN_transfer.load_state_dict(checkpoint["model_state_dict"])
    LI_CNN_transfer.eval()
    #--------------------------------------------------
    model_path = '../transfer/transferd_model/static/'
    temp_model = Generator()
    LS_GAN_transfer = Generator_FineTune(temp_model).to(device)
    checkpoint = (torch.load(model_path + 'GAN/ver9_/' + str(snr)+'dB/LS_GAN_G_model.pth'))
    LS_GAN_transfer.load_state_dict(checkpoint["model_state_dict"])
    LS_GAN_transfer.eval()
    #--------------------------------------------------
    LI_GAN_transfer = Generator_FineTune(temp_model).to(device)
    checkpoint = (torch.load(model_path + 'GAN/ver9_/' + str(snr)+'dB/LS_LI_GAN_G_model.pth'))
    LI_GAN_transfer.load_state_dict(checkpoint["model_state_dict"])
    LI_GAN_transfer.eval()
    
    return LS_CNN_trained, LI_CNN_trained, LS_GAN_trained, LI_GAN_trained, LS_CNN_transfer, LI_CNN_transfer, LS_GAN_transfer, LI_GAN_transfer

def helperEqualX(Y_noise, min_max, model, inputs_real, inputs_imag, device):
    X_out_real = model(inputs_real)    # 32x1x612x14
    X_out_imag = model(inputs_imag)
    X_out_output = torch.cat((X_out_real, X_out_imag), dim=1) # 32x2x612x14
    # De-normalized
    X_out_denormd = deMinMax(X_out_output, min_max[:,0], min_max[:,1])
    X_out_complex = torch.complex(X_out_denormd[:,0,:,:], X_out_denormd[:,1,:,:]).to(device)

    # Estimate X from H and Y
    X_out = Y_noise/X_out_complex
    return X_out

