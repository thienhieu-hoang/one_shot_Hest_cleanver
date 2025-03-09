import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata

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
    for i in range(32):
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
