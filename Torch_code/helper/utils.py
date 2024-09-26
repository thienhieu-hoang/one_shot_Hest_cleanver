import h5py
import torch
import torch.nn as nn

import os
import numpy as np
import scipy.io

import config
import datetime
import utils

######################
# Helping functions/classes for CNN
####################
# class CNN_Est(nn.Module):
#     def __init__(self):
#         super(CNN_Est, self).__init__()
        
#         self.normalization = nn.BatchNorm2d(1)
        
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
#         self.relu  = nn.ReLU() 
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
#         self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

#     def forward(self, x):
#         # Forward pass
#         out = self.normalization(x)
#         out = self.conv1(x)
#         out = self.relu(out)  
#         out = self.conv2(out)
#         out = self.relu(out)  
#         out = self.conv3(out)
#         out = self.relu(out)  
#         out = self.conv4(out)
#         out = self.relu(out) 
#         out = self.conv5(out)
#         return out

class CNN_Est(nn.Module): # CNN_Est with DropOut version1
    def __init__(self, dropOut = 0, act = 'ReLU'):
        super(CNN_Est, self).__init__()        
        self.normalization = nn.BatchNorm2d(1)
        self.dropOut = dropOut
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        if act == 'ReLU':
            self.activate  = nn.ReLU() 
        elif act == 'Tanh':
            self.activate  = nn.Tanh()
        elif act == 'Sigmoid':
            self.activate  = nn.Sigmoid()
        elif act == 'LeakyReLU':
            self.activate  = nn.LeakyReLU(negative_slope=0.01)
            
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        if dropOut != 0:
            self.dropout = nn.Dropout(p=dropOut)

    def forward(self, x):
        # Forward pass
        out = self.normalization(x)
        out = self.conv1(x)
        out = self.activate(out)  
        out = self.conv2(out)
        out = self.activate(out) 
        if self.dropOut != 0:
            out = self.dropout(out)
        out = self.conv3(out)
        out = self.activate(out)  
        out = self.conv4(out)
        out = self.activate(out) 
        if self.dropOut != 0:
            out = self.dropout(out)
        out = self.conv5(out)
        return out    

# Training loop # function for CNN
def train_loop(learning_rate, valLabels, val_loader, train_loader, model, NUM_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = nn.MSELoss()
    train_loss =[]
    vali_loss = []
    H_NN_val = torch.empty_like(valLabels) # [nVal, 2, 612, 14]
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        if (epoch == num_epochs-1):
            i = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss} ")
        
        # Validation 
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
                val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
                val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
                val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)
                
                val_outputs_real = model(val_inputs_real)
                val_loss_real = criterion(val_outputs_real, val_targets_real)
                running_val_loss += val_loss_real.item()
                
                val_outputs_imag = model(val_inputs_imag)
                val_loss_imag = criterion(val_outputs_imag, val_targets_imag)
                running_val_loss += val_loss_imag.item()
                
                # save the estimated channel at the last epoch 
                # need i because we loop over batch_size
                if (epoch == num_epochs-1): 
                    H_NN_val[i:i+val_outputs_real.size(0),0,:,:].unsqueeze(1).copy_(val_outputs_real)
                    H_NN_val[i:i+val_outputs_imag.size(0),1,:,:].unsqueeze(1).copy_(val_outputs_imag)
                    i = i+val_outputs_imag.size(0)
                
        avg_val_loss = running_val_loss / (len(val_loader)*2)
        vali_loss.append(avg_val_loss)    
                
        print(f" Val Loss: {avg_val_loss}")
    return train_loss, vali_loss, H_NN_val

def minmaxScaler(x, lower_range = -1):
    # lower_range = -1 -- scale to [-1 1] range
    # lower_range =  1 -- scale to [0 1] range
    x_min = []
    x_max = []
    x_normd = torch.empty(x.shape)
    for i in range(x.shape[0]):
        sample = x[i]
        
        # Compute max and min for the current sample
        min = sample.min()
        max = sample.max()
        if lower_range ==-1:
            x_normd[i,:,:,:] = (sample - min) / (max - min) *2 -1
        elif lower_range ==0:
            x_normd[i,:,:,:] = (sample - min) / (max - min)
            
        x_min.append(min.item())
        x_max.append(max.item())
        
    x_min = torch.tensor(x_min)
    x_max = torch.tensor(x_max)
    return x_normd, x_min, x_max

def deMinMax(x_normd, x_min, x_max, lower_range=-1):
    x_denormed = torch.empty(x_normd.shape)
    if lower_range ==-1:
        for i in range(x_normd.shape[0]):
            x_denormed[i,:,:,:] = (x_normd[i,:,:,:] +1) /2 *(x_max[i] -x_min[i]) + x_min[i]
    elif lower_range ==0:
        for i in range(x_normd.shape[0]):
            x_denormed[i,:,:,:] = (x_normd[i,:,:,:]) *(x_max[i] -x_min[i]) + x_min[i]
            
    return  x_denormed

def standardize(x):
    x_mean = []
    x_var = []
    x_normd = torch.empty(x.shape)
    for i in range(x.shape[0]):
        sample = x[i]
        
        # Compute mean and variance for the current sample
        mean = sample.mean()
        variance = sample.var()
        
        x_normd[i,:,:,:] = (sample - mean) / np.sqrt(variance)
        x_mean.append(mean.item())
        x_var.append(variance.item())
        
    x_mean = torch.tensor(x_mean)
    x_var = torch.tensor(x_var)
    return x_normd, x_mean, x_var

def deSTD(x_normd, mean, var):
    x_denormd = torch.empty(x_normd.shape)
    for i in range(x_normd.shape[0]):
        sample = x_normd[i]
        
        x_denormd[i,:,:,:] = sample* np.sqrt(var[i]) + mean[i]
    return x_denormd  

def deNorm(x_normd, x_1, x_2, norm_approach, lower_range=-1):
    # lower_range used in case of deMinMax
    # norm_approach = 'minmax' or 'std' 
    if norm_approach == 'minmax':
        x_denormd = deMinMax(x_normd, x_1, x_2, lower_range)
                            # x_1 -- x_min
                            # x_2 -- x_max
    elif norm_approach == 'std':
        x_denormd = deSTD(x_normd, x_1, x_2)
                        # x_1 == x_mean
                        # x_2 == x_var
    elif norm_approach == 'no':
        x_denormd = x_normd
        
    return x_denormd

def val_step(model, val_loader, criterion, epoch, num_epochs, H_NN_val):
    model.eval()
    i=0
    running_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets, val_targetsMin, val_targetsMax in val_loader:
            val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
            val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
            val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
            val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)
            
            val_outputs_real = model(val_inputs_real)
            val_loss_real = criterion(val_outputs_real, val_targets_real)
            running_val_loss += val_loss_real.item()
            
            val_outputs_imag = model(val_inputs_imag)
            val_loss_imag = criterion(val_outputs_imag, val_targets_imag)
            running_val_loss += val_loss_imag.item()
            
            if (epoch == num_epochs-1): # the results after the last training 
                H_NN_val[i:i+val_outputs_real.size(0),0,:,:].unsqueeze(1).copy_(val_outputs_real)
                H_NN_val[i:i+val_outputs_imag.size(0),1,:,:].unsqueeze(1).copy_(val_outputs_imag)
                
                i = i+val_outputs_imag.size(0)       
                
            
    avg_val_loss = running_val_loss / (len(val_loader)*2)
    return avg_val_loss, H_NN_val

def calNMSE(x, target):
    # x, target == ?x612x14 complex
    # return ?x1 array: nmse of each data sample and target sample
    NMSE_array = torch.empty(x.shape[0])
    for i in range(x.shape[0]):
        target_squared = torch.mean(torch.abs(target[i,:,:]) **2)
        mse_i = torch.mean(torch.abs(x[i,:,:] - target[i,:,:]) ** 2)
        NMSE_array[i] = mse_i/target_squared
    return NMSE_array    
    