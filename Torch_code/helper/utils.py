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
class CNN_Est(nn.Module):
    def __init__(self):
        super(CNN_Est, self).__init__()
        
        self.normalization = nn.BatchNorm2d(1)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        self.relu  = nn.ReLU() 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        # Forward pass
        out = self.normalization(x)
        out = self.conv1(x)
        out = self.relu(out)  
        out = self.conv2(out)
        out = self.relu(out)  
        out = self.conv3(out)
        out = self.relu(out)  
        out = self.conv4(out)
        out = self.relu(out) 
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

def minmaxScaler(x):
    x_min = []
    x_max = []
    x_normd = torch.empty(x.shape)
    for i in range(x.shape[0]):
        sample = x[i]
        
        # Compute max and min for the current sample
        min = sample.min()
        max = sample.max()
        
        x_normd[i,:,:,:] = (sample - min) / (max - min) *2 -1
        x_min.append(min.item())
        x_max.append(max.item())
    return x_normd, x_min, x_max

def deMinMax(x_normd, x_min, x_max):
    x_denormed = torch.empty(x_normd.shape)
    for i in range(x_normd.shape[0]):
        x_denormed[i,:,:,:] = (x_normd[i,:,:,:] +1) /2 *(x_max -x_min) + x_min
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
    return x_normd, x_mean, x_var