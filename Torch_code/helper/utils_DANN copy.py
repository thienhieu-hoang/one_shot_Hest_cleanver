import h5py
import torch
import torch.nn as nn

import sys
import os
import numpy as np
from collections import deque

# Gradient Reversal Layer (GRL) implementation
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_  # Reverse the gradient with lambda_
        return grad_input, None

class CNN_DANN(nn.Module): # CNN_Est2 with Domain-Adversarial Neural Network (DANN) 
    def __init__(self, dropOut = 0, act = 'ReLU', dropOutPos = [2,4]):
        # dropOutPos: positions to add DropOut Layer
        #   example: dropOutPos=[3,5] -> DropOut Layers after conv3 and conv5
        super(CNN_DANN, self).__init__()        
        self.normalization = nn.BatchNorm2d(1)
        self.dropOut = dropOut
        self.dropOutPos = dropOutPos
        
        if act == 'ReLU':
            self.activate  = nn.ReLU() 
        elif act == 'Tanh':
            self.activate  = nn.Tanh()
        elif act == 'Sigmoid':
            self.activate  = nn.Sigmoid()
        elif act == 'LeakyReLU':
            self.activate  = nn.LeakyReLU(negative_slope=0.01)
        self.activate_tanh = nn.Tanh()

        self.feature_extractor = nn.ModuleList([
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(1, 64, kernel_size=9, padding=4),  # layer 1
            nn.Conv2d(64, 64, kernel_size=5, padding=2), # layer 2
            nn.Conv2d(64, 64, kernel_size=5, padding=2), # layer 3
            nn.Conv2d(64, 32, kernel_size=5, padding=2), # layer 4
        ])

        self.num_cnn_features = 32*612*14   # 32 layers of 612x14 matrix
        
        self.channel_estimator = nn.ModuleList([
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(32, 16, kernel_size=5, padding=2), # layer 5
            nn.Conv2d(16, 8, kernel_size=5, padding=2),  # layer 6
            nn.Conv2d(8, 1, kernel_size=5, padding=2)    # layer 7
        ])
            
        if dropOut != 0:
            self.dropout = nn.Dropout(p=dropOut)

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x, lambda_=1.0):
        # Forward pass
        features = self.normalization(x)
            
        for i, layer in enumerate(self.feature_extractor): # layers 1-4
            features = layer(features)
            features = self.activate(features)
            if (i+1 in self.dropOutPos) and self.dropOut:
                features = self.dropout(features)
        
        # Apply Gradient Reversal Layer (GRL)
        features_grl = GradientReversalFn.apply(features, lambda_)
        
        for i, layer in enumerate(self.channel_estimator): #layers 5-7
            H_est = layer(features_grl)
            if (i+1 + len(self.feature_extractor) in self.dropOutPos) and self.dropOut: # shouldn't add
                H_est = self.dropout(H_est)
        
        # Domain prediction from GRL-transformed features
        domain_pred = self.domain_classifier(features_grl.view(features_grl.size(0), -1))  # Flatten features
        
        
        return H_est, domain_pred
    
    
def train_loop(learning_rate, valLabels, val_loader, train_loader, model, NUM_EPOCHS):
    # Create a deque to store the latest 5 prints
    latest_prints = deque(maxlen=5)
    
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
        latest_prints.append(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss} ")
        
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
                
        latest_prints.append(f" Val Loss: {avg_val_loss}")
        
        # Clear previous output
        sys.stdout.write("\033[F" * 6)  # Move cursor up (5 lines + 1 for spacing)
        sys.stdout.write("\033[J")  # Clear below

        # Print the latest 5 messages
        for msg in latest_prints:
            print(msg)
    return train_loss, vali_loss, H_NN_val

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

def train_loop(learning_rate, valLabels, val_loader, train_source_loader, train_target_loader, model, domain_discriminator, NUM_EPOCHS):
    latest_prints = deque(maxlen=5)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # For channel estimation
    domain_criterion = nn.CrossEntropyLoss()  # For domain classification (binary)
    
    train_loss = []
    vali_loss = []
    H_NN_val = torch.empty_like(valLabels)  # [nVal, 2, 612, 14]

    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        model.train()
        domain_discriminator.train()
        
        running_loss = 0.0
        running_domain_loss = 0.0
        
        dl_source_iter = iter(train_source_loader)
        dl_target_iter = iter(train_target_loader)

        max_batches = min(len(train_source_loader), len(train_target_loader))

        for batch_idx in range(max_batches):
            # Compute GRL weight λ (progress-based)
            p = float(batch_idx + epoch * max_batches) / (num_epochs * max_batches)
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

            # === Train on Source Domain ===
            inputs_s, targets_s = next(dl_source_iter)
            domain_labels_s = torch.zeros(inputs_s.size(0), dtype=torch.long)  # Label 0 for source

            class_preds_s, domain_preds_s = model(inputs_s, lambda_)
            loss_class_s = criterion(class_preds_s, targets_s)
            loss_domain_s = domain_criterion(domain_preds_s, domain_labels_s)

            # === Train on Target Domain ===
            inputs_t, _ = next(dl_target_iter)  # No labels for target
            domain_labels_t = torch.ones(inputs_t.size(0), dtype=torch.long)  # Label 1 for target

            _, domain_preds_t = model(inputs_t, lambda_)
            loss_domain_t = domain_criterion(domain_preds_t, domain_labels_t)

            # === Backpropagation and Optimization ===
            loss_total = loss_class_s + loss_domain_s + loss_domain_t
            optimizer.zero_grad()
            optimizer_domain.zero_grad()
            loss_total.backward()
            optimizer.step()
            optimizer_domain.step()

            running_loss += loss_class_s.item()
            running_domain_loss += (loss_domain_s.item() + loss_domain_t.item())

        avg_train_loss = running_loss / max_batches
        avg_domain_loss = running_domain_loss / (2 * max_batches)
        train_loss.append(avg_train_loss)

        latest_prints.append(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}, Domain Loss: {avg_domain_loss}")

        # === Validation ===
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
                val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
                val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
                val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)

                val_outputs_real = model(val_inputs_real, lambda_)[0]
                val_outputs_imag = model(val_inputs_imag, lambda_)[0]

                val_loss_real = criterion(val_outputs_real, val_targets_real)
                val_loss_imag = criterion(val_outputs_imag, val_targets_imag)

                running_val_loss += (val_loss_real.item() + val_loss_imag.item())

        avg_val_loss = running_val_loss / (len(val_loader) * 2)
        vali_loss.append(avg_val_loss)
        latest_prints.append(f"Val Loss: {avg_val_loss}")

        # Display latest 5 prints dynamically
        sys.stdout.write("\033[F" * 6)  # Move cursor up (5 lines + 1 for spacing)
        sys.stdout.write("\033[J")  # Clear below
        for msg in latest_prints:
            print(msg)

    return train_loss, vali_loss, H_NN_val
