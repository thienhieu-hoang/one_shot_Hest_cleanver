import h5py
import torch
import torch.nn as nn

import sys
import os
import numpy as np
from collections import deque
import utils

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
        
        # self.channel_estimator = nn.ModuleList([
        #     # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        #     nn.Conv2d(32, 16, kernel_size=5, padding=2), # layer 5
        #     nn.Conv2d(16, 8, kernel_size=5, padding=2),  # layer 6
        #     nn.Conv2d(8, 1, kernel_size=5, padding=2)    # layer 7
        # ])
        
        self.channel_estimator = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(32, 16, kernel_size=5, padding=2), # layer 5
            nn.Conv2d(16, 8, kernel_size=5, padding=2),  # layer 6
            nn.Conv2d(8, 1, kernel_size=5, padding=2)    # layer 7
        )
            
        if dropOut != 0:
            self.dropout = nn.Dropout(p=dropOut)

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim=1)
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
        H_est = self.channel_estimator(features)
        # Domain prediction from GRL-transformed features
        domain_pred = self.domain_classifier(features_grl.view(features_grl.size(0), -1))  # Flatten features
        
        # for i, layer in enumerate(self.channel_estimator): #layers 5-7
        #     features = layer(features)
        #     if (i+1 + len(self.feature_extractor) in self.dropOutPos) and self.dropOut: # shouldn't add
        #         features = self.dropout(features)
        # return features, domain_pred  # features at the end is the H_est
        
        return H_est, domain_pred  
    
def train_loop_CNN_DANN(model, train_loader, val_loader, 
                        target_loader, target_loader_f, optimizer, criterion, 
                        domain_criterion, device, norm_approach, lower_range, max_batches,
                        valLabels, targetLabels, targetLabels_f, num_epochs):
    train_loss  = []
    vali_loss_s = []
    vali_loss_t = []
    H_NN_val_s = torch.empty_like(valLabels) # [nVal, 2, 612, 14]
    H_NN_val_s_complx = torch.empty(valLabels.size(0), valLabels.size(2), valLabels.size(3)) # [nVal, 612, 14]

    H_NN_val_t = torch.empty_like(targetLabels) # [n2, 2, 612, 14]
    H_NN_val_t_complx = torch.empty(targetLabels.size(0), targetLabels.size(2), targetLabels.size(3)) # [n2, 612, 14]

    H_NN_test  = torch.empty_like(targetLabels_f) # [n3, 2, 612, 14]
    H_NN_test_complx = torch.empty(targetLabels_f.size(0), targetLabels_f.size(2), targetLabels_f.size(3)) # [n3, 612, 14]

    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_domain_loss = 0.0
        
        if (epoch == num_epochs-1):
            i1 = 0
            i2 = 0
            i3 = 0
        
        
        for batch_idx, ((inputs_s, labels_s, input_s_min, input_s_max), (inputs_t, _, inputs_t_min, inputs_t_max)) in enumerate(zip(train_loader, target_loader)):
        # for batch_idx in range(max_batches):
            # Compute GRL weight λ (progress-based)
            p = float(batch_idx + epoch * max_batches) / (num_epochs * max_batches)
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

            # === Train on Source Domain ===
            # inputs_s, labels_s, input_s_min, input_s_max = next(dl_source_iter)
            inputs_s, labels_s = inputs_s.to(device), labels_s.to(device)
            domain_labels_s = torch.zeros(inputs_s.size(0), dtype=torch.long).to(device)  # Label 0 for source domain

            H_preds_s, domain_preds_s = model(inputs_s, lambda_)
            loss_est_s = criterion(H_preds_s, labels_s)
            loss_domain_s = domain_criterion(domain_preds_s, domain_labels_s)

            # === Train on Target Domain ===
            # inputs_t, _, inputs_t_min, inputs_t_max = next(dl_target_iter)  # No labels for target
            inputs_t = inputs_t.to(device)
            domain_labels_t = torch.ones(inputs_t.size(0), dtype=torch.long).to(device)  # Label 1 for target domain
            _, domain_preds_t = model(inputs_t, lambda_)
            loss_domain_t = domain_criterion(domain_preds_t, domain_labels_t)

            # === Backpropagation and Optimization ===
            loss_total = loss_est_s + loss_domain_s + loss_domain_t
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss += loss_est_s.item()
            running_domain_loss += (loss_domain_s.item() + loss_domain_t.item())
            
            print(f"Batch {batch_idx+1}/{max_batches}, Estimation Loss: {loss_est_s.item()}, Domain prediction loss= {running_domain_loss} ", end='\r')
        
        print(f"Estimation Loss: {loss_est_s.item()}, Domain prediction loss= {running_domain_loss} ")
        avg_train_loss = running_loss / max_batches
        avg_domain_loss = running_domain_loss / (2 * max_batches)
        train_loss.append(avg_train_loss)
        
        # === Validation ===
        model.eval()
        running_val_loss_s = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, val_min, val_max in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
                val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
                val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
                val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)
                
                val_outputs_real, _ = model(val_inputs_real, lambda_=0)
                val_loss_real = criterion(val_outputs_real, val_targets_real)
                running_val_loss_s += val_loss_real.item()
                
                val_outputs_imag, _ = model(val_inputs_imag, lambda_=0)
                val_loss_imag = criterion(val_outputs_imag, val_targets_imag)
                running_val_loss_s += val_loss_imag.item()
                
                # save the estimated channel at the last epoch 
                # need i because we loop over batch_size
                if (epoch == num_epochs-1): 
                    # H_NN_val_s[i1:i1+val_outputs_real.size(0),0,:,:].unsqueeze(1).copy_(val_outputs_real)
                    # H_NN_val_s[i1:i1+val_outputs_imag.size(0),1,:,:].unsqueeze(1).copy_(val_outputs_imag)
                    # print(f"i1: {i1}")
                    # H_temp_denormd = utils.deNorm(H_NN_val_s[i1:i1+val_outputs_imag.size(0),:,:,:], val_min, val_max, norm_approach, lower_range=lower_range)
                    H_temp_denormd = utils.deNorm(torch.cat((val_outputs_real,val_outputs_imag), dim=1), val_min, val_max, norm_approach, lower_range=lower_range)
                    H_NN_val_s_complx[i1:i1+val_outputs_imag.size(0),:,:] = torch.complex(H_temp_denormd[:,0,:,:], H_temp_denormd[:,1,:,:])
                    i1 = i1+val_outputs_imag.size(0)
                
        avg_val_loss_s = running_val_loss_s / (len(val_loader)*2)
        vali_loss_s.append(avg_val_loss_s)
        print(f"Val Loss on Source domain: {avg_val_loss_s}", end= '\r')
        
        
        # === Validation on Target Domain ===
        model.eval()
        running_val_loss_t = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, val_min, val_max in target_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_outputs, _ = model(val_inputs, lambda_=0)
                val_loss_ = criterion(val_outputs, val_targets)
                running_val_loss_t += val_loss_.item()
                
                
                # save the estimated channel at the last epoch 
                # need i because we loop over batch_size
                if (epoch == num_epochs-1): 
                    H_NN_val_t_complx = utils.deNorm(val_outputs, val_min, val_max, norm_approach, lower_range=lower_range)
                    # print(f"i2: {i2}")
                    i2 = i2+val_outputs.size(0)
                
        avg_val_loss_t = running_val_loss_t / (len(val_loader))
        vali_loss_t.append(avg_val_loss_t)
        print(f"Val Loss on Target domain: {avg_val_loss_t}")


        # === Test on Target Domain (unseen/future data) ===
        model.eval()
        running_val_loss_t = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, val_min, val_max in target_loader_f:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
                val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
                val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
                val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)
                
                val_outputs_real, _ = model(val_inputs_real, lambda_=0)
                val_loss_real = criterion(val_outputs_real, val_targets_real)
                running_val_loss_t += val_loss_real.item()
                
                val_outputs_imag, _ = model(val_inputs_imag, lambda_=0)
                val_loss_imag = criterion(val_outputs_imag, val_targets_imag)
                running_val_loss_t += val_loss_imag.item()
                
                # save the estimated channel at the last epoch 
                # need i because we loop over batch_size
                if (epoch == num_epochs-1): 
                    # H_NN_test[i3:i3+val_outputs_real.size(0),0,:,:].unsqueeze(1).copy_(val_outputs_real)
                    # H_NN_test[i3:i3+val_outputs_imag.size(0),1,:,:].unsqueeze(1).copy_(val_outputs_imag)

                    # H_temp_denormd = utils.deNorm(H_NN_test[i3:i3+val_outputs_imag.size(0),:,:,:], val_min, val_max, norm_approach, lower_range=lower_range)
                    H_temp_denormd = utils.deNorm(torch.cat((val_outputs_real,val_outputs_imag), dim=1), val_min, val_max, norm_approach, lower_range=lower_range)
                    H_NN_test_complx[i3:i3+val_outputs_imag.size(0),:,:] = torch.complex(H_temp_denormd[:,0,:,:], H_temp_denormd[:,1,:,:])
                    # print(f"i3: {i3}")
                    i3 = i3+val_outputs_imag.size(0)
        
        avg_val_loss_t = running_val_loss_t / (len(val_loader)*2)
        vali_loss_t.append(avg_val_loss_t)
        print(f"Test Loss on Target domain: {avg_val_loss_t}")
    return train_loss, vali_loss_s, vali_loss_t, H_NN_val_s_complx, H_NN_val_t_complx, H_NN_test_complx
