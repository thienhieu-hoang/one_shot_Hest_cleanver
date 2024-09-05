in the _params.mat and _variable.pth files: 

    params = {   
                'SNR': snr,
                'epoc': NUM_EPOCHS,
                'rows': rowss,
                'learning_rate': learning_rate,
                'train_track_LI': train_loss,
                'val_track_LI': val_loss,
                'train_track_LS',
                'val_track_LS'
    }
    variables = {             
                'train_track_LI': train_loss,
                'val_track_LI': val_loss,
                'train_min_LI': trainData_min.cpu(),  # this should be named as 
                'train_max_LI': trainData_max.cpu(),  #                         
                'train_label_min': trainLabels_min.cpu(),     # this should be named as 'train_label_min'
                'train_label_max': trainLabels_max.cpu(),     #                         'train_label_max'
                'NMSE_LI': nmse_LI.cpu(),
                'NMSE_LI_NN': nmse_LI_NN.cpu(),
                
                'train_track_LS': train_loss,
                'val_track_LS': val_loss,
                'train_min_LS',
                'train_max_LS',
                'NMSE_LS_NN': nmse_LS_NN.cpu()
    }