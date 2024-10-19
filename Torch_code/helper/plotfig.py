import os
import matplotlib.pyplot as plt


def figLoss(train_loss=None, val_loss=None, index_save=None, figure_save_path=None, name=None, xlabel='Epoch', ylabel='Loss',
            title='Training and Validation Loss', train_loss_legend='Training Loss', 
            val_loss_legend='Validation Loss', trainD_loss_legend='Discriminator Loss',
            trainD_loss=None, GAN_mode=False):
    plt.figure(figsize=(10, 5))
    if train_loss is not None:
        if 'x' not in locals():
            x = range(1, len(train_loss) + 1) # epoch number
        plt.plot(x,train_loss, label=train_loss_legend)
    if trainD_loss is not None:
        if 'x' not in locals():
            x = range(1, len(trainD_loss) + 1) 
        plt.plot(x, trainD_loss, label=trainD_loss_legend)
    if val_loss is not None:
        if 'x' not in locals():
            x = range(1, len(val_loss) + 1) 
        plt.plot(x,val_loss, label=val_loss_legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(x)>5:
        plt.xticks(range(0, len(x) + 1, int(len(x)/5)))
    else:
        plt.xticks(range(0, len(x) + 1, 1))
    plt.legend()
    plt.savefig(os.path.join(figure_save_path,  str(index_save) + name) )
    plt.clf()
    
def figTrueChan(x, title, index_save, figure_save_path, name):
    plt.figure(figsize=(10, 5))
    plt.imshow(x,  aspect='auto', cmap='viridis', interpolation='none')
    plt.xlabel('OFDM symbol')
    plt.ylabel('Subcarrier')
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(figure_save_path,  str(index_save) + name) )
    plt.clf()
    
def figPredChan(x, title, y, index_save, figure_save_path, name):
    # x in cpu
    plt.figure(figsize=(10, 5))
    plt.imshow(x,  aspect='auto', cmap='viridis', interpolation='none')
    plt.xlabel('OFDM symbol')
    plt.ylabel('Subcarrier')
    plt.title(f'{title}, NMSE: {y:.4f}')
    plt.colorbar()
    plt.savefig(os.path.join(figure_save_path,  str(index_save) + name) )
    # plt.show()
    plt.clf()