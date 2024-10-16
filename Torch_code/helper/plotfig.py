import os
import matplotlib.pyplot as plt


def figLoss(train_loss, val_loss, index_save, figure_save_path, name, xlabel='Epoch', ylabel='Loss',
            title='Training and Validation Loss', train_loss_legend='Training Loss', 
            val_loss_legend='Validation Loss', trainD_loss_legend='Discriminator Loss',
            trainD_loss=[], GAN_mode=False):
    plt.figure(figsize=(10, 5))
    x = range(1, len(val_loss) + 1)
    plt.plot(x,train_loss, label=train_loss_legend)
    if GAN_mode:
        plt.plot(x, trainD_loss, label=trainD_loss_legend)
    plt.plot(x,val_loss, label=val_loss_legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(0, len(val_loss) + 1, int(len(val_loss)/5)))
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