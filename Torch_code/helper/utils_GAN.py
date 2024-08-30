import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
############
## Helping functions/classes for GAN
## input 1x612x14, output 1x612x14
############
class ShrinkLayer(nn.Module):
    def __init__(self, filters, kernel_size, in_channels=1, stride=2, apply_batchnorm=True, add=False, padding=1):
        super(ShrinkLayer, self).__init__()
        # initializer = nn.init.normal_(mean=0., std=0.02)
        
        # conv = nn.Conv2d(filters=filters, kernel_size=kernel_size, strides=strides_s,
        #                      padding=padding_s, kernel_initializer=initializer, use_bias=False)
        conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=False)
        ac = nn.LeakyReLU(0.2)

        components = [conv]
        if apply_batchnorm:
            components.append(nn.BatchNorm2d(filters))
            components.append(ac)
        
        self.encoder_layer = nn.Sequential(*components)

    def forward(self, x):
        return self.encoder_layer(x)


class EnlargeLayer(nn.Module):
    def __init__(self, filters, kernel_size, in_channels=1, stride=2, apply_dropout=False, add=False, padding=1):
        super(EnlargeLayer, self).__init__()
        # initializer = nn.init.normal_(mean=0., std=0.02)
        # dconv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
                                    #    padding='same', kernel_initializer=initializer, use_bias=False)
        dconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        bn = nn.BatchNorm2d(filters)
        ac = nn.ReLU()

        components = [dconv, bn]
        if apply_dropout:
            components.append(nn.Dropout(p=0.5))
        components.append(ac)
        
        self.decoder_layer = nn.Sequential(*components)

    def forward(self, x):
        return self.decoder_layer(x)

class Conv1DNet(nn.Module):
    def __init__(self,in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=0):
        super(Conv1DNet, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU()  

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        return x

class Conv1DTransposeNet(nn.Module):
    def __init__(self,in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=0):
        super(Conv1DTransposeNet, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU()  

    def forward(self, x):
        x = self.conv1d_transpose(x)
        x = self.activation(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channel=2):   # inchannel = 2 --> estimate both real and imag parts at the same time
                                        # inchannel = 1 --> estimate real and imag parts separately
        super(Generator, self).__init__()
        
        # Resize Input
        self.prep_layers = nn.ModuleList([
            EnlargeLayer(in_channels=in_channel,filters=2, kernel_size=4, stride=(2,5), add=True, padding=(4,1)),
            ShrinkLayer(in_channels=2,filters=2, kernel_size=(4,5), add=True, padding=0)
        ])

        # Encoder
        self.encoder_layers = nn.ModuleList([
            ShrinkLayer(in_channels=2,filters=64*1, kernel_size=4, apply_batchnorm=False),
            ShrinkLayer(in_channels=64*1, filters=64*2, kernel_size=4),
            ShrinkLayer(in_channels=64*2, filters=64*4, kernel_size=4),
            ShrinkLayer(in_channels=64*4, filters=64*8, kernel_size=4),
            ShrinkLayer(in_channels=64*8, filters=64*8, kernel_size=4)
        ])
        
        self.transfer_layers = nn.ModuleList([
            Conv1DNet(),
            Conv1DTransposeNet()
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            EnlargeLayer(in_channels=64*8, filters=64*8, kernel_size=4, apply_dropout=True),
            EnlargeLayer(in_channels=64*8*2, filters=64*8, kernel_size=4, apply_dropout=True),
            EnlargeLayer(in_channels=64*8+64*4, filters=64*8, kernel_size=4, apply_dropout=True),
            EnlargeLayer(in_channels=64*8+64*2, filters=64*4, kernel_size=4)
        ])
        
        # Resize output
        self.post_layers = nn.ModuleList([
            EnlargeLayer(in_channels=2, filters=in_channel, kernel_size=(4,5), stride=2, add=True, padding=0),
            ShrinkLayer(in_channels=in_channel,filters=in_channel, kernel_size=4, stride=(2,5), add=True, padding=(4,1))
        ])

        # inconv = nn.Conv2d(filters=filters, kernel_size=kernel_size, strides=strides_s,
        #                      padding=padding_s, kernel_initializer=initializer, use_bias=False)
        
        self.last = nn.ConvTranspose2d(in_channels=256+64, out_channels=2, kernel_size=4, stride=2, padding=1)
        

    def forward(self, x):
        # Pass the encoder and record xs
        for p_layer in self.prep_layers:
            x = p_layer(x)

        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)

        encoder_xs = encoder_xs[:-1][::-1]  # reverse
        
        for trans_layer in self.transfer_layers:
            x = trans_layer(x.squeeze())
            
        x= torch.unsqueeze(x, dim=-1)

        # Pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            x = torch.cat([x, encoder_xs[i]], dim=1)  # skip connect

        x = self.last(x)
        x = nn.Tanh()(x)
        
        for p_layer in self.post_layers:
            x = p_layer(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channel=2):   # inchannel = 2 --> estimate both real and imag parts at the same time
                                        # inchannel = 1 --> estimate real and imag parts separately
        
        super(Discriminator, self).__init__()
        # initializer = tf.random_normal_initializer(0., 0.02)
        # Resize Input
        self.prep_layers = nn.ModuleList([
            EnlargeLayer(in_channels=in_channel,filters=2, kernel_size=4, stride=(2,5), add=True, padding=(4,1)),
            ShrinkLayer(in_channels=2,filters=2, kernel_size=(4,5), add=True, padding=0)
        ])
        
        self.encoder_layer_1 = ShrinkLayer(in_channels= 2, filters=64, kernel_size=4, apply_batchnorm=False)
        self.encoder_layer_2 = ShrinkLayer(in_channels= 64, filters=128, kernel_size=4)
        self.encoder_layer_3 = ShrinkLayer(in_channels= 128, filters=128, kernel_size=4)

        self.zero_pad1 = nn.ZeroPad2d(padding=1)
        self.conv = nn.Conv2d(in_channels= 128, out_channels=512, kernel_size= 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.ac = nn.LeakyReLU()

        self.zero_pad2 = nn.ZeroPad2d(padding=1)
        self.last = nn.Conv2d(in_channels= 512, out_channels= 1, kernel_size= 4, stride=1, padding=0)
        
        # Resize output
        self.post_layer = ShrinkLayer(in_channels=1, filters=1, kernel_size=(8,1), stride=(6,1), padding=0)

    def forward(self, x):
        
        for p_layer in self.prep_layers:
            x = p_layer(x)
        
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        
        x = self.post_layer(x)
        
        return x

def discriminator_loss(disc_real_output, disc_generated_output):
    criterion = nn.BCEWithLogitsLoss()
    
    loss_disc_real = criterion(disc_real_output, torch.ones_like(disc_real_output))
    loss_disc_fake = criterion(disc_generated_output, torch.zeros_like(disc_generated_output))
    
    total_disc_loss = (torch.mean(loss_disc_real) + torch.mean(loss_disc_fake)) / 2
    
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):    
    criterion = nn.BCEWithLogitsLoss()
    loss_gen = criterion(disc_generated_output, torch.ones_like(disc_generated_output))
    
    # Calculate L2 loss
    # l2_loss = torch.mean(torch.abs(target - gen_output))
    
    # Calculate total generator loss
    total_gen_loss = torch.mean(loss_gen)  # + l2_weight * l2_loss
    
    return total_gen_loss


def generated_image(generator, test_input, target, t=0):
        # test_input == Nsamples x 2 x subcs x symb
        # target     == Nsamples x 2 x subcs x symb
    prediction = generator(test_input)  # H_fake of antenna 'ant' == Nsamples x subcs x symb x 2  
    display_list = [np.squeeze(test_input[:, :, :, 0]), np.squeeze(target[:, :, :, 0]), np.squeeze(prediction[:, :, :, 0])]
        # real part
    title = ['Input H_interpolated', 'Target H', 'Predicted H']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.savefig(os.path.join("generated_img", "img_"+str(t)+".png"))


def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer):
        # input_image == Nsamples x 2 x subcs x symb
        # target      == Nsamples x 2 x subcs x symb
    # Ensure models are in training mode
    generator.train()
    discriminator.train()
    
    # Zero the gradients for both optimizers
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    
    # Forward pass
    gen_output = generator(input_image)
    disc_real_output = discriminator(target)
    disc_generated_output = discriminator(gen_output.detach())
    
    # Calculate losses
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    # Backward pass and optimize for generator
    disc_generated_output2 = discriminator(gen_output)
    gen_loss = generator_loss(disc_generated_output2, gen_output, target)
    gen_loss.backward(retain_graph=True)  # retain_graph=True if using same computational graph for discriminator
    generator_optimizer.step()
    
    # Backward pass and optimize for discriminator
    disc_loss.backward()
    discriminator_optimizer.step()
    
    return gen_loss.item(), disc_loss.item()

# to run train loop over epochs
def train_GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, train_loader, val_loader, NUM_EPOCHS, loss_track, H_GAN_val):
    start_time = datetime.datetime.now()
    criterion = nn.MSELoss()
    gen_loss_track = loss_track[0] 
    disc_loss_track = loss_track[1]
    gen_val_loss_track = loss_track[2]

    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()
        running_gen_loss  = 0.0
        running_disc_loss = 0.0
        if (epoch == NUM_EPOCHS-1):
                i = 0
        print("-----\nEPOCH:", epoch)        
        # for bi, (target, input_image) in enumerate(load_image_train(file_path)):
        for bi, (input_image, target) in enumerate(train_loader):
            # input_image == H_inter == Nsamples x 2 x subcs x symb
            # target      == H_Real  == Nsamples x 2 x subcs x symb x 2 x ant
            elapsed_time = datetime.datetime.now() - start_time
            gen_loss, disc_loss = train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer)
            running_gen_loss += gen_loss
            running_disc_loss += disc_loss
            
        avg_train_gen_loss = running_gen_loss / len(train_loader)
        avg_train_disc_loss = running_disc_loss / len(train_loader)
        gen_loss_track.append(avg_train_gen_loss)
        disc_loss_track.append(avg_train_disc_loss)
        print('Generate Loss (BCE Loss in training): ', avg_train_gen_loss)
        print('Discriminate Loss(BCE Loss in training): ', avg_train_disc_loss)
        
        generator.eval()
        running_gen_val_loss  = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
                val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
                val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
                val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)
                
                val_outputs_real = generator(val_inputs_real)
                val_loss_real = criterion(val_outputs_real, val_targets_real)
                running_gen_val_loss += val_loss_real.item()
                
                val_outputs_imag = generator(val_inputs_imag)
                val_loss_imag = criterion(val_outputs_imag, val_targets_imag)
                running_gen_val_loss += val_loss_imag.item()

                # save the estimated channel at the last epoch 
                # need i because we loop over batch_size
                if (epoch == NUM_EPOCHS-1): 
                    H_GAN_val[i:i+val_outputs_real.size(0),0,:,:].unsqueeze(1).copy_(val_outputs_real)
                    H_GAN_val[i:i+val_outputs_imag.size(0),1,:,:].unsqueeze(1).copy_(val_outputs_imag)
                    i = i+val_outputs_imag.size(0)
        
        # MSE of estimated and true channels            
        avg_gen_val_loss = running_gen_val_loss/ (len(val_inputs)*2) # divided by 2 because real and imag parts 
            
        gen_val_loss_track.append(avg_gen_val_loss)
        
        print('MSE of estimated and true channels (normalized): ', avg_gen_val_loss)
    return H_GAN_val, gen_val_loss_track, disc_loss_track, gen_loss_track
