import torch
import sys
import os
from utilsPix2Pix import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import random

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y, y_min, y_max) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        # disc.zero_grad()
        # d_scaler.scale(D_loss).backward()
        # d_scaler.step(opt_disc)
        # d_scaler.update()
        
        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        # opt_gen.zero_grad()
        # g_scaler.scale(G_loss).backward()
        # g_scaler.step(opt_gen)
        # g_scaler.update()
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
        # Accumulate the total training loss
        total_val_loss += G_loss.item()
        
    avg_val_loss = total_val_loss / len(loop)
    print(f'Training Generator Loss: {avg_val_loss:.4f}')
    return avg_val_loss
            
def validate_fn(gen, val_loader, l1_loss, bce, flag_last_epoch=False):
    gen.eval()  # Set generator to evaluation mode
    total_val_loss = 0
    i = 0
    all_H = []

    with torch.no_grad():  # Disable gradient calculation for validation
        for val_inputs, val_targets, val_targetsMin, val_targetsMax in val_loader:
            val_inputs_real = val_inputs[:,0,:,:].unsqueeze(1)
            val_inputs_imag = val_inputs[:,1,:,:].unsqueeze(1)
            val_targets_real = val_targets[:,0,:,:].unsqueeze(1)
            val_targets_imag = val_targets[:,1,:,:].unsqueeze(1)

            # Forward pass to generate predictions
            val_outputs_real = gen(val_inputs_real)
            G_fake_loss = bce(val_outputs_real, val_targets_real)
            L1 = l1_loss(val_outputs_real, val_targets_real) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

            val_outputs_imag = gen(val_inputs_imag)
            G_fake_loss = bce(val_outputs_imag, val_targets_imag)
            L1 = l1_loss(val_outputs_imag, val_targets_imag) * config.L1_LAMBDA
            G_loss += G_fake_loss + L1

            G_loss = G_loss/2

            # Accumulate the total validation loss
            total_val_loss += G_loss.item()
            
            # Save one generated-target pair for inspection
            if flag_last_epoch==True:
                H = torch.cat(val_outputs_real.unsqueeze(1), val_outputs_imag.unsqueeze(1))
                all_H.append(H)
    
    H_GAN_val = torch.cat(all_H, dim=0)

    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    gen.train()  # Set generator back to training mode
    if flag_last_epoch==True:
        return avg_val_loss, H_GAN_val
    else:
        return avg_val_loss


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder= config.FILE_PATH + "/results/train_test/evaluation")


if __name__ == "__main__":
    main()
