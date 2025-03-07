#%%
import torch
import torch.nn as nn
from torchsummary import summary

#%%
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, act = 'LeakyReLU', leak_params=0.2, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial =  nn.Conv2d(
                in_channels * 2, # x, y <- concatenate along the channels
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            )
        
        if act=='Tanh':
            self.activate = nn.Tanh()
        elif act=="ReLU":
            self.activation = nn.ReLU()
        elif act=="LeakyReLU":
            self.activation = nn.LeakyReLU(leak_params)
        elif act=="Sigmoid":
            self.activation = nn.Sigmoid()
        
        self.conv0 = CNNBlock(in_channels=features[0], out_channels=features[1], kernel_size=(4,4), stride=(2,1), padding =(1,1), padding_mode="reflect")
        self.conv1 = CNNBlock(in_channels=features[1], out_channels=features[2], kernel_size=(5,4), stride=(3,2), padding =(1,1), padding_mode="reflect")
        self.conv2 = CNNBlock(in_channels=features[2], out_channels=features[3], kernel_size=(5,4), stride=(3,1), padding =(1,1), padding_mode="reflect")
        
        self.last  = nn.Conv2d(in_channels =features[3], out_channels = 1, kernel_size=(5,4), stride=(2,2), padding=1, padding_mode="reflect")

    def forward(self, x, y): # x, y -- training, target
        x = torch.cat([x, y], dim=1)
        x = self.activation(self.initial(x))
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.last(x) # not using activation here
        return x

class Discriminator_FineTune(nn.Module):
    def __init__(self, source_model):
        super(Discriminator_FineTune, self).__init__()
        
        self.initial =  source_model.initial
        
        self.activation = source_model.activation
        
        self.conv0 = source_model.conv0
        self.conv1 = source_model.conv1
        self.conv2 = source_model.conv2
        
        self.last  = source_model.last
        
        # Freeze
        layers_to_freeze = [self.initial, self.conv0, self.conv1]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x, y): # x, y -- training, target
        x = torch.cat([x, y], dim=1)
        x = self.activation(self.initial(x))
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.last(x)
        return x

def test():
    x = torch.randn((1, 1, 612, 14))
    y = torch.randn((1, 1, 612, 14))
    model = Discriminator(in_channels=1)
    preds = model(x, y)
    print(model)
    print(preds.shape)

#%%
if __name__ == "__main__":
    test()

# %%
