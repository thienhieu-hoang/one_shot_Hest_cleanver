#%%
import torch
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, down=True, act="ReLU", leaky_param=0.01, use_dropout=False):
        super(Block, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.down = down
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        if act=="Tanh":
            self.activation = nn.Tanh()
        elif act=="ReLU":
            self.activation = nn.ReLU()
        elif act=="LeakyReLU":
            self.activation = nn.LeakyReLU(leaky_param)
        elif act=="Sigmoid":
            self.activation = nn.Sigmoid()
            
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=1, leaky_param=0.01, features=64):
        super().__init__()
        self.initial_down = nn.Conv2d(in_channels, features, kernel_size=(8,4), stride=(2,1), padding=1, padding_mode="reflect")
        # if act=="Tanh":
        #     self.activation = nn.Tanh()
        # elif act=="ReLU":
        #     self.activation = nn.ReLU()
        # elif act=="LeakyReLU":
        #     self.activation = nn.LeakyReLU(leaky_param)
        # elif act=="Sigmoid":
        #     self.activation = nn.Sigmoid()
            
        self.act_tanh = nn.Tanh()
        self.act_leaky = nn.LeakyReLU(leaky_param)
        self.act_relu = nn.ReLU()
        
        self.down1 = Block(features    , features * 2, kernel_size=(8,3), stride=(2,1), padding=(1,1), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        self.down2 = Block(features * 2, features * 4, kernel_size=(8,3), stride=(2,1), padding=(1,0), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        self.down3 = Block(features * 4, features * 8, kernel_size=(7,3), stride=(2,1), padding=(0,0), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        self.down4 = Block(features * 8, features * 8, kernel_size=(8,3), stride=(2,1), padding=(0,0), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        self.down5 = Block(features * 8, features * 8, kernel_size=(7,3), stride=(1,1), padding=(0,0), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        self.down6 = Block(features * 8, features * 8, kernel_size=(4,3), stride=(1,1), padding=(0,0), down=True, act="LeakyReLU", leaky_param=leaky_param, use_dropout=False) # 
        
        self.bottleneck = nn.Conv2d(features * 8, features * 8, kernel_size=(4,3), stride=(1,1), padding=(0,0))
        
        self.up1 = Block(features * 8    , features * 8, kernel_size=(4,3), stride=(1,1), padding=(0,0), down=False, act="ReLU", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, kernel_size=(4,3), stride=(1,1), padding=(0,0), act="ReLU", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, kernel_size=(7,3), stride=(1,1), padding=(0,0), act="ReLU", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, kernel_size=(8,3), stride=(2,1), padding=(0,0), act="ReLU", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, kernel_size=(7,3), stride=(2,1), padding=(0,0), act="ReLU", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, kernel_size=(8,3), stride=(2,1), padding=(1,0), act="ReLU", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features    , down=False, kernel_size=(8,3), stride=(2,1), padding=(1,1), act="ReLU", use_dropout=False)
        
        self.final_up = nn.ConvTranspose2d(features * 2, in_channels, kernel_size=(8,4), stride=(2,1), padding=1)

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.act_relu(self.bottleneck(d7))
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.act_tanh(self.final_up(torch.cat([up7, d1], 1)))

class Generator_FineTune(nn.Module):
    def __init__(self, source_model):
        super(Generator_FineTune, self).__init__()
        self.initial_down = source_model.initial_down
        self.act_tanh  = source_model.act_tanh
        self.act_leaky = source_model.act_leaky
        self.act_relu  = source_model.act_relu
        
        self.down1 = source_model.down1 # 
        self.down2 = source_model.down2 # 
        self.down3 = source_model.down3 # 
        self.down4 = source_model.down4 # 
        self.down5 = source_model.down5 # 
        self.down6 = source_model.down6 # 
        
        self.bottleneck = source_model.bottleneck
        
        self.up1 = source_model.up1
        self.up2 = source_model.up2
        self.up3 = source_model.up3
        self.up4 = source_model.up4
        self.up5 = source_model.up5
        self.up6 = source_model.up6
        self.up7 = source_model.up7
        
        self.final_up = source_model.final_up
        
        # Freeze 
        layers_to_freeze = [self.initial_down, self.down1, self.down2, self.down3, self.down4, self.down5] #, self.down6,
                            # self.bottleneck, self.up1, self.up2, self.up3, self.up4, self.up5]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.act_relu(self.bottleneck(d7))
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.act_tanh(self.final_up(torch.cat([up7, d1], 1)))
    
#%%
def test():
    x = torch.randn((1, 1, 612, 14))  # (samples, channels, H, W)
    model = Generator(in_channels=1, features=64)
    preds = model(x)
    print(preds.shape)
    print(model)
    summary(model, input_size=(1,612,14), device='cpu')

#%%
if __name__ == "__main__":
    test()
