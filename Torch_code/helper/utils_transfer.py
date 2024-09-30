import torch.nn as nn

class FineTuneModel(nn.Module):
    def __init__(self, source_model):
        super(FineTuneModel, self).__init__()
        
        # Copy layers from the source model
        self.normalization = source_model.normalization
        self.conv1 = source_model.conv1
        self.conv2 = source_model.conv2
        self.conv3 = source_model.conv3
        self.conv4 = source_model.conv4
        self.activate = source_model.activate
        
        self.dropOutPos = source_model.dropOutPos
        self.dropOut = source_model.dropOut
        if source_model.dropOut != 0:
            self.dropout = nn.Dropout(p=source_model.dropOut)
        
        # Freeze conv1 to conv4
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        
        # Replace conv5
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        
        # Add 3 more layers
        self.conv6 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.conv7 = nn.Conv2d( 8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    def forward(self, x):
        x = self.normalization(x)
        if (0 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv1(x))
        if (1 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv2(x))
        if (2 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv3(x))
        if (3 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv4(x))
        if (4 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv5(x))
        if (5 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.activate(self.conv6(x))
        if (6 in self.dropOutPos) and self.dropOut:
            x = self.dropout(x)  
        x = self.conv7(x)  
        return x