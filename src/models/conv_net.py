import torch
import torch.nn as nn
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self,num_classes=105, include_top=True):
        super(ConvNet,self).__init__()
        
        self.include_top = include_top

        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Shape= (256,12,75,75)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.conv4=nn.Conv2d(in_channels=32,out_channels=44,kernel_size=3,stride=1,padding=1)
        #Shape= (256,44,56,56)
        self.relu4=nn.ReLU()
        #Shape= (256,44,56,56)

        
        

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features= 44*56*56, out_features= 1024)),
            ("fc2", nn.Linear(in_features= 1024 , out_features= 256)),
            ("fc_dropout", nn.Dropout(0.25)),
            ("fc3", nn.Linear(in_features= 256 ,out_features=num_classes)),
        ]))
        
        # self.dropout = nn.Dropout(0.25)
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        output=self.pool2(output)
            
        output=self.conv4(output)
        output=self.relu4(output)
        # print(output.shape)

        if not self.include_top:
            return output

        output=output.view(-1, 44*56*56)
            
        # output=self.fc1(output) 
        # output=self.fc2(output)  
        # output = self.dropout(output)
        # output=self.fc3(output)
        output = self.fc(output)
        
        return output
