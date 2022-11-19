import torch
import torch.nn as nn
from collections import OrderedDict
import math

class ConvNet(nn.Module):
    def __init__(self,num_classes=105, include_top=True):
        super(ConvNet,self).__init__()
        
        self.include_top = include_top

        #Input shape= (*,3,224,224)
        
        # self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        # #Shape= (*,12,224,224)
        # self.bn1=nn.BatchNorm2d(num_features=32)
        # #Shape= (*,12,224,224)
        # self.relu1=nn.ReLU()
        #Shape= (*,12,224,224)
        self.layer1 = self.make_conv_layer(3, 32, 3, 1, 1)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Shape= (*,12,112,112)
        
        # self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        # #Shape= (*,20,112,112)
        # self.relu2=nn.ReLU()
        #Shape= (*,20,112,112)
        self.layer2 = self.make_conv_layer(32, 128, 3, 2, 1)

        self.layer2_2 = self.make_conv_layer(128, 64, 3, 2, 1)
        # self.layer2_3 = self.make_conv_layer(64, 32, 3, 1, 1)
        
        # self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        # #Shape= (*,32,112,112)
        # self.bn3=nn.BatchNorm2d(num_features=128)
        # #Shape= (*,32,112,112)
        # self.relu3=nn.ReLU()
        #Shape= (*,32,112,112)

        self.layer3 = self.make_conv_layer(64, 256, 3, 2, 1)

        # self.layer3_2 = self.make_conv_layer(128, 128, 3, 2, 3)
        # self.layer3_3 = self.make_conv_layer(128, 64, 3, 1, 3)
        
        self.pool2=nn.MaxPool2d(kernel_size=2)

        # self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=3,padding=1)
        # #Shape= (*,44,56,56)
        # self.relu4=nn.ReLU()
        #Shape= (*,44,56,56)
        self.layer4 = self.make_conv_layer(256, 128, 3, 2, 1)

        self.layer4_2 = self.make_conv_layer(128, 512, 3, 2, 1)
        # self.layer4_3 = self.make_conv_layer(256, 128, 3, 1, 1)

        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        # self.bn5=nn.BatchNorm2d(num_features=512)
        # self.relu5=nn.ReLU()
        self.layer5 = self.make_conv_layer(512, 1024, 2, 2, 1)

        self.pool3=nn.MaxPool2d(kernel_size=2)
        
        # self.conv6 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, stride=3, padding=1)
        # self.relu6=nn.ReLU()
        self.layer6 = self.make_conv_layer(1024, 512, 3, 2, 1)

        self.layer6_2 = self.make_conv_layer(512, 2048, 3, 2, 1)
        # self.layer6_3 = self.make_conv_layer(1024, 512, 3, 1, 1)

        # self.conv7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=3, padding=1)
        # self.layer7 = self.make_conv_layer(512, 2048, 3, 2, 3)

        # self.fc = nn.Sequential(OrderedDict([
        #     # ("fc1", nn.Linear(in_features= 44*56*56, out_features= 1024)),
        #     # ("fc2", nn.Linear(in_features= 1024 , out_features= 256)),
        #     ("fc2", nn.Linear(in_features= 2048 , out_features= 256)),
        #     # ("fc_dropout", nn.Dropout(0.25)),
        #     ("fc3", nn.Linear(in_features= 256 ,out_features=num_classes)),
        # ]))

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # self.dropout = nn.Dropout(0.25)

    def make_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
    
    #Feed forwad function    
    def forward(self,input):
        # output=self.conv1(input)
        # output=self.bn1(output)
        # output=self.relu1(output)
        output = self.layer1(input)

        output=self.pool(output)
            
        # output=self.conv2(output)
        # output=self.relu2(output)
        output = self.layer2(output)
        output = self.layer2_2(output)
        # output = self.layer2_3(output)
            
        # output=self.conv3(output)
        # output=self.bn3(output)
        # output=self.relu3(output)
        output = self.layer3(output)
        # output = self.layer3_2(output)
        # output = self.layer3_3(output)

        output=self.pool2(output)
            
        # output=self.conv4(output)
        # output=self.relu4(output)
        output = self.layer4(output)
        output = self.layer4_2(output)
        # output = self.layer4_3(output)
        # print(output.shape)

        # output=self.conv5(output)
        # output=self.bn5(output)
        # output=self.relu5(output)
        output = self.layer5(output)

        output = self.pool3(output)

        # output=self.conv6(output)
        # output=self.relu6(output)
        output = self.layer6(output)
        output = self.layer6_2(output)
        # output = self.layer6_3(output)

        # output=self.conv7(output)
        # output = self.layer7(output)

        if not self.include_top:
            return output

        # output=output.view(-1, 44*56*56)
        output=output.view(-1, 2048)
            
        # output=self.fc1(output) 
        # output=self.fc2(output)  
        # output = self.dropout(output)
        # output=self.fc3(output)
        output = self.fc(output)
        
        return output
