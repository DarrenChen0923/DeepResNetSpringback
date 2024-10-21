import torch.nn as nn
import torch.nn.functional as fc

from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels) )
   
    def forward(self, x):
        out = fc.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = fc.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 16  
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        if args.grid !=5 :
            self.layer3 = self._make_layer(64, 3, stride=2)
            self.layer4 = self._make_layer(128, 3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        if args.grid ==20:
            self.fc = nn.Linear(512, num_classes)  
        else :
            self.fc = nn.Linear(128, num_classes)
 
    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = fc.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        if args.grid !=5 :
            x = self.layer3(x)
            x = self.layer4(x)
        x = fc.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
