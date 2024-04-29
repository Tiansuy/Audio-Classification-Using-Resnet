import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Create model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(31*2*32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn(self.pool(F.relu(self.conv1(x))))
        x = self.bn(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 31*2*32)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

import torch
import torch.nn as nn
import torchvision.models as models

class CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一层卷积以适应32x32的输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改最后的全连接层以匹配目标类别数
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

