import torch
import torch.nn as nn
import torch.nn.functional as F


class Target_Model(nn.Module):
    def __init__(self):
        super(Target_Model, self).__init__()
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(5408, 1024)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), 1)  # N,6,224,224
        x = F.relu(self.conv1(x))  # N,6,220,220
        x = F.max_pool2d(x, 2, 2)  # N,6,110,110
        x = F.relu(self.conv2(x))  # N,16,108,108
        x = F.max_pool2d(x, 2, 2)  # N,16,54,54
        x = F.relu(self.conv3(x))  # N,8,52,52
        x = F.max_pool2d(x, 2, 2)  # N,8,26,26
        x = x.view(-1, 5408)  # N,5408
        x = F.relu(self.fc1(x))  # N,1024
        x = F.relu(self.fc2(x))  # N,500
        x = self.fc3(x)  # N,2
        return x


class Neigbor_Model(nn.Module):
    def __init__(self):
        super(Neigbor_Model, self).__init__()
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.dropout0 = nn.Dropout2d(p=0.3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(5408, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), 1)  # N,6,224,224
        x = F.relu(self.conv1(x))  # N,6,220,220
        x = F.max_pool2d(x, 2, 2)  # N,6,110,110
        x = F.relu(self.conv2(x))  # N,16,108,108
        #x = self.dropout0(x)
        x = F.max_pool2d(x, 2, 2)  # N,16,54,54
        x = F.relu(self.conv3(x))  # N,8,52,52
        x = F.max_pool2d(x, 2, 2)  # N,8,26,26
        x = x.view(-1, 5408)  # N,5408
        x = F.relu(self.fc1(x))  # N,1024
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # N,500
        x = self.fc3(x)  # N,2
        return x


def target_model():
    model = Target_Model()
    return model


def neighbor_model():
    model = Neigbor_Model()
    return model
