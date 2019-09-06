import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class DynamicDiscriminator_19(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), ncf=64):
        super(DynamicDiscriminator_19, self).__init__()
        nc = input_shape[0]  # NCHW
        self.input_height = input_shape[1]
        self.ncf = ncf
        height = self.input_height
        self.conv1 = nn.Conv2d(6, ncf, 3)
        height = (height - 3) // 2
        self.conv2 = nn.Conv2d(ncf, ncf, 3)
        height = (height - 2) // 2
        self.conv3 = nn.Conv2d(ncf, ncf, 3)
        height = (height - 2) // 2
        self.conv4 = nn.Conv2d(ncf, ncf, 3)
        height = (height - 2) // 2
        # self.conv5 = nn.Conv2d(ncf, ncf, 3)
        # height = (height - 2) // 2
        if self.input_height == 128:
            self.fc1 = nn.Linear(height * ncf * height, 1024)
        else:
            self.fc1 = nn.Linear(5408, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, cur_obs, next_obs):
        """
        :param cur_obs: Current observation (in image format)
        :param next_obs: Next observation (in image format)
        :return: Probability of two consecutive observation can happen (value from 0 to 1)
        """
        x = torch.cat((cur_obs, next_obs), 1)  # N,6,224,224  if 128 => 6,128,128
        x = F.relu(self.conv1(x))  # N,6,220,220  if 128 => 6,124,124
        x = F.max_pool2d(x, 2, 2)  # N,6,110,110  if 128 => 6,62,62
        x = F.relu(self.conv2(x))  # N,16,108,108 if 128 => 16,60,60
        x = F.max_pool2d(x, 2, 2)  # N,16,54,54   if 128 => 16,30,30
        x = F.relu(self.conv3(x))  # N,8,52,52    if 128 => 8,28,28
        x = F.max_pool2d(x, 2, 2)  # N,8,26,26    if 128 => 8,14,14
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # N,5408       if 128 => N,1568
        x = F.relu(self.fc1(x))  # N,1024
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # N,500
        x = self.fc3(x)  # N,2
        return x

    def predict(self, cur_obs, next_obs):
        x = self.forward(cur_obs, next_obs)
        prob = F.softmax(x)
        return prob

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DynamicDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), ncf=32):
        super(DynamicDiscriminator, self).__init__()
        nc = input_shape[0]  # NCHW
        self.input_height = input_shape[1]
        self.ncf = ncf
        height = self.input_height
        self.conv1 = nn.Conv2d(6, ncf, 3)
        height = (height -2)//2
        self.conv2 = nn.Conv2d(ncf, ncf, 3)
        height = (height -2)//2
        self.conv3 = nn.Conv2d(ncf, ncf, 3)
        height = (height - 2) // 2
        # self.conv4 = nn.Conv2d(ncf, ncf, 3)
        # height = (height - 2) // 2
        # self.conv5 = nn.Conv2d(ncf, ncf, 3)
        # height = (height - 2) // 2
        if self.input_height == 128:
            self.fc1 = nn.Linear(height*ncf*height, 1024)
        else:
            self.fc1 = nn.Linear(height*ncf*height, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 2)



    def forward(self, cur_obs, next_obs):
        """
        :param cur_obs: Current observation (in image format)
        :param next_obs: Next observation (in image format)
        :return: Probability of two consecutive observation can happen (value from 0 to 1)
        """
        # cur_obs = self.attention(cur_obs,20)
        # next_obs = self.attention(next_obs,20)
        x = torch.cat((cur_obs, next_obs), 1) # N,6,224,224  if 128 => 6,128,128
        x = F.relu(self.conv1(x))  # N,6,220,220  if 128 => 6,124,124
        x = F.max_pool2d(x, 2, 2)  # N,6,110,110  if 128 => 6,62,62
        x = F.relu(self.conv2(x))  # N,16,108,108 if 128 => 16,60,60
        x = F.max_pool2d(x, 2, 2)  # N,16,54,54   if 128 => 16,30,30
        x = F.relu(self.conv3(x))  # N,8,52,52    if 128 => 8,28,28
        x = F.max_pool2d(x, 2, 2)  # N,8,26,26    if 128 => 8,14,14
        # x = F.relu(self.conv4(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # N,5408       if 128 => N,1568
        x = F.relu(self.fc1(x))  # N,1024
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # N,500
        x = self.fc3(x)  # N,2
        return x

    def attention(self,x, padding = 20):
        # attendtion to center part
        # x[N,C,X,Y]
        x[:,:,:padding,:] =0
        x[:, :, -padding:, :] = 0
        x[:, :, :, :padding] = 0
        x[:, :, :, -padding:] = 0
        return  x

    def predict(self,cur_obs, next_obs):
        x = self.forward(cur_obs,next_obs)
        prob = F.softmax(x)
        return prob

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0.02)


class TargetDetector(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(TargetDetector, self).__init__()
        nc = input_shape[0]  # NCHW
        self.input_height = input_shape[1]
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        if self.input_height == 128:
            self.fc1 = nn.Linear(1568, 1024)
        else:
            self.fc1 = nn.Linear(5408, 1024)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, cur_obs, goal_description):
        """
        :param cur_obs: Current observation (in image format)
        :param goal_description: State describes goal (in image format)
        :return: Probability of current observation reaches to target (value from 0 to 1)
        """
        x = torch.cat((goal_description, cur_obs), 1)  # N,6,224,224
        x = F.relu(self.conv1(x))  # N,6,220,220
        x = F.max_pool2d(x, 2, 2)  # N,6,110,110
        x = F.relu(self.conv2(x))  # N,16,108,108
        x = F.max_pool2d(x, 2, 2)  # N,16,54,54
        x = F.relu(self.conv3(x))  # N,8,52,52
        x = F.max_pool2d(x, 2, 2)  # N,8,26,26
        x = x.view(x.size(0), -1)  # N,5408
        x = F.relu(self.fc1(x))  # N,1024
        x = F.relu(self.fc2(x))  # N,500
        x = self.fc3(x)  # N,2
        return x


    def predict(self,cur_obs, goal_description):
        x = self.forward(cur_obs,goal_description)
        prob = F.softmax(x)
        return prob

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0.02)




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
    # model = Target_Model()
    model = TargetDetector(input_shape =(3,128,128))
    return model


def neighbor_model():
    # model = Neigbor_Model()
    model = DynamicDiscriminator(input_shape =(3,128,128))
    return model
