import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DynamicDiscriminator(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), ncf=32):
        super(DynamicDiscriminator, self).__init__()
        nc = input_shape[0]  # NCHW
        self.feature_size = self.chunk_num * self.chunk_size
        self.ncf = ncf

    def forward(self, cur_obs, next_obs):
        """
        :param cur_obs: Current observation (in image format)
        :param next_obs: Next observation (in image format)
        :return: Probability of two consecutive observation can happen (value from 0 to 1)
        """
        prob = None
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

    def forward(self, cur_obs, goal_description):
        """
        :param cur_obs: Current observation (in image format)
        :param goal_description: State describes goal (in image format)
        :return: Probability of current observation reaches to target (value from 0 to 1)
        """
        prob = None
        return prob

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0.02)