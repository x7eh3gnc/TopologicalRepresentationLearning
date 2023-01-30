import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import datetime
import time

import os
import argparse

from lib.utils import *

class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_channels))

    def forward(self, input_data):
        return self.main(input_data)

class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        out = torch.permute(input_data.view(-1, self.num_points, self.num_channels), (0, 2, 1))
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out

class InputTNet(nn.Module):
    def __init__(self, input_dim, num_points):
        super(InputTNet, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(self.input_dim, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, self.input_dim**2)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, self.input_dim, self.input_dim) 
        out = torch.einsum("bij,bkj->bik", input_data.view(-1, self.num_points, self.input_dim), matrix)
        out = out.view(-1, self.input_dim)
        return out

class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.einsum("bij,bkj->bik", input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out

class PointNet(nn.Module):
    def __init__(self, input_dim, num_points, num_labels):
        super(PointNet, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.input_dim, self.num_points), 
            NonLinear(self.input_dim, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points), 
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels),
            )

    def forward(self, input_data):
        return self.main(input_data)