import numpy as np
import matplotlib.pyplot as plt
from gudhi.dtm_rips_complex import DTMRipsComplex
import torch
import torch.nn as nn
import random
import datetime
import time

import os

from lib.utils import *
from lib.pointnet import *

def _dist_matrix_to_pointwise_feature_vector(dist, weight1, bias1, weight2=None, bias2=None, activation=torch.tanh, ds_op="sum"):
    value = dist.unsqueeze(3) 
    
    for i in range(len(weight1)):
        value = activation(torch.einsum("bijk,kl->bijl", value, weight1[i]) + bias1[i])
    if ds_op == "max":
        value = torch.max(value, axis=2).values # maxの場合
    elif ds_op == "sum":
        value = torch.sum(value, axis=2) # sumの場合
    elif ds_op == "concat":
        value = torch.cat([torch.max(value, axis=2).values, torch.sum(value, axis=1)], dim=2)
    else:
        raise NotImplementedError

    if (weight2 is not None) and (bias2 is not None):
        for i in range(len(weight2)):
            value = activation(torch.einsum("bik,kl->bil", value, weight2[i]) + bias2[i])
    return value

def dist_matrix_delay_calc(X, params, activation, tar_X=None, ds_op="sum", input_format="coord"):
    weight_all_1, bias_all_1, weight_all_2, bias_all_2, weight_all_3, bias_all_3, weight_pt_1, bias_pt_1, weight_pt_2, bias_pt_2, weight_delay, bias_delay = params

    if input_format == "coord":
        if tar_X is None:
            tar_X = X.clone()
        dist = torch.cdist(X, X, p=2)
        tar_dist = torch.cdist(tar_X, X, p=2)
    elif input_format == "dist":
        if tar_X is not None:
            raise NotImplementedError
        dist = X.clone()
        tar_dist = X.clone()
    else:
        raise NotImplementedError
    value_all = _dist_matrix_to_pointwise_feature_vector(
        dist, weight_all_1, bias_all_1, activation=activation, ds_op=ds_op
    )
    value_all = deepsets(value_all, weight_all_2, weight_all_3, bias_all_2, bias_all_3, ds_op=ds_op, activation=activation) 
    value_point = _dist_matrix_to_pointwise_feature_vector(
        tar_dist, weight_pt_1, bias_pt_1, weight_pt_2, bias_pt_2, activation=activation, ds_op=ds_op
    )
    value = torch.stack([value_all for _ in range(tar_dist.shape[1])], dim=1) 
    value = torch.cat([value, value_point], dim=2)
    for i in range(len(weight_delay)):
        value = activation(torch.einsum("bij,jk->bik", value, weight_delay[i]) + bias_delay[i])
    return value[:, :, 0]

def MLP_coordinate2vec(X, weight, bias):
    activation = torch.tanh
    value = X.clone()
    for i in range(len(weight)):
        value = activation(torch.einsum("bij,jk->bik", value, weight[i]) + bias[i])
    return value

class TopoRep:
    def __init__(self, args_dict, input_dim=None, output_dim=None, point_num=None, delay_calc_params=None, cls_params=None, reducer_params=None, out_feature_params=None, last_linear=False):
        self.method = args_dict.get("method")
        self.dtm_k = args_dict.get("dtm_k") # used when method == "dtm"
        self.narrow = (args_dict.get("narrow", 0) == 1)
        self.normalize = (args_dict.get("normalize", 0) == 1)
        self.out_feature = args_dict.get("out_feature")
        self.input_format = args_dict.get("input_format")
        self.task = args_dict.get("task")
        self.cls_model = args_dict.get("cls_model")

        ### fixed paramemters ###
        self.base_filt = "weighted"
        self.PI_weight = "linear"
        self.PI_grid_num = 15
        self.PI_h = 0.5
        self.pi_max = 4.0
        self.delay_calc_activation = torch.relu
        self.ds_op = "sum"

        ### Reflect functional input ###
        self.input_dim = (None if self.input_format == "dist" else input_dim)
        self.output_dim = output_dim
        self.point_num = point_num
        self.delay_calc_params = delay_calc_params
        self.cls_params = cls_params
        self.out_feature_params = out_feature_params
        self.reducer_params = reducer_params
        self.last_linear = last_linear

        if self.delay_calc_params is None:
            if self.method == "dtm":
                self.delay_calc_params = []
            elif self.method == "deepsets_coord":
                assert self.input_format == "coord"

                ds_out_dim = 16
                coor_out_dim = 4
                op_dim_inc = (2 if self.ds_op == "concat" else 1)

                # deepsets
                weight_ds_1, bias_ds_1 = MLP_layers_init([self.input_dim, 128, 128, 64])
                weight_ds_2, bias_ds_2 = MLP_layers_init([op_dim_inc*64, 128, 128, ds_out_dim])
                # coordinate feature vector
                weight_coord, bias_coord = MLP_layers_init([self.input_dim, 64, 128, coor_out_dim])
                # delay_calc
                weight_delay, bias_delay = MLP_layers_init([coor_out_dim + ds_out_dim, 128, 1])

                self.delay_calc_params = [
                    weight_ds_1, bias_ds_1, weight_ds_2, bias_ds_2, 
                    weight_delay, bias_delay, weight_coord, bias_coord, 
                ]
            
            elif self.method == "dist":
                point_cloud_feature_dim = 8
                pointwise_feature_dim = 4
                op_dim_inc = (2 if self.ds_op == "concat" else 1)

                weight_all_1, bias_all_1 = MLP_layers_init([1, 128, 128, 64])
                weight_all_2, bias_all_2 = MLP_layers_init([op_dim_inc*64, 128, 128, 64])
                weight_all_3, bias_all_3 = MLP_layers_init([op_dim_inc*64, 128, 128, point_cloud_feature_dim])
                weight_pt_1, bias_pt_1 = MLP_layers_init([1, 128, 128, 64])
                weight_pt_2, bias_pt_2 = MLP_layers_init([op_dim_inc*64, 128, 128, pointwise_feature_dim])
                weight_delay, bias_delay = MLP_layers_init([point_cloud_feature_dim + pointwise_feature_dim, 128, 128, 128, 1])

                self.delay_calc_params = [
                    weight_all_1, bias_all_1, weight_all_2, bias_all_2, weight_all_3, bias_all_3, 
                    weight_pt_1, bias_pt_1, weight_pt_2, bias_pt_2, 
                    weight_delay, bias_delay
                ]
        
        ### If parameters are not designated, ititialize them. ###
        if self.out_feature_params is None:
            if self.out_feature == "none":
                self.out_feature_params = []
                self.cls_input_dim = self.PI_grid_num**2
            elif self.out_feature.lower() == "deepsets":
                self.deepsets = DeepSets(self.input_dim, 16)
                self.out_feature_params = self.deepsets.ds_weight1 + self.deepsets.ds_weight2 + self.deepsets.ds_bias1 + self.deepsets.ds_bias2
                self.cls_input_dim = self.PI_grid_num ** 2 + 16
            elif self.out_feature.lower() == "pointnet":
                self.pointnet = PointNet(self.input_dim, self.point_num, 16)
                self.out_feature_params = list(self.pointnet.parameters())
                self.cls_input_dim = self.PI_grid_num ** 2 + 16
                new_param = self.pointnet.state_dict()
                new_param['main.0.main.6.bias'] = torch.eye(self.input_dim, self.input_dim).view(-1)
                new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
                self.pointnet.load_state_dict(new_param)
            if self.task == "ae":
                self.cls_input_dim = 16+16
        # If designated, read them.
        else:
            if self.out_feature.lower() == "deepsets":
                self.deepsets = DeepSets(self.input_dim, self.out_feature_params, init_params=self.out_feature_params)
            elif self.out_feature.lower() == "pointnet":
                try:
                    self.pointnet = PointNet(self.input_dim, self.point_num, 16)
                    try:
                        print("a")
                        for i, param in enumerate(self.pointnet.parameters()): 
                            param = param.detach()
                            param = self.out_feature_params[i]
                    except:
                        print("b")
                        self.pointnet.load_state_dict(self.out_feature_params)
                except:
                    self.pointnet = PointNet(self.input_dim, self.point_num, 64)
                    try:
                        print("c")
                        for i, param in enumerate(self.pointnet.parameters()):
                            param = param.detach()
                            param = self.out_feature_params[i]
                    except:
                        print("d")
                        self.pointnet.load_state_dict(self.out_feature_params)
        
        ### If task == ae, the dimension of the output of topological representation will be recuded by linear MLP. ###
        if self.task == "ae":
            if self.reducer_params is None:
                self.reducer = MultiLayerPerceptron(width_list=[self.PI_grid_num ** 2, 16], activation=lambda x:x, task=self.task)
                self.reducer_params = self.reducer.weight + self.reducer.bias
            else:
                self.reducer = MultiLayerPerceptron(
                    width_list=[self.PI_grid_num ** 2, 16], 
                    activation=lambda x:x, 
                    task=self.task, 
                    weight_init=[self.reducer_params[0]], 
                    bias_init=[self.reducer_params[1]]
                )
        else:
            self.reducer_params = []

        ### If parameters are not designated, ititialize them. ###
        if self.cls_params is None:
            if self.cls_model == "deep":
                cls_act = torch.relu
                self.task_solver = MultiLayerPerceptron(
                    width_list=[self.cls_input_dim, 256, 256, self.output_dim], 
                    activation=cls_act, task=self.task, last_linear=self.last_linear
                )
                self.cls_params = self.task_solver.weight + self.task_solver.bias
            elif self.cls_model == "linear":
                cls_act = lambda x: x
                self.task_solver = MultiLayerPerceptron(
                    width_list=[self.cls_input_dim, self.output_dim], 
                    activation=cls_act, task=self.task, last_linear=self.last_linear, 
                    # bias=0.0
                )
                self.cls_params = self.task_solver.weight + self.task_solver.bias
            elif self.cls_model == "2-linear":
                cls_act = lambda x: x
                self.task_solver = MultiLayerPerceptron(
                    width_list=[self.cls_input_dim, 64, self.output_dim], 
                    activation=cls_act, task=self.task, last_linear=self.last_linear, 
                    # bias=0.0
                )
                self.cls_params = self.task_solver.weight + self.task_solver.bias

    def delay_calc(self, X):
        if self.method == "dtm":
            delay_list = []
            if self.input_format == "dist":
                dist = X.detach()
            elif self.input_format == "coord":
                dist = torch.cdist(X, X).detach()
            for i in range(X.shape[0]):
                delay = DTMRipsComplex(distance_matrix=dist[i, :, :].detach().numpy(), k=self.dtm_k).weights
                delay_list.append(delay)
            return torch.tensor(delay_list).to(torch.float32)
        elif self.method == "dist":
            return dist_matrix_delay_calc(X, self.delay_calc_params, self.delay_calc_activation, ds_op=self.ds_op, input_format=self.input_format)
        else:
            raise NotImplementedError

    def get_heatmap_tensors(self, X):
        delay = self.delay_calc(X)
        heatmap_tensors = get_heatmap_tensors(
            X, delay, 
            base_filt=self.base_filt, input_format=self.input_format, PI_weight=self.PI_weight, 
            h=self.PI_h, pi_range_x=self.pi_max, pi_range_y=self.pi_max, grid_num_x=self.PI_grid_num, grid_num_y=self.PI_grid_num
        )
        return heatmap_tensors
    
    def concat_out_feature(self, X, heatmap_tensors):
        value = heatmap_tensors.reshape(heatmap_tensors.shape[0], heatmap_tensors.shape[1] * heatmap_tensors.shape[2])

        if self.task == "ae":
            value = self.reducer.output(value)

        if self.out_feature == "none":
            pass
        elif self.out_feature.lower() == "deepsets":
            value = torch.cat([value, self.deepsets.output(X)], axis=1)
            pass
        elif self.out_feature.lower() == "pointnet":
            value = torch.cat([value, self.pointnet(X.view(-1, X.shape[2]))], axis=1)
            pass
        else:
            raise NotImplementedError

        if self.normalize:
            value = nn.functional.normalize(value, dim=0)
        
        return value

    def get_representation(self, X):
        heatmap_tensors = self.get_heatmap_tensors(X)
        return self.concat_out_feature(X, heatmap_tensors)
    
    def entire_network(self, X, heatmap_tensors=None, rot=False):
        if (self.input_format == "dist") and (self.out_feature != "none"):
            raise NotImplementedError
        if rot:
            X = random_rotation(X)
        if heatmap_tensors is None:
            value = self.get_representation(X)
        else:
            value = self.concat_out_feature(X, heatmap_tensors)
        return self.task_solver.output(value)