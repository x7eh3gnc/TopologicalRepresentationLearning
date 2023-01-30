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
from lib.pointnet import *

class PointNetEntireNetwork:
    def __init__(self, input_dim, output_dim, num_points, task, after_network=False, last_linear=False, connect_dim=64, normalize=False):
        self.after_network = after_network
        self.normalize = normalize
        if after_network:
            self.input_dim = input_dim
            self.pointnet = PointNet(input_dim, num_points, connect_dim)
            self.after_mlp = MultiLayerPerceptron(
                width_list=[connect_dim, 256, 256, output_dim], 
                activation=torch.relu, 
                task=task, 
                last_linear=last_linear
            )
            self.other_params = self.after_mlp.weight + self.after_mlp.bias
        else:
            self.pointnet = PointNet(input_dim, num_points, output_dim)
            if "cls" in task:
                self.after_func = torch.nn.Softmax(dim=1)
            elif task == "reg" or task == "ae":
                self.after_func = lambda x: x
            else:
                raise NotImplementedError
            self.other_params = []
        
        new_param = self.pointnet.state_dict()
        new_param['main.0.main.6.bias'] = torch.eye(input_dim, input_dim).view(-1)
        new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
        self.pointnet.load_state_dict(new_param)

    def output(self, X, rot=False):
        if rot:
            X = random_rotation(X)
        pointnet_output = self.pointnet(X.view(-1, self.input_dim))
        if self.normalize:
            pointnet_output = nn.functional.normalize(pointnet_output, dim=0)
        if self.after_network:
            return self.after_mlp.output(pointnet_output)
        else:
            return self.after_func(pointnet_output)

def learning_process(args_dict, setting_suffix, CV_idx=None, fold_num=10):
    start_time = time.time()
    exp_mode = args_dict.get("exp_mode")
    task = args_dict.get("task")
    connect_dim = args_dict.get("connect_dim", 16)
    normalize = args_dict.get("normalize")
    rot = (args_dict.get("rot") == 1)
    translation = (args_dict.get("translation") == 1)
    optimizer = args_dict.get("optimizer")
    iter_num = args_dict.get("iter_num")
    batch_size = args_dict.get("batch_size")
    lr = float(args_dict.get("lr"))
    after_opt = (args_dict.get("after_opt") == 1)

    trainX = torch.load(f"data/{exp_mode}_train").detach().to(torch.float32)
    testX = torch.load(f"data/{exp_mode}_test").detach().to(torch.float32)
    if "cls" in task:
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.long)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.long)
        output_dim = int(task.split("_cls")[0])
    elif task == "reg":
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.float32)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.float32)
        output_dim = 1
    elif task == "ae":
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.float32)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.float32)
        output_dim = trainy.shape[1] * trainy.shape[2]
    else:
        raise NotImplementedError
        
    if "cls" in task:
        criterion = nn.CrossEntropyLoss()
        after_network = False
        last_linear = False
    elif task == "reg":
        criterion = nn.MSELoss()
        after_network = False
        last_linear = False 
    elif task == "ae":
        criterion = lambda pred, gt: wasserstein_loss(torch.stack([pred[i, :].view(gt.shape[1], gt.shape[2]) for i in range(gt.shape[0])]), gt)
        after_network = True
        last_linear = True

    if CV_idx is not None:
        train_boolian_slice = [i % fold_num != CV_idx for i in range(trainX.shape[0])]
        test_boolian_slice = [i % fold_num == CV_idx for i in range(trainX.shape[0])]
        testX = trainX[test_boolian_slice, :, :]
        trainX = trainX[train_boolian_slice, :, :]
        testy = trainy[test_boolian_slice]
        trainy = trainy[train_boolian_slice]

    if "cls" in task:
        print(trainX.shape, testX.shape, trainy.shape, testy.shape)
        train_y_dict = {}
        test_y_dict = {}
        for lab in list(trainy):
            if float(lab) in train_y_dict.keys():
                train_y_dict[float(lab)] += 1
            else:
                train_y_dict[float(lab)] = 1
        for lab in list(testy):
            if float(lab) in test_y_dict.keys():
                test_y_dict[float(lab)] += 1
            else:
                test_y_dict[float(lab)] = 1
        if train_y_dict.keys() != test_y_dict.keys():
            print("[WARNING] The set of label in train data and test data is different. ")
        print("train_y_dict", train_y_dict)
        print("test_y_dict", test_y_dict)

    input_dim = trainX.shape[2]
    num_points = trainX.shape[1]

    if translation:
        trainX = trainX - torch.stack([torch.mean(trainX, dim=1) for _ in range(trainX.shape[1])], dim=1)
        testX  = testX  - torch.stack([torch.mean(testX,  dim=1) for _ in range(testX.shape[1])], dim=1)

    model = PointNetEntireNetwork(
        input_dim=input_dim, output_dim=output_dim, num_points=num_points, 
        task=task, after_network=after_network, last_linear=last_linear, 
        connect_dim=connect_dim, normalize=normalize
    )

    if optimizer == "sgd":
        opt = torch.optim.SGD(list(model.pointnet.parameters()) + model.other_params, lr=lr) 
    elif optimizer == "adam":
        opt = torch.optim.Adam(list(model.pointnet.parameters()) + model.other_params, lr=lr)
    else:
        raise NotImplementedError
    iter_max = iter_num

    train_loss_list = []
    test_loss_list = []

    for step in range(iter_max):
        opt.zero_grad()

        batch_sample = random.sample(range(trainX.shape[0]), batch_size)
        X_batch = trainX[batch_sample, :, :]
        y_batch = model.output(X_batch, rot=rot)
        batch_loss = criterion(y_batch, trainy[batch_sample])
        batch_loss.backward()
        opt.step()

        with torch.no_grad():
            train_output = model.output(trainX)
            test_output = model.output(testX, rot=rot)
            train_loss = criterion(train_output, trainy)
            test_loss = criterion(test_output, testy)
        
        train_loss_list.append(float(train_loss))
        test_loss_list.append(float(test_loss))
        
        if step % 100 == 0:
            print(f"Step {step}: train loss = {float(train_loss)}, test loss = {float(test_loss)}")

    if after_opt and model.other_params:
        for x in model.pointnet.parameters():
            x.requires_grad = False
        opt_final = torch.optim.Adam(model.other_params, lr=lr)
        for step in range(3000):
            batch_sample = random.sample(range(trainX.shape[0]), 100)
            batch_predict = model.output(trainX[batch_sample, :, :], rot=rot)
            batch_loss = criterion(batch_predict, trainy[batch_sample])
            opt_final.zero_grad()
            batch_loss.backward()
            opt_final.step()
            if step % 1000 == 0:
                print(f"Step {step}: batch loss = {float(batch_loss)}", flush=True) 
                
    train_output = model.output(trainX)
    train_pred_label = [int(x) for x in torch.max(train_output, dim=1).indices]
    count = sum([x == y for x, y in zip(train_pred_label, trainy)])
    print(f"Loss (train): {criterion(train_output, trainy)}", flush=True) 
    if "cls" in task: 
        print(f"Accuracy (train): {float(count/trainX.shape[0])}", flush=True) 

    test_output = model.output(testX, rot=rot)
    test_pred_label = [int(x) for x in torch.max(test_output, dim=1).indices]
    count = sum([x == y for x, y in zip(test_pred_label, testy)]) 
    print(f"Loss (test): {criterion(test_output, testy)}", flush=True) 
    if "cls" in task: 
        print(f"Accuracy (test): {float(count/testX.shape[0])}", flush=True) 

    end_time = time.time()
    td = datetime.timedelta(seconds=end_time - start_time)
    print("Execution time: ", str(td))

    if "cls" in task:
        return 100*count/testX.shape[0]
    elif task == "reg" or task == "ae":
        return float(criterion(test_output, testy))
    else:
        return None

if __name__ == "__main__":
    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9),'JST')).strftime('%Y%m%d')

    psr  = argparse.ArgumentParser(
        prog='main_pointnet',
        usage='',
        description=''
    )

    psr.add_argument('-e', '--exp_mode', default="ART06,nonrotae,200,60", help='Experiment data will be accessed with `data/{exp_mode}_train`.')
    psr.add_argument('-ts', '--task', default="ae", help='2_cls/10_cls/reg/ae')
    psr.add_argument('-no', '--normalize', type=int, default=1, help='If set 1, feature vector will be normalized')
    psr.add_argument('-r', '--rot', type=int, default=1, help='If set 1, input data will be randomly rotated as long as the input format is coordinate.')
    psr.add_argument('-tl', '--translation', type=int, default=1, help='If set 1, input data will be translated to locate its center at 0 as long as the input format is coordinate.')
    psr.add_argument('-om', '--optimizer', default="adam", help='sgd/adam')
    psr.add_argument('-it', '--iter_num', type=int, default=3000, help='# of iteration.')
    psr.add_argument('-b', '--batch_size', type=int, default=30, help='batch size')
    psr.add_argument('-l', '--lr', default="1e-2", help='learning rate')
    psr.add_argument('-ao', '--after_opt', type=int, default=1, help='If set 1, solver will be solely optimized after learning of representation.')
    psr.add_argument('-t', '--date', default=date_str, help='')
    psr.add_argument('-id', '--exp_id', default=None, help='experimetal index or designate 5fold/10fold')

    args = psr.parse_args()
    args_dict = vars(args)

    no_output_settings = {"input_format", "date"}
    print("### SETTINGS ###")
    setting_suffix = ""
    for key, value in args_dict.items():
        if key in no_output_settings:
            continue
        print(f"{key}={value}")
        setting_suffix = f"{setting_suffix},{key}={value}"
    print("########")

    if (args.exp_id is not None) and "fold" in args.exp_id:
        fold_num = int((args.exp_id).split("fold")[0])
        print(f"---------- {fold_num} FOLD CROSS VALIDATION ----------")
        result_list = []
        for i in range(fold_num):
            _setting_suffix = f"PointNet-{i}of{fold_num}{setting_suffix}"
            print(f"----- Cross Validation Index: {i+1} -----")
            res = learning_process(args_dict=args_dict, setting_suffix=_setting_suffix, CV_idx=i, fold_num=fold_num)
            if "cls" in args.task:
                print(f"Test accuracy: {res}")
            else: 
                print(f"Test loss: {res}")
            result_list.append(res)
        print("----- Final Result -----")
        print("List of result: ", result_list)
        avg = sum(result_list) / fold_num
        print("Average result: ", avg)
        print("Result std: ", np.sqrt(sum([(x-avg)**2 for x in result_list])/fold_num))
    elif args.exp_id is not None:
        _setting_suffix = f"PointNet_{args.exp_id},{setting_suffix}"
        learning_process(args_dict, setting_suffix=_setting_suffix)
    else:
        _setting_suffix = f"PointNet,{setting_suffix}"
        learning_process(args_dict, setting_suffix=_setting_suffix)