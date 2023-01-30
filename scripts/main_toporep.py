import numpy as np
import matplotlib.pyplot as plt
from gudhi.dtm_rips_complex import DTMRipsComplex
import torch
import torch.nn as nn
import random
import datetime
import time
from pprint import pprint
from copy import deepcopy

import os
import argparse

from lib.utils import *
from lib.toporep import *
from lib.pointnet import *

def learning_process(args_dict, setting_suffix, CV_idx=None, fold_num=10):
    """
    Learn topological representations.
    
    If CV_idx != None, cross validation will be conducted.
    `fold_num` is a folding number for cross-validation, and `CV_idx` represents which index this learning process is.
    """
    start_time = time.time()

    ### Settings for data ###
    exp_mode = args_dict.get("exp_mode")
    task = args_dict.get("task")
    input_format = args_dict.get("input_format")
    if input_format == "dist":
        translation = False
    else:
        translation = (args_dict.get("translation", 1) == 1)
    if input_format == "dist":
        rot = False
    else:
        rot = (args_dict.get("rot") == 1)

    ### Settings for optimization ###
    optimizer = args_dict.get("optimizer")
    reg = args_dict.get("reg") # regression (none/l1/l2)
    iter_num = args_dict.get("iter_num")
    batch_size = args_dict.get("batch_size")
    lr = float(args_dict.get("lr"))
    after_opt = (args_dict.get("after_opt") == 1) # optimize the task-solving network after learning representation

    ### Reading data files & evaluation metrics ###
    trainX = torch.load(f"data/{exp_mode}_train").detach().to(torch.float32)
    testX = torch.load(f"data/{exp_mode}_test").detach().to(torch.float32)
    if "cls" in task:
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.long)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.long)
        second_step_criterion = nn.CrossEntropyLoss()
        third_step_criterion = nn.CrossEntropyLoss()
        output_dim = int(task.split("_cls")[0])
    elif task == "reg":
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.float32)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.float32)
        second_step_criterion = nn.MSELoss()
        third_step_criterion = nn.MSELoss()
        output_dim = 1
    elif task == "ae":
        trainy = torch.load(f"data/{exp_mode}_train_label").detach().to(torch.float32)
        testy = torch.load(f"data/{exp_mode}_test_label").detach().to(torch.float32)
        def second_step_criterion(pred, gt): 
            return wasserstein_loss(torch.stack([pred[i, :].view(gt.shape[1], gt.shape[2]) for i in range(gt.shape[0])]), gt)
        def third_step_criterion(pred, gt): 
            return wasserstein_loss(torch.stack([pred[i, :].view(gt.shape[1], gt.shape[2]) for i in range(gt.shape[0])]), gt)
        output_dim = trainy.shape[1] * trainy.shape[2]
    else:
        raise NotImplementedError
    
    ## When we conduct cross validation, training dataset is splitted in to `fold_num` and the one of them is used as test data. ##
    if CV_idx is not None:
        train_boolian_slice = [i % fold_num != CV_idx for i in range(trainX.shape[0])]
        test_boolian_slice = [i % fold_num == CV_idx for i in range(trainX.shape[0])]
        testX = trainX[test_boolian_slice, :, :]
        trainX = trainX[train_boolian_slice, :, :]
        testy = trainy[test_boolian_slice]
        trainy = trainy[train_boolian_slice]
        
    point_num = trainX.shape[1]

    ## translation ##
    if translation:
        trainX = trainX - torch.stack([torch.mean(trainX, dim=1) for _ in range(trainX.shape[1])], dim=1)
        testX  = testX  - torch.stack([torch.mean(testX,  dim=1) for _ in range(testX.shape[1])], dim=1)
    
    ### Initialization of network ###
    if task == "2_cls":
        output_dim = 2
        last_linear = False
    elif task == "10_cls":
        output_dim = 10
        last_linear = False
    elif task == "reg":
        output_dim = 1
        last_linear = False
    elif task == "ae":
        output_dim = trainX.shape[2] * trainX.shape[1]
        last_linear = True
    model = TopoRep(args_dict, 
        input_dim=trainX.shape[2], output_dim=output_dim, point_num=trainX.shape[1], 
        last_linear=last_linear
    )

    ### Optimized parameters ###
    params = []
    for x in model.delay_calc_params:
        if type(x[0]) is list:
            for y in x:
                params += y
        else:
            params += x
    params += model.cls_params
    params += model.out_feature_params
    params += model.reducer_params

    ### Defining Optimizer ###
    if optimizer == "sgd":
        opt_third_step = torch.optim.SGD(params, lr=lr)
    elif optimizer == "adam":
        opt_third_step = torch.optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError

    ### If `method` == "dtm", PI can be pre-calculated, which makes optimization efficient ##
    if model.method == "dtm":
        train_PI = model.get_heatmap_tensors(trainX)
    batch_loss_list = []

    for step in range(iter_num):
        batch_sample = random.sample(range(trainX.shape[0]), batch_size)
        if model.method == "dtm":
            batch_predict = model.entire_network(trainX[batch_sample, :, :], heatmap_tensors=train_PI[batch_sample, :, :],rot=rot)
        else:
            batch_predict = model.entire_network(trainX[batch_sample, :, :], rot=rot)

        if reg == "none":
            batch_loss = second_step_criterion(batch_predict, trainy[batch_sample])
        elif reg == "l1":
            if task == "ae":
                batch_loss = second_step_criterion(batch_predict, trainy[batch_sample]) + torch.linalg.norm(model.reducer_params[0], ord=1)
            else:
                batch_loss = second_step_criterion(batch_predict, trainy[batch_sample]) + torch.linalg.norm(model.cls_params[0], ord=1)
        elif reg == "l2":
            if task == "ae":
                batch_loss = second_step_criterion(batch_predict, trainy[batch_sample]) + torch.linalg.norm(model.reducer_params[0]) ** 2
            else:
                batch_loss = second_step_criterion(batch_predict, trainy[batch_sample]) + torch.linalg.norm(model.cls_params[0]) ** 2
        else:
            raise NotImplementedError

        opt_third_step.zero_grad()
        batch_loss.backward()
        batch_loss_list.append(float(batch_loss))

        if step % 100 == 0:
            print(f"Step {step}: batch loss = {float(batch_loss)}", flush=True) 
        
        opt_third_step.step()

    ### Optimization of task_solver after learning representations ###
    if after_opt:
        for x in model.delay_calc_params:
            for y in x:
                if type(y) is list:
                    for z in y:
                        z.requires_grad = False
                else:
                    y.requires_grad = False
        for x in model.out_feature_params:
            x.requires_grad = False
        for x in model.reducer_params:
            x.requires_grad = False
        opt_final = torch.optim.Adam(model.cls_params, lr=lr)
        if model.method != "dtm":
            train_PI = model.get_heatmap_tensors(trainX)
        for step in range(3000):
            batch_sample = random.sample(range(trainX.shape[0]), 100)
            batch_predict = model.entire_network(
                trainX[batch_sample, :, :], 
                heatmap_tensors=train_PI[batch_sample, :, :], 
                rot=rot
            )
            batch_loss = third_step_criterion(batch_predict, trainy[batch_sample])
            opt_final.zero_grad()
            batch_loss.backward()
            opt_final.step()

            if step % 1000 == 0:
                print(f"Step {step}: batch loss = {float(batch_loss)}", flush=True) 

    ret = {}
    
    with torch.no_grad():
        train_output = model.entire_network(trainX)
        train_pred_label = [int(x) for x in torch.max(train_output, dim=1).indices]
        count = sum([x == y for x, y in zip(train_pred_label, trainy)])
        print(f"Loss (train): {third_step_criterion(train_output, trainy)}", flush=True) 
        if "cls" in task: 
            print(f"Accuracy (train): {float(count/trainX.shape[0])}", flush=True) 

        test_output = model.entire_network(testX, rot=rot)
        test_pred_label = [int(x) for x in torch.max(test_output, dim=1).indices]
        count = sum([x == y for x, y in zip(test_pred_label, testy)])
        print(f"Loss (test): {third_step_criterion(test_output,testy)}", flush=True) 
        if "cls" in task: 
            print(f"Accuracy (test): {float(count/testX.shape[0])}", flush=True) 
    
    if "cls" in task:
        ret["Test_accuracy"] = 100*count/testX.shape[0]
    else:
        ret["Test_loss"] = float(third_step_criterion(test_output, testy))

    for k in ret.keys():
        if type(ret[k]).__module__ == "torch":
            ret[k] = ret[k].detach().numpy()
    
    end_time = time.time()
    td = datetime.timedelta(seconds=end_time - start_time)
    print("Execution time: ", str(td))

    return ret

if __name__ == "__main__":
    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9),'JST')).strftime('%Y%m%d')

    psr  = argparse.ArgumentParser(
        prog='topological representation learning',
        usage='usage',
        description='descripsion'
    )

    psr.add_argument('-m', '--method', default="dist", help='dist/dtm')
    psr.add_argument('-o', '--out_feature', default="pointnet", help='none/pointnet/deepsets. DNN method concatenated with topological reporesentation.')
    psr.add_argument('-no', '--normalize', type=int, default=1, help='If set 1, Normalize feature vector')
    psr.add_argument('-in', '--input_format', default="coord", help='coord/dist. Input format can be coordinate and distance matrix.')
    psr.add_argument('-e', '--exp_mode', default="ART06,nonrotae,200,60", help='Experiment data will be accessed with `data/{exp_mode}_train`.')
    psr.add_argument('-ts', '--task', default="ae", help='2_cls/10_cls/reg/ae')
    psr.add_argument('-r', '--rot', type=int, default=1, help='If set 1, input data will be randomly rotated as long as the input format is coordinate.')
    psr.add_argument('-tl', '--translation', type=int, default=1, help='If set 1, input data will be translated to locate its center at 0 as long as the input format is coordinate.')
    psr.add_argument('-om', '--optimizer', default="adam", help='sgd/adam')
    psr.add_argument('-re', '--reg', default="none", help='none/l1/l2')
    psr.add_argument('-it', '--iter_num', type=int, default=3000, help='# of iteration.')
    psr.add_argument('-b', '--batch_size', type=int, default=30, help='batch size.')
    psr.add_argument('-l', '--lr', default="1e-2", help='Learning rate. 1e-2 for cls/ae and 1e-3 for reg is recommended.')
    psr.add_argument('-ao', '--after_opt', type=int, default=1, help='If set 1, solver will be solely optimized after learning of representation.')
    psr.add_argument('-c', '--cls_model', default="deep", help='deep/linear, network to solve task.')
    psr.add_argument('-t', '--date', default=date_str, help='')
    psr.add_argument('-id', '--exp_id', default=None, help='experimetal index or designate 5fold/10fold')
    psr.add_argument('-k', '--dtm_k', type=int, default=2, help='Parameter of DTM if method == "dtm".')

    args = psr.parse_args()
    args_dict = vars(args)

    ### Output of settings and designating file name ###
    no_output_settings = {"input_format", "rot", "translation", "date", "init_mode", "base_filt", "PI_weight", "PI_grid_num", "PI_h", "delay_actmax", "ds_op"}
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
            _setting_suffix = f"FL4PC_-{i+1}of{fold_num}{setting_suffix}" 
            print(f"----- Cross Validation Index: {i+1} -----")
            res = learning_process(args_dict=args_dict, setting_suffix=_setting_suffix, CV_idx=i, fold_num=fold_num)
            pprint(res)
            result_list.append(res)
        print("----- Final Result -----")
        for col in result_list[0].keys():
            avg = sum([x[col] for x in result_list]) / fold_num
            print(f"Average of {col}: {avg}")
            print(f"Std of {col}: {np.sqrt(sum([(x[col]-avg)**2 for x in result_list])/fold_num)}")
    elif args.exp_id is not None:
        _setting_suffix = f"FL4PC_{args.exp_id},{setting_suffix}"
        learning_process(args_dict, setting_suffix=_setting_suffix)
    else:
        _setting_suffix = f"FL4PC,{setting_suffix}"
        learning_process(args_dict, setting_suffix=_setting_suffix)