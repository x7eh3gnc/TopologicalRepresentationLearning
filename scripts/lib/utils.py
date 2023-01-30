import torch
import numpy as np
import gudhi as gd
from gudhi.weighted_rips_complex import WeightedRipsComplex
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

def PI_value(grid, points, h=1, h_list=None, PI_weight="linear", density="gaussian"):
    if type(grid).__module__ != "torch":
        grid = torch.tensor(grid).to(torch.float32)
    ret = torch.zeros_like(grid[:, :, 0])
    
    if density == "gaussian":
        # for p, q in _points:
        for p, q in points:
            dist = torch.tensor([[torch.linalg.norm(grid[i, j, :] - torch.tensor([p, q])) for j in range(grid.shape[1])] for i in range(grid.shape[0])])
            if PI_weight == "linear":
                ret += (1 / (h**2)) * q * torch.exp(- (dist**2 / (2*(h**2))))
            elif PI_weight == "quad": 
                ret += (1 / (h**2)) * (q ** 2)* torch.exp(- (dist**2 / (2*(h**2))))
            elif PI_weight == "tanh":
                ret += (1 / (h**2)) * torch.tanh(q) * torch.exp(- (dist**2 / (2*(h**2))))
    elif density == "cone":
        for p, q in points:
            dist = torch.tensor([[torch.linalg.norm(grid[i, j, :] - torch.tensor([p, q])) for j in range(grid.shape[1])] for i in range(grid.shape[0])])
            if PI_weight == "linear":
                ret += (6 / (np.pi * (h**2))) * q * torch.fmin(h - dist, torch.zeros_like(dist))
            elif PI_weight == "quad": 
                ret += (6 / (np.pi * (h**2))) * (q ** 2)* torch.fmin(h - dist, torch.zeros_like(dist))
            elif PI_weight == "tanh":
                ret += (6 / (np.pi * (h**2))) * torch.tanh(q) * torch.fmin(h - dist, torch.zeros_like(dist))
    else:
        raise NotImplementedError
    return ret

def MLP_solver(heatmap_tensors, **kwargs):
    value = heatmap_tensors.reshape(heatmap_tensors.shape[0],heatmap_tensors.shape[1]*heatmap_tensors.shape[2])
    return MLP_network(value, **kwargs)

def MLP_network(value, weight, bias, activation=torch.tanh, task="cls", last_linear=False):
    for i in range(len(weight)-1):
        value = activation(torch.einsum("bi,ik->bk", value, weight[i]) + bias[i])
    if last_linear:
        value = torch.einsum("bi,ik->bk", value, weight[-1]) + bias[-1]
    else:
        value = activation(torch.einsum("bi,ik->bk", value, weight[-1]) + bias[-1])

    if "cls" in task:
        m = torch.nn.Softmax(dim=1)
        value = m(value)
    return value

def tensor_list_detach(tensor_list):
    return [x.detach() for x in tensor_list]

def get_simplex_tree(X, delay=None, base_filt="rips", input_format="coord", max_dim=2):
    if delay is None:
        delay = torch.zeros_like(X[:, 0])

    if input_format == "coord":
        dist = torch.cdist(X, X)
    elif input_format == "dist":
        dist = X.clone()
    else:
        raise NotImplementedError

    if base_filt == "rips":
        dist = dist.detach()
        dist += torch.stack([delay for _ in range(X.shape[0])], dim=0)
        dist += torch.stack([delay for _ in range(X.shape[0])], dim=1)
        filt = gd.RipsComplex(
            distance_matrix = dist, 
            max_edge_length = np.inf
        ) 
    elif base_filt == "weighted":
        filt = WeightedRipsComplex(
            distance_matrix = dist, 
            weights = delay
        )
    else:
        raise NotImplementedError
    
    return filt.create_simplex_tree(max_dimension=max_dim)

def get_one_heatmap_tensor(X, delay=None, base_filt="rips", input_format="coord", PI_weight="linear", pers_pair=None, h=1, pi_range_x=2.0, pi_range_y=2.0, grid_num_x=10, grid_num_y=10):
    if type(X).__module__ != "torch":
        X = torch.tensor(X).to(torch.float32)
    if (delay is not None) and (type(delay).__module__ != "torch"):
        delay = torch.tensor(delay).to(torch.float32)
    if delay is None:
        delay = torch.zeros_like(X[:, 0])

    if pers_pair is None:
        simplex_tree = get_simplex_tree(X, delay, base_filt=base_filt, input_format=input_format, max_dim=2)
        barcode = simplex_tree.persistence()
        pers_pair = simplex_tree.persistence_pairs()
        
    points = []
    if input_format == "coord":
        dist = torch.cdist(X, X)
    elif input_format == "dist":
        dist = X.clone()
    else:
        raise NotImplementedError
    dist += torch.stack([delay for _ in range(X.shape[0])], dim=0)
    dist += torch.stack([delay for _ in range(X.shape[0])], dim=1)
    for x in pers_pair:
        if len(x[0]) == 2:
            points.append(
                (
                    dist[x[0][0], x[0][1]], 
                    torch.max(dist[x[1][0], x[1][1]], torch.max(dist[x[1][1], x[1][2]], dist[x[1][2], x[1][0]])) - dist[x[0][0], x[0][1]]
                )
            )
        elif len(x[0]) == 3:
            raise NotImplementedError
    
    grid = torch.tensor([[[x, y] for y in list(np.linspace(0, pi_range_y, grid_num_y))] for x in list(np.linspace(0, pi_range_x, grid_num_x))]).to(torch.float32)
    return PI_value(grid, points, h=0.5)

def get_heatmap_tensors(X, delay=None, base_filt="rips", input_format="coord", PI_weight="linear", h=1, pi_range_x=2.0, pi_range_y=2.0, grid_num_x=10, grid_num_y=10):
    if delay is None:
        delay = torch.zeros_like(X[:, :, 0])

    pers_pair_list = [None for _ in range(X.shape[0])]

    heatmap_tensor_list = []
    for i in range(X.shape[0]): 
        heatmap_tensor = get_one_heatmap_tensor(
            X=X[i, :, :], delay=delay[i, :], 
            base_filt=base_filt, input_format=input_format, PI_weight=PI_weight, 
            pers_pair=pers_pair_list[i], 
            h=h, pi_range_x=pi_range_x, pi_range_y=pi_range_y, grid_num_x=grid_num_x, grid_num_y=grid_num_y
        )
        heatmap_tensor_list.append(heatmap_tensor)

    return torch.stack(heatmap_tensor_list, dim=0)

def deepsets(X, weight1, weight2, bias1, bias2, activation=torch.tanh, ds_op="sum"):
    """
    -----------
    ## parameters

    X: input tensor with shape (# data, # points, dim)
    ds_op: "sum" or "max" or "concat"

    ## output
    tensor with shape (# data, feature vector dim)
    """
    value = X.clone()
    for i in range(len(weight1)):
        value = activation(torch.einsum("bij,jk->bik", value, weight1[i]) + bias1[i]) 
    if ds_op == "max":
        value = torch.max(value, axis=1).values
    elif ds_op == "sum":
        value = torch.sum(value, axis=1)
    elif ds_op == "concat":
        value = torch.cat([torch.max(value, axis=1).values, torch.sum(value, axis=1)], dim=1)
    else:
        raise NotImplementedError
    for i in range(len(weight2)):
        value = activation(torch.einsum("bj,jk->bk", value, weight2[i]) + bias2[i])
    return value

def MLP_layers_init(neuron_num_list, loc=0, bias=0.1):
    weight_list = []
    bias_list = []
    for i in range(len(neuron_num_list)-1):
        weight_list.append(torch.normal(loc, bias, size=(neuron_num_list[i], neuron_num_list[i+1]), requires_grad=True))
        bias_list.append(torch.normal(loc, bias, size=(neuron_num_list[i+1], ), requires_grad=True))
    return weight_list, bias_list

def draw_heatmap(heatmap_array, ax, vmin=0.0, vmax=2.0):
    if type(heatmap_array).__module__ == "torch":
        heatmap_array = heatmap_array.detach().numpy()
        
    sns.heatmap(  
        pd.DataFrame(heatmap_array.T).iloc[::-1], 
        cmap="jet", 
        xticklabels=True, 
        yticklabels=True, 
        vmin = vmin, 
        vmax = vmax, 
        ax = ax
    )

def wasserstein_loss(x1, x2):
    """
    x1, x2: torch.tensor  (# data, # points, dim)
    """
    dist = torch.cdist(x1, x2)
    ret_list = []
    for i in range(x1.shape[0]):
        a = np.ones(x1.shape[1]) / x1.shape[1]
        b = np.ones(x2.shape[1]) / x2.shape[1]
        P = torch.tensor(ot.emd(a, b, dist[i, :, :].detach().numpy())).to(torch.float32)
        ret_list.append(torch.einsum("ij,ij", P, dist[i, :, :]))
    return sum(ret_list) / len(ret_list)

def random_rotation(X):
    """
    X: torch.tensor  (# data, # points, dim)
    """
    # creating a list of rotation matrix
    theta = 2 * np.pi * np.random.rand(X.shape[0])
    if X.shape[2] == 3:
        rot_mat_list = [
            torch.tensor([
                [np.cos(theta[k]), -np.sin(theta[k]), 0.], 
                [np.sin(theta[k]),  np.cos(theta[k]), 0.], 
                [0., 0., 1.]
            ]).to(torch.float32) for k in range(X.shape[0])
        ]
    else:
        rot_mat_list = [
            torch.tensor([
                [np.cos(theta[k]), -np.sin(theta[k])], 
                [np.sin(theta[k]),  np.cos(theta[k])], 
            ]).to(torch.float32) for k in range(X.shape[0])
        ]
    rotated_X = torch.stack([torch.einsum("ij,kj->ki", A, X[l, :, :]) for l, A in enumerate(rot_mat_list)], dim=0)
    return rotated_X

class DeepSets:
    def __init__(self, input_dim, output_dim, init_params=None):
        if init_params is None:
            self.ds_weight1, self.ds_bias1 = MLP_layers_init([input_dim, 128, 128, 128, 64])
            self.ds_weight2, self.ds_bias2 = MLP_layers_init([64, 128, 128, 128, output_dim])
        else:
            if len(init_params[4].shape) == 1:
                self.ds_weight1, self.ds_bias1 = init_params[:4], init_params[4:8]
                self.ds_weight2, self.ds_bias2 = init_params[8:12], init_params[12:16]
            else:
                self.ds_weight1, self.ds_bias1 = init_params[:4], init_params[8:12]
                self.ds_weight2, self.ds_bias2 = init_params[4:8], init_params[12:16]

    def output(self, X):
        return deepsets(X, self.ds_weight1, self.ds_weight2, self.ds_bias1, self.ds_bias2, activation=torch.relu)

class MultiLayerPerceptron:
    def __init__(self, width_list, activation=torch.tanh, task="cls", last_linear=False, loc=0.0, bias=0.1, weight_init=None, bias_init=None):
        if (weight_init is not None) and (bias_init is not None):
            self.weight, self.bias = weight_init, bias_init
        else:
            self.weight, self.bias = MLP_layers_init(width_list, loc=loc, bias=bias)
        self.activation = activation
        self.task = task
        self.last_linear = last_linear

    def output(self, value):
        return MLP_network(value, weight=self.weight, bias=self.bias, activation=self.activation, task=self.task, last_linear=self.last_linear)