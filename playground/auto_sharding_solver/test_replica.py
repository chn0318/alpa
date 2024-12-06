from collections import defaultdict
from enum import Enum
import io
import numpy as np
import time
import matplotlib.pyplot as plt
import contextlib
from hlo import *
from cluster_env import ClusterEnvironment
from solver import solve_auto_sharding, SolverOption

MB = 1024 ** 2
def get_mlp_n_layer_computation(num_layers, batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w_first = HloParameter((input_dim, hidden_dim))
        w_inter = []
        for i in range(num_layers - 2):
            manual_strategy = "S0" if i % 2 == 0 else "S1"
            w_inter.append(HloParameter((hidden_dim, hidden_dim)))
        w_last = HloParameter((hidden_dim, output_dim))

        # forward
        h_first = HloDot(x, w_first)
        h_now = h_first
        h_inter = []
        for i in range(num_layers - 2):
            h_now = HloDot(h_now, w_inter[i])
            h_inter.append(h_now)
        h_last = HloDot(h_now, w_last)

        loss = HloSubtract(h_last, y)

        # backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)
        grad_h_now = grad_loss

        grad_w_last = HloDot(h_inter[-1], grad_h_now,
                             lhs_contracting_dims=(0,),
                             rhs_contracting_dims=(0,),)
        new_w_last = HloSubtract(w_last, grad_w_last)
        grad_h_now = HloDot(grad_h_now, w_last,
                             lhs_contracting_dims=(1,),
                             rhs_contracting_dims=(1,),)

        new_w_inter = []
        for i in range(num_layers - 3, -1, -1):
            grad_w = HloDot(h_inter[i-1], grad_h_now,
                            lhs_contracting_dims=(0,),
                            rhs_contracting_dims=(0,),)
            new_w = HloSubtract(w_inter[i], grad_w)
            grad_h_now = HloDot(grad_h_now, w_inter[i],
                                lhs_contracting_dims=(1,),
                                rhs_contracting_dims=(1,),)
            new_w_inter.append(new_w)

        grad_w_first = HloDot(x, grad_h_now,
                              lhs_contracting_dims=(0,),
                              rhs_contracting_dims=(0,),)
        new_w_first = HloSubtract(w_first, grad_w_first)

        out = HloTuple([new_w_first] + new_w_inter + [new_w_last])

        # alias
        alias_list = [(w_first, new_w_first), (w_last, new_w_last)] +\
            [(w_old, w_new) for w_old, w_new in zip(w_inter, reversed(new_w_inter))]
        computation.set_alias(alias_list)
    return computation

class operator_device_pair:
    def __init__(self, op, device):
        if not isinstance(op, int):
            raise TypeError("op must be an integer.")
        if not isinstance(device, tuple):
            raise TypeError("device must be a tuple.")
        
        self.op = op
        self.device = device

    def __hash__(self):
        return hash((self.op, self.device))

    def __eq__(self, other):
        return isinstance(other, operator_device_pair) and \
               self.op == other.op and \
               self.device == other.device

    def __repr__(self):
        return f"operator_device_pair(op={self.op}, device={self.device})"


execution_times = []
num_layers = 12
batch_size = 128
hidden_dim = 1024
node_number = 500

my_dict = {}

def update_dict(key):
    if key in my_dict:
        my_dict[key] += 1 
    else:
        my_dict[key] = 1

for i in range(1,num_layers):
    for j in range(i+1, num_layers+1):
        computation = get_mlp_n_layer_computation(j-i+1, batch_size, hidden_dim,hidden_dim, hidden_dim)
        device = [(i, 4) for i in range(1, node_number + 1)]
        with contextlib.redirect_stdout(io.StringIO()):
            for mesh_shape in device:
                pair = operator_device_pair(j-i+1,mesh_shape)
                device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
                cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],memory_per_device=1000 * MB)
                objective = solve_auto_sharding(computation, cluster_env)
                update_dict(pair)
        
sorted_values = [item[1] for item in sorted(my_dict.items(), key=lambda x: x[1], reverse=True)]
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_values)), sorted_values)
plt.xlabel('Operator-device pair', fontsize=14)
plt.ylabel('Hit times', fontsize=14)
plt.title('Cache hit times for Operator-device pair', fontsize=16)
plt.xticks([])
plt.show()