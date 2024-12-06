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

execution_times = []
num_layers = 32
batch_size = 128
hidden_dim = 4096

computation = get_mlp_n_layer_computation(num_layers, batch_size, hidden_dim,hidden_dim, hidden_dim)
# N: the number of nodes;
# Assume each node has 4 GPU
for N in [200]:
    print("Node size =", N)
    start_time = time.time()
    device = [(i, 4) for i in range(1, N + 1)]
    for mesh_shape in device:
        device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
        cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                             memory_per_device=1000 * MB)
        objective = solve_auto_sharding(computation, cluster_env)

    end_time = time.time()
    execution_times.append(end_time - start_time)
    print("Time elapse", end_time - start_time)


# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 10), execution_times, label="Execution Time")
# plt.xlabel("N (Number of devices)")
# plt.ylabel("Execution Time (seconds)")
# plt.title("Execution Time per Iteration as N increases")
# plt.legend()
# plt.grid(True)

# plt.savefig("execution_time_plot.png")