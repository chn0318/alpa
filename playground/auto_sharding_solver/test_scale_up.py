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

def get_llm_computation(num_layers, seq_len, batch_size, model_dim, hidden_dim, vocab_size):
    computation = HloComputation()
    with computation:
        # 参数定义
        x = HloParameter((batch_size, seq_len, vocab_size))  # 输入序列 (one-hot encoded)
        y = HloParameter((batch_size, seq_len, vocab_size))  # 目标序列 (one-hot encoded)
        embeddings = HloParameter((vocab_size, model_dim))  # 词嵌入矩阵
        positional_encoding = HloParameter((seq_len, model_dim))  # 位置编码
        layers = []

        for i in range(num_layers):
            layers.append({
                "w_q": HloParameter((model_dim, model_dim)),  # Query 权重
                "w_k": HloParameter((model_dim, model_dim)),  # Key 权重
                "w_v": HloParameter((model_dim, model_dim)),  # Value 权重
                "w_out": HloParameter((model_dim, model_dim)),  # Attention输出权重
                "w_ff1": HloParameter((model_dim, hidden_dim)),  # FFN第一层权重
                "w_ff2": HloParameter((hidden_dim, model_dim)),  # FFN第二层权重
            })

        # 前向传播
        h_now = HloDot(x, embeddings)  # 嵌入层
        h_now = HloAdd(h_now, HloBroadcast(positional_encoding, (batch_size, seq_len, model_dim)))

        for i in range(num_layers):
            # 多头自注意力
            q = HloDot(h_now, layers[i]["w_q"])  # Query
            k = HloDot(h_now, layers[i]["w_k"])  # Key
            v = HloDot(h_now, layers[i]["w_v"])  # Value

            attention_scores = HloDot(q, k, lhs_contracting_dims=(2,), rhs_contracting_dims=(2,))
            attention_scores = HloSoftmax(attention_scores, axis=2)
            attention_output = HloDot(attention_scores, v)
            attention_output = HloDot(attention_output, layers[i]["w_out"])  # 输出变换

            # 残差连接 + LayerNorm
            h_now = HloAdd(h_now, attention_output)
            h_now = HloLayerNorm(h_now)

            # 前向FFN
            ff1 = HloDot(h_now, layers[i]["w_ff1"])
            ff2 = HloDot(ff1, layers[i]["w_ff2"])
            h_now = HloAdd(h_now, ff2)  # 残差连接
            h_now = HloLayerNorm(h_now)

        # 输出层 + 损失计算
        logits = HloDot(h_now, HloTranspose(embeddings, perm=(1, 0)))  # 投影回词汇表
        loss = HloSoftmaxCrossEntropy(logits, y)

        # 反向传播
        grad_loss = HloGradient(loss)  # 假设存在梯度计算接口
        new_layers = []
        for i in range(num_layers - 1, -1, -1):
            # 更新FFN权重
            grad_w_ff2 = HloDot(HloTranspose(layers[i]["w_ff1"]), grad_loss)
            new_w_ff2 = HloSubtract(layers[i]["w_ff2"], grad_w_ff2)

            grad_w_ff1 = HloDot(HloTranspose(layers[i]["w_ff2"]), grad_loss)
            new_w_ff1 = HloSubtract(layers[i]["w_ff1"], grad_w_ff1)

            # 更新Attention权重
            grad_w_out = HloDot(HloTranspose(layers[i]["w_v"]), grad_loss)
            new_w_out = HloSubtract(layers[i]["w_out"], grad_w_out)

            grad_w_v = HloDot(HloTranspose(layers[i]["w_k"]), grad_loss)
            new_w_v = HloSubtract(layers[i]["w_v"], grad_w_v)

            grad_w_k = HloDot(HloTranspose(layers[i]["w_q"]), grad_loss)
            new_w_k = HloSubtract(layers[i]["w_k"], grad_w_k)

            grad_w_q = HloDot(HloTranspose(h_now), grad_loss)
            new_w_q = HloSubtract(layers[i]["w_q"], grad_w_q)

            # 存储更新后的权重
            new_layers.append({
                "w_q": new_w_q,
                "w_k": new_w_k,
                "w_v": new_w_v,
                "w_out": new_w_out,
                "w_ff1": new_w_ff1,
                "w_ff2": new_w_ff2,
            })

        # 更新嵌入层权重
        grad_embeddings = HloDot(HloTranspose(x), grad_loss)
        new_embeddings = HloSubtract(embeddings, grad_embeddings)

        # 输出新的权重
        out = HloTuple([new_embeddings] + [layer for layer in reversed(new_layers)])

        # 设置别名优化
        alias_list = [(embeddings, new_embeddings)] + [
            (layers[i][key], new_layers[num_layers - 1 - i][key])
            for i in range(num_layers)
            for key in layers[i]
        ]
        computation.set_alias(alias_list)

    return computation
execution_times = []
num_layers = 12
batch_size = 128
hidden_dim = 1024

computation = get_mlp_n_layer_computation(num_layers, batch_size, hidden_dim,hidden_dim, hidden_dim)
# N: the number of nodes;
# Assume each node has 4 GPU
for N in [1,5,25,50,100,200,500,1000,3000]:
    print("Node size =", N)
    start_time = time.time()
    device = [(i, 4) for i in range(1, N + 1)]
    with contextlib.redirect_stdout(io.StringIO()):
        for mesh_shape in device:
            device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
            cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                             memory_per_device=1000 * MB)
            objective = solve_auto_sharding(computation, cluster_env)

    end_time = time.time()
    execution_times.append(end_time - start_time)
    print("Time elapse", end_time - start_time)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), execution_times, label="Execution Time")
plt.xlabel("N (Number of devices)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time per Iteration as N increases")
plt.legend()
plt.grid(True)

plt.savefig("execution_time_plot.png")