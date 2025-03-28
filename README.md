# Alpa simulation

### 1. Setup

**Experiment environment：**

Operating System: Ubuntu 22.04.5 LTS

Kernel Version: 5.15.0-125-generic

CUDA Version: 12.1.1

NVIDIA Driver Version: 535.216.03

cuDNN Version: 8.9.7.29

#### Install Docker：

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```bash
# Install the Docker packages.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

####  Install NVIDIA Container Toolkit 

```bash
# Configure the production repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
# Install the NVIDIA Container Toolkit packages
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### Start Container

**Image link:**

https://hub.docker.com/r/jiaodong/alpa

```python
# Start Container
docker run --gpus all \
  --network host \
  --shm-size=9.51gb \
  -v ~/alpa:/alpa \
  -it jiaodong/alpa /bin/bash
```

`--gpus all`:
It allows the container to access all available GPUs on the host machine. It requires NVIDIA Container Toolkit to be installed.

`--network host`:
This option makes the container share the host's network stack. It allows the container to access network interfaces and ports exactly as if it were running on the host directly. 

`--shm-size=9.51gb`:
 This sets the size of the container’s shared memory (/dev/shm) to 9.51 GB.



### 2. Key function Analysis

- ##### `solve_auto_sharding():` 

This function is the core solver in Alpa's intra-op parallelization framework.

It takes an HLO computation and a cluster environment as input, then constructs and solves an Integer Linear Programming (ILP) problem to determine the most efficient sharding strategy for each operator in the computation graph.
**Input:**

`computation`: The HLO module or computation graph to be parallelized.

`cluster_env`: Cluster environment metadata, including device mesh topology, memory, and communication cost models.

`solver_option`: Optional parameters controlling the solver's behavior.

```python
def solve_auto_sharding(computation, cluster_env, solver_option=None):
	......
    # For each HLO instruction, enumerate possible sharding strategies,
    # and compute the cost metrics for each strategy, including:
    # - Communication cost
    # - Memory cost
    # - Resharding cost (when input layout doesn't match required layout)
    computation.build_strategy_and_cost(cluster_env, solver_option)

    # Build all constants for ILP
    N = len(computation.instructions)
    M = cluster_env.memory_per_device
	
    ...
    # Traverse each HLO instruction
    for i in range(N):
        ins = computation.instructions[i]
        s_len.append(len(ins.strategies))
        L.append([ins.index for ins in liveness_dict[i]])
        c.append(ins.compute_costs)        # Compute cost for each strategy
        d.append(ins.communication_costs)  # Communication cost for each strategy
        m.append(ins.memory_costs)         # Memory usage for each strategy

        if ins.follow_ins is not None:
            follow_pair.append((ins.index, ins.follow_ins.index))

        for op_idx, operand in enumerate(ins.operands):
            # Record the dependency edge: from operand (source) to this instruction (destination)
            E.append((operand.index, i))

            src = operand.index  # The index of the operand instruction (producer)
            dst = i              # The index of the current instruction (consumer)

            cost = []
            # For all combinations of strategies between producer and consumer,
            # retrieve the resharding cost needed to transform the src's output
            # into the required input format for the dst
            for p in range(len(computation.instructions[src].strategies)):     # src's strategies
                for q in range(len(computation.instructions[dst].strategies)):  # dst's strategies
                    cost.append(ins.resharding_costs[q][op_idx][p])
            # Add the resharding cost list for this edge (resharding cost matrix: q × p)
            r.append(cost)


    # Simplify the graph by merging nodes
	...

    # Deal with alias
	...
    

    #s_val       List[int]: selected strategy index for each instruction
    #e_val       List[int]: selected edge resharding plan for each dependency edge
    #objective   float:     total cost of the selected sharding plan (objective function value)
    s_val, e_val, objective, status = call_solver(N, M, s_len, s_follow, E, A, L,
                                                  c, d, m, r, v, s_init=None)
    
  
    # Print Result
	...
    
    return objective

```

- `call_solver()`   

This function convert the structure list data into flat NumPy arrays, then pass them to the lower-level `_call_solver_serialized_args()` function to perform the actual optimization.

- `_call_solver_serialized_args()`

This function performs the core ILP optimization for Alpa's intra-op process.

The mathematical formulation of the ILP problem is: 

P1: minimize  
∑₍ᵥ ∈ V₎ sᵥᵀ (cᵥ + dᵥ) + ∑₍ₑᵤᵥ ∈ E₎ sᵤᵀ Rᵤᵥ sᵥ

Subject to:  
sᵥ ∈ {0,1}^{kᵥ}, and ∑ᵢ sᵥᵢ = 1 for all v ∈ V

```python
def _call_solver_serialized_args(...):
    ...

    # === Build the ILP problem ===
    # Builds an ILP problem using the pulp optimization library, with:
    # - Objective: Minimize total compute, communication, and resharing costs
    # - Constraints:
    #   - Each node chooses exactly one strategy
    #   - Total memory usage at any time must not exceed device budget
    #   - Edge resharding strategies must be consistent with node strategies
    #   - Alias instructions must not use incompatible strategies

    prob = LpProblem("myProblem", LpMinimize)

    # Objective function: minimize compute + communication + resharing cost
    obj = 0
    for i in range(N):
        obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])  # compute + comm
    for i in range(len(E)):
        obj += lpDot(e[i], r[i])  # resharing
    prob += obj

	...
    
    # === Solve the ILP problem ===
    # Solves the ILP problem using the CBC solver (PULP_CBC_CMD)
    solver = pulp.PULP_CBC_CMD(...)
    prob.solve(solver)

    # ===  Extract and validate solution ===
    # Extracts the final solution:
    # - s_val: Selected strategy index for each instruction
    # - e_val: Selected resharing plan for each edge
    # - objective: Total cost of the chosen sharding plan

    return s_val, e_val, objective, status

```

### 3. Simulation

#### 3.1 Intra-op

In the intra-op simulation, we use the function `solve_auto_sharding()` introduced in Section 2 to perform the simulation.

For the experiment with the LLaMA2-13B model, we need to construct the HLO computation graph of the model. The specific parameters are set as follows:

```python
def get_LLaMa_forward_computation(batch_size, seq_len, hidden_dim, num_head, num_layers,ffn_dim, force_replicated_output):
    """
    Implements the forward pass of the LLaMA model using HLO operations.
    
    - The function consists of multiple Transformer blocks, each containing:
        1. RMSNorm before attention computation
        2. Multi-Head Self-Attention (MHSA)
        3. Residual connection
        4. RMSNorm before feedforward computation
        5. FeedForward Network (FFN)
        6. Residual connection
    - The final output is the hidden states after all layers.
    - The function supports force-replicated output if required.
    
    Args:
        batch_size (int): Number of sequences in a batch.
        seq_len (int): Sequence length.
        hidden_dim (int): Model hidden size.
        num_head (int): Number of attention heads.
        num_layers (int): Number of Transformer layers (blocks).
        force_replicated_output (bool): Whether to force replicated output.

    Returns:
        computation (HloComputation): The computation graph containing the LLaMA forward pass.
    """
```

After that, we call `solve_auto_sharding()`, where the `ClusterEnvironment` parameter allows us to specify the number of nodes in the cluster, the number of GPUs per node, the bandwidth between and within nodes, and the memory capacity of a single GPU:

```python
mesh_shape = [50,4] # 50 nodes, each node has 4 GPU
device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                     memory_per_device=32 * GB,
                                     solver_option=solver_option)
```

**filepath:** `playground\auto_sharding_solver\simulate_intra_op.py`

#### 3.2 Inter-op

In the inter-op simulation part, I implemented the dynamic programming algorithm described in the paper to perform the simulation. The specific dynamic programming recurrence is as follows:

F(s, k, d; t_max) =  
    min over k ≤ i ≤ K such that n_s * m_s ≤ d:
        if t_intra((o_k, ..., o_i), Mesh(n_s, m_s), s) ≤ t_max:
            t_intra((o_k, ..., o_i), Mesh(n_s, m_s), s) 
            + F(s - 1, i + 1, d - n_s * m_s; t_max)

Additionally, I implemented early pruning in the code to optimize the computational complexity. 

**filepath:** `playground\auto_sharding_solver\simulate_inter_op.py`



> *Note:* Original README starts from here
---
**Note: Alpa is not actively maintained currently. It is available as a research artifact. The core algorithm in Alpa has been merged into XLA, which is still being maintained. https://github.com/openxla/xla/tree/main/xla/hlo/experimental/auto_sharding**


<div align="center">
<img src="https://github.com/alpa-projects/alpa/blob/main/docs/logo/alpa-logo-cropped.png" alt="logo" width="250"></img>
<br></br>
</div>

[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)
[![Build Jaxlib](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml)

[**Documentation**](https://alpa-projects.github.io) | [**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

Alpa is a system for training and serving large-scale neural networks.

Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training and serving these large-scale neural networks require complicated distributed system techniques.
Alpa aims to automate large-scale distributed training and serving with just a few lines of code.

The key features of Alpa include:  

💻 **Automatic Parallelization**. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism. 

🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.

✨ **Tight Integration with Machine Learning Ecosystem**. Alpa is backed by open-source, high-performance, and production-ready libraries such as [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla), and [Ray](https://github.com/ray-project/ray).

## Serving
The code below shows how to use huggingface/transformers interface and Alpa distributed backend for large model inference.
Detailed documentation is in [Serving OPT-175B using Alpa](https://alpa-projects.github.io/tutorials/opt_serving.html).

```python
from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
tokenizer.add_bos_token = False

# Load the model. Alpa automatically downloads the weights to the specificed path
model = get_model(model_name="alpa/opt-2.7b", path="~/opt_weights/")

# Generate
prompt = "Paris is the capital city of"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)
```

## Training
Use Alpa's decorator ``@parallelize`` to scale your single-device training code to distributed clusters.
Check out the [documentation](https://alpa-projects.github.io) site and
[examples](https://github.com/alpa-projects/alpa/tree/main/examples) folder
for installation instructions, tutorials, examples, and more.

```python
import alpa

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = model_state.forward(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```

## Learning more
- [Papers](docs/publications/publications.rst)
- [Google AI blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
- [OSDI 2022 talk slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit?usp=sharing)
- [ICML 2022 big model tutorial](https://sites.google.com/view/icml-2022-big-model/home)
- [GTC 2023 talk video](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51337/)

## Getting Involved
- Connect to Alpa developers via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing code.

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
