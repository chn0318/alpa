import numpy as np
from hlo import *
from cluster_env import ClusterEnvironment
from solver import solve_auto_sharding, SolverOption


MB = 1024 ** 2
GB = 1024 ** 3
def get_attention_forward_computation(input, batch_size, seq_len, hidden_dim, num_head, force_replicated_output):
        per_head = hidden_dim // num_head
        # hidden states
        hidden_states = HloReshape(input, (batch_size * seq_len, hidden_dim))

        # query matmul
        weight_query_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_query_dense_ = HloReshape(weight_query_dense, (hidden_dim, hidden_dim))
        query = HloDot(hidden_states, weight_query_dense_)
        query = HloReshape(query, (batch_size, seq_len, num_head, per_head))

        # query bias_add
        bias_query_dense = HloParameter((num_head, per_head))
        bias_query_dense_ = HloBroadcast(bias_query_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        query = HloAdd(query, bias_query_dense_)

        # query normalization
        c = HloConstant(0.125)
        c = HloBroadcast(c, (batch_size, seq_len, num_head, per_head))
        query = HloMutiply(c, query)
        # query transpose
        query = HloTranspose(query, [0, 2, 1, 3])

        # key matmul
        weight_key_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_key_dense_ = HloReshape(weight_key_dense, (hidden_dim, hidden_dim))
        key = HloDot(hidden_states, weight_key_dense_)
        key = HloReshape(key, (batch_size, seq_len, num_head, per_head))

        # key bias_add
        bias_key_dense = HloParameter((num_head, per_head))
        bias_key_dense_ = HloBroadcast(bias_key_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        key = HloAdd(key, bias_key_dense_)

        # key transpose
        key = HloTranspose(key, [0, 2, 3, 1])

        # att_weight
        att_weight = HloDot(query, key,
                            lhs_batch_dims=(0,1), lhs_contracting_dims=(3,),
                            rhs_batch_dims=(0,1), rhs_contracting_dims=(2,))

        # mask
        mask = HloParameter((batch_size, seq_len))

        # attention_bias_pred
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))
        pred = HloCompare(mask, zero)

        # all zero
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))

        # all neg-infinity
        neg_inf = HloConstant(-1e10)
        neg_inf = HloBroadcast(neg_inf, (batch_size, seq_len))

        # attention bias
        select = HloSelect(pred, zero, neg_inf)

        # attention bias_add
        att_bias = HloBroadcast(select, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 3))
        att_weight = HloAdd(att_weight, att_bias)

        # softmax_max
        max_reduce = HloReduce(att_weight, dimensions=(3,))
        max_reduce = HloBroadcast(max_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        diff = HloSubtract(att_weight, max_reduce)
        exp = HloExp(diff)
        # softmax_sum
        sum_reduce = HloReduce(exp, dimensions=(3,))
        sum_reduce = HloBroadcast(sum_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        # softmax_norm
        softmax = HloDiv(exp, sum_reduce)

        # value matmul
        weight_value_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_value_dense_ = HloReshape(weight_value_dense, (hidden_dim, hidden_dim))
        value = HloDot(hidden_states, weight_value_dense_)
        value = HloReshape(value, (batch_size, seq_len, num_head, per_head))

        # value bias_add
        bias_value_dense = HloParameter((num_head, per_head))
        bias_value_dense_ = HloBroadcast(bias_value_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        value = HloAdd(value, bias_value_dense_)

        # value transpose
        value = HloTranspose(value, [0, 2, 3, 1])

        # self attention
        self_att = HloDot(value, softmax,
                          lhs_batch_dims=(0, 1), lhs_contracting_dims=(3,),
                          rhs_batch_dims=(0, 1), rhs_contracting_dims=(3,))
        self_att = HloTranspose(self_att, [0, 3, 1, 2])
        self_att = HloReshape(self_att, [batch_size * seq_len, hidden_dim])

        # out matmul
        weight_out_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_out_dense_ = HloReshape(weight_out_dense, (hidden_dim, hidden_dim))
        out = HloDot(self_att, weight_out_dense_)
        out = HloReshape(out, (batch_size, seq_len, hidden_dim))

        # out bias_add
        bias_out_dense = HloParameter((hidden_dim,))
        bias_out_dense_ = HloBroadcast(bias_out_dense, (batch_size, seq_len, hidden_dim), dimensions=(2,))
        out = HloAdd(out, bias_out_dense_)

        if force_replicated_output:
            out = HloForceReplicated(out)


        return out


def get_feedforward_computation(input, batch_size, seq_len, hidden_dim, ffn_dim):
    """
    Implements the FeedForward layer of a Llama:
    - Linear transformation W1
    - Simulate SwiGLU activation (using HloSelect as a replacement for HloMaximum)
    - Linear transformation W2
    """

    # Reshape hidden_states
    hidden_states = HloReshape(input, (batch_size * seq_len, hidden_dim))  # (B*S, H)

    # Linear 1: W1 * x + b1
    weight_ffn_1 = HloParameter((hidden_dim, ffn_dim))  # (H, ffn_dim)
    bias_ffn_1 = HloParameter((ffn_dim,))  # (ffn_dim,)

    ffn_hidden = HloDot(hidden_states, weight_ffn_1)  # (B*S, ffn_dim)

    # Broadcast bias to match shape (B, S, ffn_dim)
    bias_ffn_1_broadcasted = HloBroadcast(bias_ffn_1, (batch_size* seq_len, ffn_dim), dimensions=(1,))
    ffn_hidden = HloAdd(ffn_hidden, bias_ffn_1_broadcasted)  # (B, S, ffn_dim)

    # Simulate SwiGLU activation
    zero = HloConstant(0)
    zero_broadcast = HloBroadcast(zero, (batch_size*seq_len, ffn_dim))  # Ensure shape matches
    pred = HloCompare(ffn_hidden, zero_broadcast)  # Generate boolean tensor, checking if > 0
    activated = HloSelect(pred, ffn_hidden, zero_broadcast)  # Select max(0, x)

    # Linear 2: W2 * activated + b2
    weight_ffn_2 = HloParameter((ffn_dim, hidden_dim))
    output = HloDot(activated, weight_ffn_2)
    bias_ffn_2 = HloParameter((hidden_dim,))
    output = HloAdd(output, HloBroadcast(bias_ffn_2, (batch_size * seq_len, hidden_dim), dimensions=(1,)))

    # Reshape output
    output = HloReshape(output, (batch_size, seq_len, hidden_dim))

    return output



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
    computation = HloComputation()
    
    with computation:
        # Initial Hidden States
        hidden_states = HloParameter((batch_size, seq_len, hidden_dim))

        for i in range(num_layers):  # Add multiple Attention Blocks

            # Normalization (Simulating RMSNorm)
            c = HloConstant(0.125)
            c = HloBroadcast(c, (batch_size, seq_len, hidden_dim))
            RMSNorm_states = HloMutiply(c, hidden_states)

            # Single Attention Layer
            attn_output = get_attention_forward_computation(
                RMSNorm_states,batch_size, seq_len, hidden_dim, num_head, force_replicated_output
            )

            # Residual Connection
            hidden_states = HloAdd(hidden_states, attn_output)

            # Normalization before FeedForward Layer (Simulating RMSNorm)
            c = HloConstant(0.125)
            c = HloBroadcast(c, (batch_size, seq_len, hidden_dim))
            RMSNorm_states = HloMutiply(c, hidden_states)

            # FeedForward Network (FFN)
            ff_output = get_feedforward_computation(RMSNorm_states, batch_size, seq_len, hidden_dim, ffn_dim)

            # Residual Connection
            hidden_states = HloAdd(hidden_states, ff_output)

        out = hidden_states

        if force_replicated_output:
            out = HloForceReplicated(out)

        return computation

    

if __name__ == '__main__':
    """
    - The parameters are configured to match those of the LLaMA 2-13B model.
    - The computation includes multiple Transformer layers, each with:
        1. Multi-Head Self-Attention (MHSA)
        2. Feed Forward Network (FFN)
        3. RMSNorm normalization
        4. Residual connections
    
    - The model processes sequences of length 4096.
    - The batch size is set to 4 megabytes (MB).
    - The hidden dimension, attention heads, and layer count follow the original LLaMA 2-13B specification.
    """

    # Define model parameters (matching LLaMA 2-13B configuration)
    batch_size = 4 * MB  # Batch size scaled to memory availability
    seq_len = 4096       # Context length per sequence
    hidden_dim = 5120    # Dimensionality of model hidden states
    ffn_dim = 20480      # Feed-forward network expansion dimension (4x hidden_dim)
    num_head = 40        # Number of attention heads in multi-head attention
    num_layer = 40       # Number of Transformer layers (depth of the model)

    # Define model parameters (matching LLaMA 2-70B configuration)
    # batch_size = 4 * MB  # Batch size scaled to memory availability
    # seq_len = 4096       # Context length per sequence
    # hidden_dim = 8192    # Dimensionality of model hidden states
    # ffn_dim = 32768      # Feed-forward network expansion dimension (4x hidden_dim)
    # num_head = 64        # Number of attention heads in multi-head attention
    # num_layer = 80       # Number of Transformer layers (depth of the model)

    # # Initialize computation graph for LLaMA 2 forward pass
    computation = get_LLaMa_forward_computation(
        batch_size, seq_len, hidden_dim, num_head, num_layer, ffn_dim, True
    )

    mesh_shape = [50,4] # 50 nodes, each node has 4 GPU
    device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)

    solver_option = SolverOption()
    solver_option.force_all_gather_cost = 1e8
    
    cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 0.01],
                                     memory_per_device=32 * GB,
                                     solver_option=solver_option)
    
    # Caculate the best intra-op shard for LLaMa on cluster
    objective = solve_auto_sharding(computation, cluster_env, solver_option)