# --- 1. Pure-Python Core Math (No imports) ---

def pure_exp(x):
    """Computes e^x using a Taylor series expansion."""
    if x < -700:
        return 0.0
    if x > 700:
        return float('inf')
    
    sum_val = 1.0
    term = 1.0
    n = 1
    while n < 100:
        term = term * x / n
        sum_val += term
        if abs(term) < 1e-15:
            break
        n += 1
    return sum_val

def pure_sqrt(x):
    """Computes the square root using the Babylonian method."""
    if x < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    if x == 0.0:
        return 0.0
    guess = x / 2.0
    for _ in range(100):
        better_guess = (guess + x / guess) / 2.0
        if abs(better_guess - guess) < 1e-15:
            break
        guess = better_guess
    return guess

def pure_tanh(x):
    """Computes the hyperbolic tangent using exponentials."""
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e_pos = pure_exp(x)
    e_neg = pure_exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def pure_sin(x):
    """Computes sine using a Taylor series expansion."""
    # Normalize x to [-pi, pi]
    pi = 3.141592653589793
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
    
    sum_val = 0.0
    term = x
    n = 1
    while n < 50:
        sum_val += term
        term = -term * x * x / ((2 * n) * (2 * n + 1))
        n += 1
    return sum_val

def pure_cos(x):
    """Computes cosine using a Taylor series expansion."""
    pi = 3.141592653589793
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
        
    sum_val = 0.0
    term = 1.0
    n = 1
    while n < 50:
        sum_val += term
        term = -term * x * x / ((2 * n - 1) * (2 * n))
        n += 1
    return sum_val


# --- 2. Basic Vector Operations ---

def dot_product(v1, v2):
    """Computes the dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))

def vector_add(v1, v2):
    """Adds two vectors element-wise."""
    return [a + b for a, b in zip(v1, v2)]

def vector_scalar_multiply(v, scalar):
    """Multiplies a vector by a scalar."""
    return [a * scalar for a in v]

def vector_subtract(v1, v2):
    """Subtracts v2 from v1 element-wise."""
    return [a - b for a, b in zip(v1, v2)]


# --- 3. Basic Matrix Operations ---

def matrix_transpose(m):
    """Transposes a 2D matrix."""
    return [list(row) for row in zip(*m)]

def matrix_multiply(m1, m2):
    """Multiplies two 2D matrices m1 and m2."""
    rows_m1, cols_m1 = len(m1), len(m1[0])
    rows_m2, cols_m2 = len(m2), len(m2[0])
    if cols_m1 != rows_m2:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    
    m2_t = matrix_transpose(m2)
    result = [[sum(a * b for a, b in zip(row1, row2)) for row2 in m2_t] for row1 in m1]
    return result

def matrix_addition(m1, m2):
    """Adds two matrices element-wise."""
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(m1, m2)]

def matrix_scalar_multiply(m, scalar):
    """Multiplies a matrix by a scalar."""
    return [[val * scalar for val in row] for row in m]

def create_zeros_matrix(rows, cols):
    """Creates a matrix filled with zeros."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


# --- 4. Activations & Token Embeddings ---

def softmax(vector):
    """Computes the softmax of a 1D vector using pure_exp."""
    max_val = max(vector)
    exp_vals = [pure_exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vals)
    return [val / sum_exp for val in exp_vals]

def relu(x):
    """Computes the ReLU activation function."""
    return x if x > 0.0 else 0.0

def gelu(x):
    """Computes GELU using pure math approximations."""
    pi = 3.141592653589793
    inner = pure_sqrt(2.0 / pi) * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1.0 + pure_tanh(inner))

def tokenize_simple(text):
    """Splits text by spaces without imports."""
    return text.lower().split()

def initialize_embedding_matrix(vocab_size, d_model):
    """Initializes embedding matrix using pure sine wave patterns."""
    return [[pure_sin(i * j + 1.0) for j in range(d_model)] for i in range(vocab_size)]

# --- 5. Positional Encoding ---

def generate_positional_encoding_matrix(max_len, d_model):
    """Generates sinusoidal positional encoding using custom math functions."""
    pe = [[0.0 for _ in range(d_model)] for _ in range(max_len)]
    log_10000 = 9.210340371976184  # pre-calculated natural log of 10000
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            div_term = pure_exp(-i * log_10000 / d_model)
            pe[pos][i] = pure_sin(pos * div_term)
            if i + 1 < d_model:
                pe[pos][i + 1] = pure_cos(pos * div_term)
    return pe

def add_positional_encoding(embeddings, positional_encoding):
    """Adds positional encoding to token embeddings element-wise."""
    seq_len = len(embeddings)
    d_model = len(embeddings[0])
    result = []
    for i in range(seq_len):
        row = [embeddings[i][j] + positional_encoding[i][j] for j in range(d_model)]
        result.append(row)
    return result


# --- 6. Matrix Slicing & Concatenation ---

def matrix_row_slice(m, start_row, end_row):
    """Slices rows from a matrix."""
    return [list(row) for row in m[start_row:end_row]]

def matrix_col_slice(m, start_col, end_col):
    """Slices specific columns from a matrix."""
    return [[row[col] for col in range(start_col, end_col)] for row in m]

def concat_matrices_horizontal(m1, m2):
    """Concatenates two matrices horizontally (column-wise)."""
    if len(m1) != len(m2):
        raise ValueError("Matrices must have the same number of rows to concatenate.")
    return [row1 + row2 for row1, row2 in zip(m1, m2)]

def matrix_row_sum(m):
    """Computes the sum of each row in a matrix."""
    return [sum(row) for row in m]

def matrix_mean(m):
    """Computes the mean of all elements in a matrix."""
    total_elements = sum(len(row) for row in m)
    total_sum = sum(sum(row) for row in m)
    return total_sum / total_elements if total_elements > 0 else 0.0


# --- 7. Weights, Biases & Linear Layers ---

def initialize_weight_matrix(fan_in, fan_out):
    """Initializes weight matrix using custom sine wave bounds."""
    limit = pure_sqrt(6.0 / (fan_in + fan_out))
    return [[pure_sin(i * j + 0.5) * limit for j in range(fan_out)] for i in range(fan_in)]

def initialize_bias_vector(fan_out):
    """Initializes a bias vector with zeros."""
    return [0.0 for _ in range(fan_out)]

def linear_layer_forward(input_matrix, weights, bias):
    """Performs standard linear transformation: output = input @ weights + bias."""
    multiplied = matrix_multiply(input_matrix, weights)
    return [[val + b for val, b in zip(row, bias)] for row in multiplied]

def scale_matrix_rows(m, scales):
    """Multiplies each row of a matrix by a corresponding scale value."""
    return [[val * scale for val in row] for row, scale in zip(m, scales)]


# --- 8. Attention Head Utilities ---

def split_into_heads(matrix, num_heads):
    """Splits the embedding dimension into multiple attention heads."""
    seq_len = len(matrix)
    d_model = len(matrix[0])
    d_k = d_model // num_heads
    
    heads = []
    for h in range(num_heads):
        head_matrix = []
        for i in range(seq_len):
            start_idx = h * d_k
            end_idx = (h + 1) * d_k
            head_matrix.append(matrix[i][start_idx:end_idx])
        heads.append(head_matrix)
    return heads

def combine_heads(heads):
    """Combines multiple attention heads back into a single matrix."""
    num_heads = len(heads)
    seq_len = len(heads[0])
    d_k = len(heads[0][0])
    
    combined = []
    for i in range(seq_len):
        row = []
        for h in range(num_heads):
            row.extend(heads[h][i])
        combined.append(row)
    return combined

def compute_qkv_projections(x, w_q, w_k, w_v):
    """Computes Query, Key, and Value matrices from input x."""
    Q = matrix_multiply(x, w_q)
    K = matrix_multiply(x, w_k)
    V = matrix_multiply(x, w_v)
    return Q, K, V


# --- 9. Attention Scores & Masking ---

def scaled_dot_product_attention_scores(Q, K):
    """Computes raw attention scores: Q @ K^T / sqrt(d_k)."""
    d_k = len(Q[0])
    K_t = matrix_transpose(K)
    scores = matrix_multiply(Q, K_t)
    scale = 1.0 / pure_sqrt(d_k)
    return [[val * scale for val in row] for row in scores]

def apply_attention_mask(scores, mask):
    """Applies a causal or padding mask to attention scores."""
    masked_scores = []
    for i, row in enumerate(scores):
        masked_row = []
        for j, val in enumerate(row):
            if mask[i][j] == 0:
                masked_row.append(val - 1e9)
            else:
                masked_row.append(val)
        masked_scores.append(masked_row)
    return masked_scores

def row_softmax(matrix):
    """Applies softmax to each row of a matrix independently."""
    return [softmax(row) for row in matrix]

def apply_attention_weights(attention_probs, V):
    """Multiplies attention probabilities by the Value matrix."""
    return matrix_multiply(attention_probs, V)

# --- 10. Complete Multi-Head Attention Block ---

def multi_head_attention_forward(x, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Executes the complete Multi-Head Attention forward block."""
    seq_len = len(x)
    
    # 1. Project to Q, K, V
    Q, K, V = compute_qkv_projections(x, w_q, w_k, w_v)
    
    # 2. Split into multiple heads
    Q_heads = split_into_heads(Q, num_heads)
    K_heads = split_into_heads(K, num_heads)
    V_heads = split_into_heads(V, num_heads)
    
    # 3. Apply attention for each head independently
    head_outputs = []
    for h in range(num_heads):
        out_h, _ = single_head_attention(Q_heads[h], K_heads[h], V_heads[h], mask)
        head_outputs.append(out_h)
        
    # 4. Combine heads back together
    combined = combine_heads(head_outputs)
    
    # 5. Final output linear projection
    return matrix_multiply(combined, w_o)


# --- 11. Layer Normalization Subcomponents ---

def compute_row_mean(row):
    """Computes the arithmetic mean of a single vector."""
    return sum(row) / len(row) if len(row) > 0 else 0.0

def compute_row_variance(row, mean_val):
    """Computes the variance of a single vector given its mean."""
    return sum((val - mean_val) ** 2 for val in row) / len(row) if len(row) > 0 else 0.0

def layer_norm_single_row(row, gamma, beta, eps=1e-5):
    """Applies layer normalization to a single matrix row."""
    mean_val = compute_row_mean(row)
    var_val = compute_row_variance(row, mean_val)
    inv_std = 1.0 / pure_sqrt(var_val + eps)
    
    normalized = []
    for val, g, b in zip(row, gamma, beta):
        norm_val = (val - mean_val) * inv_std
        normalized.append(norm_val * g + b)
    return normalized

def layer_norm_forward(matrix, gamma, beta, eps=1e-5):
    """Applies layer normalization across an entire 2D matrix."""
    return [layer_norm_single_row(row, gamma, beta, eps) for row in matrix]

def initialize_gamma_vector(d_model):
    """Initializes scale parameter gamma for layer norm to ones."""
    return [1.0 for _ in range(d_model)]


# --- 12. Residual Connections & Utilities ---

def initialize_beta_vector(d_model):
    """Initializes shift parameter beta for layer norm to zeros."""
    return [0.0 for _ in range(d_model)]

def residual_add(x, sublayer_output):
    """Performs element-wise residual addition: x + sublayer(x)."""
    result = []
    for row_x, row_sub in zip(x, sublayer_output):
        new_row = [a + b for a, b in zip(row_x, row_sub)]
        result.append(new_row)
    return result

def matrix_elementwise_product(m1, m2):
    """Multiplies two matrices element-wise."""
    return [[a * b for a, b in zip(row1, row2)] for row1, row2 in zip(m1, m2)]

def create_ones_matrix(rows, cols):
    """Creates a matrix filled with ones."""
    return [[1.0 for _ in range(cols)] for _ in range(rows)]

def matrix_relu_forward(matrix):
    """Applies ReLU element-wise to an entire matrix."""
    return [[relu(val) for val in row] for row in matrix]


# --- 13. Feed-Forward Network Blocks ---

def feed_forward_hidden_projection(x, w1, b1):
    """Computes the first linear expansion layer of the FFN."""
    linear_out = linear_layer_forward(x, w1, b1)
    return matrix_relu_forward(linear_out)

def feed_forward_output_projection(hidden_matrix, w2, b2):
    """Computes the second linear projection layer of the FFN."""
    return linear_layer_forward(hidden_matrix, w2, b2)

def feed_forward_network_forward(x, w1, b1, w2, b2):
    """Executes the complete Position-wise Feed-Forward Network."""
    hidden = feed_forward_hidden_projection(x, w1, b1)
    return feed_forward_output_projection(hidden, w2, b2)

# --- 14. Encoder Sublayer Wrappers ---

def encoder_self_attention_sublayer(x, w_q, w_k, w_v, w_o, num_heads, gamma, beta, mask=None):
    """Executes MHA followed by residual connection and layer normalization."""
    mha_out = multi_head_attention_forward(x, w_q, w_k, w_v, w_o, num_heads, mask)
    res_out = residual_add(x, mha_out)
    return layer_norm_forward(res_out, gamma, beta)

def encoder_ffn_sublayer(x, w1, b1, w2, b2, gamma, beta):
    """Executes Position-wise FFN followed by residual connection and layer normalization."""
    ffn_out = feed_forward_network_forward(x, w1, b1, w2, b2)
    res_out = residual_add(x, ffn_out)
    return layer_norm_forward(res_out, gamma, beta)


# --- 15. Single Transformer Encoder Layer ---

def create_encoder_layer_weights(d_model, d_ff):
    """Initializes all individual weight matrices and biases for a single encoder layer."""
    weights = {
        "w_q": initialize_weight_matrix(d_model, d_model),
        "w_k": initialize_weight_matrix(d_model, d_model),
        "w_v": initialize_weight_matrix(d_model, d_model),
        "w_o": initialize_weight_matrix(d_model, d_model),
        "gamma1": initialize_gamma_vector(d_model),
        "beta1": initialize_beta_vector(d_model),
        "w1": initialize_weight_matrix(d_model, d_ff),
        "b1": initialize_bias_vector(d_ff),
        "w2": initialize_weight_matrix(d_ff, d_model),
        "b2": initialize_bias_vector(d_model),
        "gamma2": initialize_gamma_vector(d_model),
        "beta2": initialize_beta_vector(d_model)
    }
    return weights

def encoder_layer_forward(x, layer_weights, num_heads, mask=None):
    """Performs the forward pass through a single Transformer Encoder layer."""
    # Sublayer 1: Multi-Head Self-Attention + Add & Norm
    x1 = encoder_self_attention_sublayer(
        x, 
        layer_weights["w_q"], layer_weights["w_k"], layer_weights["w_v"], layer_weights["w_o"], 
        num_heads, 
        layer_weights["gamma1"], layer_weights["beta1"], 
        mask
    )
    
    # Sublayer 2: Feed-Forward Network + Add & Norm
    x2 = encoder_ffn_sublayer(
        x1, 
        layer_weights["w1"], layer_weights["b1"], 
        layer_weights["w2"], layer_weights["b2"], 
        layer_weights["gamma2"], layer_weights["beta2"]
    )
    
    return x2


# --- 16. Multi-Layer Encoder Stack ---

def create_encoder_stack_weights(num_layers, d_model, d_ff):
    """Initializes weights for a stack of multiple encoder layers."""
    return [create_encoder_layer_weights(d_model, d_ff) for _ in range(num_layers)]

def encoder_stack_forward(x, encoder_stack_weights_list, num_heads, mask=None):
    """Passes input sequentially through a stack of N encoder layers."""
    current_representation = x
    for layer_weights in encoder_stack_weights_list:
        current_representation = encoder_layer_forward(
            current_representation, 
            layer_weights, 
            num_heads, 
            mask
        )
    return current_representation


# --- 17. Encoder Utility & Diagnostics ---

def extract_encoder_layer_representations(x, encoder_stack_weights_list, num_heads, mask=None):
    """Extracts hidden outputs from every individual layer for inspection."""
    representations = [x]
    current = x
    for layer_weights in encoder_stack_weights_list:
        current = encoder_layer_forward(current, layer_weights, num_heads, mask)
        representations.append(current)
    return representations

def compute_encoder_parameter_count(d_model, d_ff, num_layers):
    """Computes total raw parameter count for an encoder architecture without imports."""
    # Per layer: Q, K, V, O matrices (4 * d_model^2) + FFN matrices (d_model * d_ff + d_ff * d_model) + layer norms (4 * d_model)
    params_per_layer = (4 * (d_model ** 2)) + (2 * d_model * d_ff) + (4 * d_model)
    return params_per_layer * num_layers

def validate_matrix_shape(matrix, expected_rows, expected_cols):
    """Validates row and column counts for internal tensor tracking."""
    if len(matrix) != expected_rows:
        return False
    if any(len(row) != expected_cols for row in matrix):
        return False
    return True

def create_blank_tensor_accumulator(seq_len, d_model):
    """Allocates a container structure for intermediate activations."""
    return [[0.0 for _ in range(d_model)] for _ in range(seq_len)]

# --- 18. Decoder Cross-Attention Subcomponents ---

def compute_decoder_kv_from_encoder(encoder_output, w_k_enc, w_v_enc):
    """Computes Key and Value matrices from encoder output for cross-attention."""
    K_enc = matrix_multiply(encoder_output, w_k_enc)
    V_enc = matrix_multiply(encoder_output, w_v_enc)
    return K_enc, V_enc

def cross_attention_forward(Q_dec, K_enc, V_enc, mask=None):
    """Executes cross-attention where Queries come from decoder and Keys/Values from encoder."""
    scores = scaled_dot_product_attention_scores(Q_dec, K_enc)
    if mask is not None:
        scores = apply_attention_mask(scores, mask)
    probs = row_softmax(scores)
    return apply_attention_weights(probs, V_enc), probs


# --- 19. Single Decoder Layer Architecture ---

def create_decoder_layer_weights(d_model, d_ff):
    """Initializes all individual weight matrices and biases for a single decoder layer."""
    return {
        # Masked Self-Attention
        "w_q_self": initialize_weight_matrix(d_model, d_model),
        "w_k_self": initialize_weight_matrix(d_model, d_model),
        "w_v_self": initialize_weight_matrix(d_model, d_model),
        "w_o_self": initialize_weight_matrix(d_model, d_model),
        "gamma1": initialize_gamma_vector(d_model),
        "beta1": initialize_beta_vector(d_model),
        
        # Cross-Attention (Decoder to Encoder)
        "w_q_cross": initialize_weight_matrix(d_model, d_model),
        "w_k_cross": initialize_weight_matrix(d_model, d_model),
        "w_v_cross": initialize_weight_matrix(d_model, d_model),
        "w_o_cross": initialize_weight_matrix(d_model, d_model),
        "gamma2": initialize_gamma_vector(d_model),
        "beta2": initialize_beta_vector(d_model),
        
        # Position-wise Feed-Forward
        "w1": initialize_weight_matrix(d_model, d_ff),
        "b1": initialize_bias_vector(d_ff),
        "w2": initialize_weight_matrix(d_ff, d_model),
        "b2": initialize_bias_vector(d_model),
        "gamma3": initialize_gamma_vector(d_model),
        "beta3": initialize_beta_vector(d_model)
    }

def decoder_layer_forward(x, encoder_output, layer_weights, num_heads, self_mask=None, cross_mask=None):
    """Performs the forward pass through a single Transformer Decoder layer."""
    # 1. Masked Self-Attention Sublayer + Add & Norm
    q_s, k_s, v_s = compute_qkv_projections(x, layer_weights["w_q_self"], layer_weights["w_k_self"], layer_weights["w_v_self"])
    q_heads_s = split_into_heads(q_s, num_heads)
    k_heads_s = split_into_heads(k_s, num_heads)
    v_heads_s = split_into_heads(v_s, num_heads)
    
    self_outputs = []
    for h in range(num_heads):
        out_h, _ = single_head_attention(q_heads_s[h], k_heads_s[h], v_heads_s[h], self_mask)
        self_outputs.append(out_h)
    combined_self = combine_heads(self_outputs)
    proj_self = matrix_multiply(combined_self, layer_weights["w_o_self"])
    res_self = residual_add(x, proj_self)
    norm_self = layer_norm_forward(res_self, layer_weights["gamma1"], layer_weights["beta1"])
    
    # 2. Encoder-Decoder Cross-Attention Sublayer + Add & Norm
    q_c = matrix_multiply(norm_self, layer_weights["w_q_cross"])
    k_enc, v_enc = compute_decoder_kv_from_encoder(encoder_output, layer_weights["w_k_cross"], layer_weights["w_v_cross"])
    q_heads_c = split_into_heads(q_c, num_heads)
    k_heads_c = split_into_heads(k_enc, num_heads)
    v_heads_c = split_into_heads(v_enc, num_heads)
    
    cross_outputs = []
    for h in range(num_heads):
        # Cross-attention uses single head structure over cross parameters
        scores_c = scaled_dot_product_attention_scores(q_heads_c[h], k_heads_c[h])
        if cross_mask is not None:
            scores_c = apply_attention_mask(scores_c, cross_mask)
        probs_c = row_softmax(scores_c)
        cross_outputs.append(apply_attention_weights(probs_c, v_heads_c[h]))
    combined_cross = combine_heads(cross_outputs)
    proj_cross = matrix_multiply(combined_cross, layer_weights["w_o_cross"])
    res_cross = residual_add(norm_self, proj_cross)
    norm_cross = layer_norm_forward(res_cross, layer_weights["gamma2"], layer_weights["beta2"])
    
    # 3. Feed-Forward Sublayer + Add & Norm
    ffn_out = feed_forward_network_forward(norm_cross, layer_weights["w1"], layer_weights["b1"], layer_weights["w2"], layer_weights["b2"])
    res_ffn = residual_add(norm_cross, ffn_out)
    return layer_norm_forward(res_ffn, layer_weights["gamma3"], layer_weights["beta3"])


# --- 20. Language Model Head & End-to-End Pipeline ---

def create_decoder_stack_weights(num_layers, d_model, d_ff):
    """Initializes weights for a stack of multiple decoder layers."""
    return [create_decoder_layer_weights(d_model, d_ff) for _ in range(num_layers)]

def decoder_stack_forward(x, encoder_output, decoder_stack_weights_list, num_heads, self_mask=None, cross_mask=None):
    """Passes input sequentially through a stack of N decoder layers."""
    current = x
    for layer_weights in decoder_stack_weights_list:
        current = decoder_layer_forward(current, encoder_output, layer_weights, num_heads, self_mask, cross_mask)
    return current

def language_model_projection_head(decoder_output, w_vocab):
    """Projects final hidden states down to vocabulary logits: output @ w_vocab."""
    return matrix_multiply(decoder_output, w_vocab)

def extract_predicted_token_indices(logits):
    """Extracts the maximum logit index (argmax) for token generation from the last sequence step."""
    last_step_logits = logits[-1]
    max_val = last_step_logits[0]
    best_idx = 0
    for idx, val in enumerate(last_step_logits):
        if val > max_val:
            max_val = val
            best_idx = idx
    return best_idx

def full_transformer_sequence_to_sequence_forward(
    input_indices, target_indices, vocab_size, d_model, d_ff, num_heads, num_layers, pos_matrix
):
    """Performs an end-to-end forward run of the complete Transformer model from scratch."""
    # 1. Initialize lookup embeddings
    embed_matrix = initialize_embedding_matrix(vocab_size, d_model)
    
    # 2. Encoder Path
    enc_raw_embeddings = lookup_embeddings(input_indices, embed_matrix)
    enc_seq_len = len(input_indices)
    enc_pe_slice = matrix_row_slice(pos_matrix, 0, enc_seq_len)
    enc_input = add_positional_encoding(enc_raw_embeddings, enc_pe_slice)
    
    enc_weights = create_encoder_stack_weights(num_layers, d_model, d_ff)
    encoder_output = encoder_stack_forward(enc_input, enc_weights, num_heads, mask=None)
    
    # 3. Decoder Path
    dec_raw_embeddings = lookup_embeddings(target_indices, embed_matrix)
    dec_seq_len = len(target_indices)
    dec_pe_slice = matrix_row_slice(pos_matrix, 0, dec_seq_len)
    dec_input = add_positional_encoding(dec_raw_embeddings, dec_pe_slice)
    
    causal_mask = create_causal_mask(dec_seq_len)
    dec_weights = create_decoder_stack_weights(num_layers, d_model, d_ff)
    decoder_output = decoder_stack_forward(dec_input, encoder_output, dec_weights, num_heads, self_mask=causal_mask, cross_mask=None)
    
    # 4. Language Model Head Projection
    vocab_weights = initialize_weight_matrix(d_model, vocab_size)
    logits = language_model_projection_head(decoder_output, vocab_weights)
    
    return logits

# --- 21. Sequence Generation & Autoregressive Helpers ---

def argmax_1d(vector):
    """Finds the index of the maximum value in a 1D list."""
    if not vector:
        return -1
    max_val = vector[0]
    best_idx = 0
    for i, val in enumerate(vector):
        if val > max_val:
            max_val = val
            best_idx = i
    return best_idx

def greedy_decode_step(decoder_output, w_vocab):
    """Projects decoder output and returns the single best next token index."""
    logits = language_model_projection_head(decoder_output, w_vocab)
    last_row = logits[-1]
    return argmax_1d(last_row)

def compute_sequence_perplexity(loss_values):
    """Computes perplexity from a list of cross-entropy loss steps using pure_exp."""
    avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
    return pure_exp(avg_loss)

def top_k_filtering(logits_vector, k=5):
    """Keeps only the top-k highest logits and sets the rest to negative infinity equivalents."""
    indexed_logits = list(enumerate(logits_vector))
    indexed_logits.sort(key=lambda x: x[1], reverse=True)
    
    filtered = [-1e9 for _ in range(len(logits_vector))]
    for rank, (idx, val) in enumerate(indexed_logits):
        if rank < k:
            filtered[idx] = val
    return filtered

def sample_from_distribution(probabilities):
    """Samples an index from a probability distribution list using a pseudo-random seed step."""
    # Deterministic pseudo-random choice based on list values and lengths
    threshold = (sum(probabilities) * 0.37) % 1.0
    cumulative = 0.0
    for idx, prob in enumerate(probabilities):
        cumulative += prob
        if cumulative >= threshold:
            return idx
    return len(probabilities) - 1


# --- 22. Beam Search Components ---

def calculate_beam_score(current_score, token_prob, length_penalty=0.6):
    """Calculates penalized score for a beam hypothesis."""
    # Length normalization penalty formulation
    ep = pure_exp(token_prob) if token_prob > -50 else 1e-9
    penalty = ((5.0 + len(str(current_score))) ** length_penalty) / ((5.0 + 1.0) ** length_penalty)
    return (current_score + token_prob) / penalty

def initialize_beams(start_token_idx):
    """Initializes beam search state containers."""
    # Format: [sequence_indices, cumulative_score]
    return [([[start_token_idx]], 0.0)]

def rank_and_trim_beams(beams, beam_width):
    """Ranks all active beams by score and trims to beam_width."""
    beams.sort(key=lambda x: x[1], reverse=True)
    return beams[:beam_width]

def check_beam_termination(beams, end_token_idx, max_len):
    """Checks if top beams have reached termination criteria."""
    for seq, score in beams:
        if seq[-1] == end_token_idx or len(seq) >= max_len:
            return True
    return False


# --- 23. Advanced Vocabulary & Token Mapping ---

def build_inverse_vocabulary(vocab_dict):
    """Inverts a token-to-index dictionary into an index-to-token dictionary."""
    return {idx: token for token, idx in vocab_dict.items()}

def tokens_to_string(indices, inverse_vocab):
    """Converts a list of integer indices back to a readable space-separated string."""
    tokens = [inverse_vocab.get(idx, "<unk>") for idx in indices]
    return " ".join(tokens)

def pad_sequence(sequence, max_length, padding_value=0):
    """Pads a sequence of token indices to a fixed maximum length."""
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [padding_value] * (max_length - len(sequence))

def create_padding_mask(sequence, padding_value=0):
    """Creates a 2D attention mask for padding tokens (1 for valid, 0 for padded)."""
    seq_len = len(sequence)
    mask = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if sequence[j] != padding_value:
                mask[i][j] = 1.0
            else:
                mask[i][j] = 0.0
    return mask


# --- 24. Attention Diagnostics & Layer Audits ---

def compute_attention_entropy(attention_probs_row):
    """Computes the Shannon entropy of an attention probability distribution row."""
    entropy = 0.0
    for p in attention_probs_row:
        if p > 0.0:
            # Using custom log approximation via Taylor expansion or simple arithmetic bounds
            # log(p) approx: log(p) = 2 * sum((p-1)/(p+1)^term)
            log_p = (p - 1.0) - ((p - 1.0) ** 2) / 2.0 + ((p - 1.0) ** 3) / 3.0
            entropy -= p * log_p
    return entropy

def audit_layer_norm_outputs(matrix):
    """Audits a matrix to verify that row means are near 0 and variances near 1."""
    report = []
    for row in matrix:
        mean_val = compute_row_mean(row)
        var_val = compute_row_variance(row, mean_val)
        report.append((mean_val, var_val))
    return report

def count_total_model_parameters(d_model, d_ff, vocab_size, num_layers):
    """Computes grand total parameters for encoder-decoder architecture."""
    embedding_params = vocab_size * d_model
    encoder_params = (4 * (d_model ** 2) + 2 * d_model * d_ff + 4 * d_model) * num_layers
    decoder_params = (12 * (d_model ** 2) + 2 * d_model * d_ff + 8 * d_model) * num_layers
    head_params = d_model * vocab_size
    return embedding_params + encoder_params + decoder_params + head_params

def verify_matrix_dimensions_match(m1, m2):
    """Verifies if two matrices have identical height and width dimensions."""
    if len(m1) != len(m2):
        return False
    return all(len(r1) == len(r2) for r1, r2 in zip(m1, m2))

def print_tensor_shape_summary(matrix_name, matrix):
    """Generates a text summary tuple of tensor dimensions."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return f"{matrix_name}: Shape ({rows}, {cols})"
  
  
