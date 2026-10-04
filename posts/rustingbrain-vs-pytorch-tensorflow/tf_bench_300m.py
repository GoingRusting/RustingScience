"""Speed-only benchmark: same architecture/shapes as the RustingLLM run
(d_model=768, n_layers=16, GQA 12/4 heads, SwiGLU FFN), fed random token
ids. No real training, no dataset needed. Sweeps batch size and seq_len,
runs each shape N_TRIALS times so you get mean +/- stddev, not one lucky
number. MoE is off by default (USE_MOE=False) -- TF has no native
sparse-routing kernel, so an MoE run here would be "TF computes all
experts densely" vs Rust's real sparse top-k, not a fair kernel test.

Run (after the real training run on the GPU is done or paused):
    source .venv-tf/bin/activate.fish
    set -x LD_LIBRARY_PATH (find .venv-tf/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | string join ':')
    python scripts/tf_bench_300m.py
"""
import statistics
import time
import tensorflow as tf

# TF pre-caps its allocator well under the card's real memory by default;
# without this it OOMs on shapes that fit fine (seen: batch=4/seq=1024 on
# a free 12GB card capped at 9.3GiB). Let it grow to what's actually free.
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

VOCAB = 32004
D_MODEL = 768
N_LAYERS = 16
N_HEADS = 12
N_KV_HEADS = 4
HEAD_DIM = 64
D_FF = 1408
MOE_D_FF = 640
NUM_EXPERTS = 8
EXPERTS_PER_TOKEN = 2
MOE_FROM_LAYER = 4  # matches train.rs: n_layers/4..n_layers
USE_MOE = False  # dense-only isolates kernel speed from the routing-compute
# confound (see docstring); flip to True only knowing the comparison then
# includes "TF computes all experts" vs "Rust computes top-k only".

import os
import sys

SHAPE = (int(os.environ.get("BENCH_BATCH", 1)), int(os.environ.get("BENCH_SEQ", 512)))
N_TRIALS = 5
STEPS_PER_TRIAL = 10
WARMUP = 5


def rope(x, seq_len, head_dim):
    inv_freq = 1.0 / (10000.0 ** (tf.range(0, head_dim, 2, dtype=tf.float32) / head_dim))
    pos = tf.cast(tf.range(seq_len), tf.float32)
    freqs = tf.einsum("s,d->sd", pos, inv_freq)
    emb = tf.concat([freqs, freqs], axis=-1)
    cos = tf.cast(tf.cos(emb), x.dtype)[None, None, :, :]
    sin = tf.cast(tf.sin(emb), x.dtype)[None, None, :, :]
    x1, x2 = tf.split(x, 2, axis=-1)
    rotated = tf.concat([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = self.add_weight(shape=(dim,), initializer="ones", trainable=True)

    def call(self, x):
        x32 = tf.cast(x, tf.float32)
        variance = tf.reduce_mean(tf.square(x32), axis=-1, keepdims=True)
        normed = x32 * tf.math.rsqrt(variance + self.eps)
        return tf.cast(normed, x.dtype) * tf.cast(self.gain, x.dtype)


class GQAAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.group = n_heads // n_kv_heads
        self.wq = tf.keras.layers.Dense(n_heads * head_dim, use_bias=False)
        self.wk = tf.keras.layers.Dense(n_kv_heads * head_dim, use_bias=False)
        self.wv = tf.keras.layers.Dense(n_kv_heads * head_dim, use_bias=False)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x, mask):
        b, s = tf.shape(x)[0], tf.shape(x)[1]
        q = tf.transpose(tf.reshape(self.wq(x), (b, s, self.n_heads, self.head_dim)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(self.wk(x), (b, s, self.n_kv_heads, self.head_dim)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(self.wv(x), (b, s, self.n_kv_heads, self.head_dim)), (0, 2, 1, 3))
        q = rope(q, self.seq_len, self.head_dim)
        k = rope(k, self.seq_len, self.head_dim)
        k = tf.repeat(k, self.group, axis=1)
        v = tf.repeat(v, self.group, axis=1)
        scores = tf.matmul(q, k, transpose_b=True) / tf.cast(self.head_dim ** 0.5, q.dtype)
        scores = tf.cast(scores, tf.float32) + tf.cast(mask, tf.float32)
        attn = tf.cast(tf.nn.softmax(scores, axis=-1), v.dtype)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (b, s, self.n_heads * self.head_dim))
        return self.wo(out)


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.up = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.down = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        return self.down(tf.nn.silu(self.gate(x)) * self.up(x))


class MoE(tf.keras.layers.Layer):
    """Dense-compute top-k MoE: every expert runs on every token. Not
    FLOP-sparse like the Rust gather/scatter kernel. See module docstring."""

    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.router = tf.keras.layers.Dense(num_experts, use_bias=False)
        self.experts = [SwiGLU(d_model, d_ff) for _ in range(num_experts)]
        self.shared = SwiGLU(d_model, d_ff)

    def call(self, x):
        logits = tf.cast(self.router(x), tf.float32)
        top_val, top_idx = tf.math.top_k(logits, k=self.top_k)
        weights = tf.nn.softmax(top_val, axis=-1)
        out = tf.zeros_like(x, dtype=tf.float32)
        for expert_id, expert in enumerate(self.experts):
            expert_out = tf.cast(expert(x), tf.float32)
            match = tf.reduce_sum(
                tf.where(tf.equal(top_idx, expert_id), weights, 0.0), axis=-1, keepdims=True
            )
            out += expert_out * match
        return tf.cast(out, x.dtype) + self.shared(x)


class Block(tf.keras.layers.Layer):
    def __init__(self, layer_idx, seq_len):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = GQAAttention(D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, seq_len)
        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = (
            MoE(D_MODEL, MOE_D_FF, NUM_EXPERTS, EXPERTS_PER_TOKEN)
            if USE_MOE and layer_idx >= MOE_FROM_LAYER
            else SwiGLU(D_MODEL, D_FF)
        )

    def call(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Model(tf.keras.Model):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed = tf.keras.layers.Embedding(VOCAB, D_MODEL)
        self.blocks = [Block(i, seq_len) for i in range(N_LAYERS)]
        self.norm = RMSNorm(D_MODEL)

    def call(self, ids):
        mask = tf.linalg.band_part(tf.ones((self.seq_len, self.seq_len)), -1, 0)
        mask = (1.0 - mask) * -1e9
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return tf.matmul(x, self.embed.embeddings, transpose_b=True)


def bench_shape(batch, seq_len):
    model = Model(seq_len)
    optimizer = tf.keras.optimizers.Adam(3e-4)
    ids = tf.random.uniform((batch, seq_len + 1), 0, VOCAB, dtype=tf.int32)
    inputs, targets = ids[:, :-1], ids[:, 1:]

    @tf.function(jit_compile=True)
    def step():
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=tf.cast(logits, tf.float32)
                )
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    losses = []
    for _ in range(WARMUP):
        losses.append(float(step()))

    trial_times = []
    for _ in range(N_TRIALS):
        start = time.time()
        for _ in range(STEPS_PER_TRIAL):
            losses.append(float(step()))
        trial_times.append((time.time() - start) / STEPS_PER_TRIAL)

    params = sum(int(tf.size(v)) for v in model.trainable_variables)
    tokens_per_step = batch * seq_len
    tok_per_sec = [tokens_per_step / t for t in trial_times]
    return {
        "params_m": params / 1e6,
        "ms_per_step_mean": statistics.mean(trial_times) * 1000,
        "ms_per_step_stdev": statistics.stdev(trial_times) * 1000 if len(trial_times) > 1 else 0.0,
        "tok_per_sec_mean": statistics.mean(tok_per_sec),
        "loss_finite": all(l == l and abs(l) < 1e6 for l in losses),  # NaN/Inf check
        "loss_last": losses[-1],
    }


def main():
    print(f"{'batch':>6} {'seq':>6} {'params':>9} {'ms/step':>18} {'tok/sec':>10} {'loss_ok':>8} {'final_loss':>10}")
    batch, seq_len = SHAPE
    r = bench_shape(batch, seq_len)
    print(
        f"{batch:6d} {seq_len:6d} {r['params_m']:8.1f}M "
        f"{r['ms_per_step_mean']:8.1f}+-{r['ms_per_step_stdev']:<5.1f} "
        f"{r['tok_per_sec_mean']:10.0f} {str(r['loss_finite']):>8} {r['loss_last']:10.3f}"
    )


if __name__ == "__main__":
    main()
