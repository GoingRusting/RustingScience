"""Same benchmark as tf_bench_300m.py, PyTorch instead of TF. Same
architecture (d_model=768, n_layers=16, GQA 12/4 heads, SwiGLU FFN),
same shapes, same trial/warmup counts, dense-only (MoE off, see
tf_bench_300m.py docstring for why). torch.compile is PyTorch's
XLA-equivalent fast path -- used here so this is "PyTorch at its best",
same standard as jit_compile=True on the TF side.

Run:
    .venv-torch/bin/python scripts/torch_bench_300m.py
"""
import os
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 32004
D_MODEL = 768
N_LAYERS = 16
N_HEADS = 12
N_KV_HEADS = 4
HEAD_DIM = 64
D_FF = 1408
GROUP = N_HEADS // N_KV_HEADS

SHAPE = (int(os.environ.get("BENCH_BATCH", 1)), int(os.environ.get("BENCH_SEQ", 512)))
N_TRIALS = 5
STEPS_PER_TRIAL = 10
WARMUP = 5

device = "cuda"
dtype = torch.float16


def rope_cache(seq_len, head_dim):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("s,d->sd", pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: [batch, heads, seq, head_dim]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(variance + self.eps)
        return normed.to(x.dtype) * self.gain.to(x.dtype)


class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(GROUP, dim=1)
        v = v.repeat_interleave(GROUP, dim=1)
        # is_causal SDPA: fused kernel, PyTorch's equivalent of a hand-rolled
        # flash-attention path -- the fairest "PyTorch at its best" comparison.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = GQAAttention(D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM)
        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = SwiGLU(D_MODEL, D_FF)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(D_MODEL)

    def forward(self, ids, cos, sin):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return F.linear(x, self.embed.weight)


def bench_shape(batch, seq_len):
    # GradScaler expects fp32 master weights/grads with autocast doing the
    # fp16 casting internally -- a model manually cast to fp16 (model.to(dtype))
    # has fp16 grads, which GradScaler.unscale_ refuses ("Attempting to
    # unscale FP16 gradients"). Keep the model fp32, autocast the forward.
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    compiled = torch.compile(model)
    cos, sin = rope_cache(seq_len, HEAD_DIM)

    ids = torch.randint(0, VOCAB, (batch, seq_len + 1), device=device)
    inputs, targets = ids[:, :-1], ids[:, 1:]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = compiled(inputs, cos, sin)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    losses = []
    for _ in range(WARMUP):
        losses.append(step())
    torch.cuda.synchronize()

    trial_times = []
    for _ in range(N_TRIALS):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(STEPS_PER_TRIAL):
            losses.append(step())
        torch.cuda.synchronize()
        trial_times.append((time.time() - start) / STEPS_PER_TRIAL)

    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = batch * seq_len
    tok_per_sec = [tokens_per_step / t for t in trial_times]
    return {
        "params_m": params / 1e6,
        "ms_per_step_mean": statistics.mean(trial_times) * 1000,
        "ms_per_step_stdev": statistics.stdev(trial_times) * 1000 if len(trial_times) > 1 else 0.0,
        "tok_per_sec_mean": statistics.mean(tok_per_sec),
        "loss_finite": all(l == l and abs(l) < 1e6 for l in losses),
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
