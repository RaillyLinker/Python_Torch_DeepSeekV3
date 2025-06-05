import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (batch, seq)
        # output: (batch, seq, emb_size) scaled by sqrt(emb_size)
        return self.embedding(tokens) * self.scale


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, seq)
        tok_emb = self.token(x)  # (batch, seq, d_model)
        return self.dropout(tok_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., 2*k) -> (.., k), (.., k)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(dim: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0, "Rotary dimension must be even"
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )  # (dim/2,)
    t = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
    sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
    return cos, sin


def apply_rotary(q_r: torch.Tensor, k_r: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    # q_r, k_r: shape (batch, heads, seq_len, rot_dim)
    q_rot = (q_r * cos) + (rotate_half(q_r) * sin)
    k_rot = (k_r * cos) + (rotate_half(k_r) * sin)
    return q_rot, k_rot


class MultiheadLatentAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            latent_dim_q: Optional[int] = None,
            latent_dim_kv: Optional[int] = None,
            dropout: float = 0.1,
            max_seq_len: int = 131072,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim % 2 == 0, "head_dim must be even to split for RoPE"
        self.rot_dim = self.head_dim // 2
        self.cont_dim = self.head_dim - self.rot_dim  # typically = rot_dim

        self.dropout = nn.Dropout(dropout)

        if latent_dim_q is None:
            latent_dim_q = d_model // 2
        if latent_dim_kv is None:
            latent_dim_kv = d_model // 4
        self.latent_dim_q = latent_dim_q
        self.latent_dim_kv = latent_dim_kv

        self.latent_proj_q = nn.Linear(d_model, latent_dim_q, bias=False)
        self.latent_proj_kv = nn.Linear(d_model, latent_dim_kv, bias=False)

        self.rot_q = nn.Linear(latent_dim_q, self.num_heads * self.rot_dim * 2, bias=False)
        self.rot_k = nn.Linear(latent_dim_kv, self.num_heads * self.rot_dim * 2, bias=False)

        self.q_up = nn.Linear(latent_dim_q, d_model, bias=False)
        self.k_up = nn.Linear(latent_dim_kv, d_model, bias=False)
        self.v_up = nn.Linear(latent_dim_kv, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos_buf, sin_buf = build_rotary_pos_emb(self.rot_dim, max_seq_len)
        self.register_buffer("rotary_cos", cos_buf, persistent=True)
        self.register_buffer("rotary_sin", sin_buf, persistent=True)

    def forward(
            self,
            x: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        dtype = x.dtype

        z_q = self.latent_proj_q(x)  # (batch, seq_len, latent_dim_q)
        z_kv = self.latent_proj_kv(x)  # (batch, seq_len, latent_dim_kv)

        rot_q_out = self.rot_q(z_q).view(batch_size, seq_len, self.num_heads, self.rot_dim * 2)
        rot_k_out = self.rot_k(z_kv).view(batch_size, seq_len, self.num_heads, self.rot_dim * 2)

        q_cont = self.q_up(z_q).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_cont = self.k_up(z_kv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_up(z_kv).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q_r = rot_q_out[..., : self.rot_dim]  # (batch, seq_len, heads, rot_dim)
        q_r = q_r.permute(0, 2, 1, 3)  # (batch, heads, seq_len, rot_dim)
        k_r = rot_k_out[..., : self.rot_dim]  # (batch, seq_len, heads, rot_dim)
        k_r = k_r.permute(0, 2, 1, 3)  # (batch, heads, seq_len, rot_dim)

        cos = self.rotary_cos[:, :, :seq_len, :self.rot_dim].to(dtype=dtype)  # (1,1,seq_len,rot_dim)
        sin = self.rotary_sin[:, :, :seq_len, :self.rot_dim].to(dtype=dtype)

        q_r_rot, k_r_rot = apply_rotary(q_r, k_r, cos, sin)  # both (batch, heads, seq_len, rot_dim)

        q_cont = q_cont.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_cont = k_cont.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_c, _ = q_cont.split([self.cont_dim, self.rot_dim], dim=-1)  # (batch, heads, seq_len, cont_dim)
        k_c, _ = k_cont.split([self.cont_dim, self.rot_dim], dim=-1)

        q = torch.cat((q_c, q_r_rot), dim=-1)  # (batch, heads, seq_len, head_dim)
        k = torch.cat((k_c, k_r_rot), dim=-1)  # (batch, heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask  # (batch, heads, seq_len, seq_len)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # Pre-norm → sublayer → dropout → residual add
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)  # (batch, seq, d_ff*2)
        x = self.activation(x)  # (batch, seq, d_ff)
        x = self.linear2(x)  # (batch, seq, d_model)
        return self.dropout(x)


class MoEPositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int,
            top_k: int = 2,
            num_shared_experts: int = 1,
            dropout: float = 0.1,
            noise_std: float = 1.0,
            capacity_factor: float = 1.0,
            bias_update_speed: float = 0.1,
    ):
        super().__init__()
        assert num_shared_experts < num_experts, "num_shared_experts must be less than num_experts"

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        self.bias_update_speed = bias_update_speed

        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, self.num_routed_experts, bias=False)

        self.expert_bias = nn.Parameter(torch.zeros(self.num_routed_experts))

    @torch.no_grad()
    def update_bias(self, expert_load: torch.Tensor):
        target = expert_load.mean().item()
        delta = (expert_load - target) / (expert_load + 1e-6) * self.bias_update_speed
        self.expert_bias.data = self.expert_bias.data - delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model
        total_tokens = batch_size * seq_len

        flat_x = x.view(total_tokens, d_model)  # (Ntokens, d_model)

        if self.num_shared_experts > 0:
            shared_sum = torch.zeros(total_tokens, d_model, device=x.device, dtype=x.dtype)
            for e in range(self.num_shared_experts):
                shared_sum += self.experts[e](flat_x)
        else:
            shared_sum = torch.zeros(total_tokens, d_model, device=x.device, dtype=x.dtype)

        gate_logits = self.gate(flat_x)  # (Ntokens, num_routed_experts)
        if self.training and self.noise_std > 0.0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        gate_logits = gate_logits + self.expert_bias.unsqueeze(0)  # broadcast over tokens

        gate_affinity = torch.sigmoid(gate_logits)  # (Ntokens, num_routed_experts)

        topk_vals, topk_idx = gate_affinity.topk(self.top_k, dim=-1)  # each (Ntokens, top_k)

        capacity = max(1, int(self.capacity_factor * total_tokens / self.num_routed_experts))

        routed_sum = torch.zeros(total_tokens, d_model, device=x.device, dtype=x.dtype)
        expert_load = torch.zeros(self.num_routed_experts, device=x.device, dtype=x.dtype)

        for e_r in range(self.num_routed_experts):
            mask_e = (topk_idx == e_r).any(dim=-1)  # (Ntokens,)
            positions = torch.nonzero(mask_e, as_tuple=False).squeeze(-1)  # (n_e,)
            n_e = positions.numel()
            if n_e == 0:
                continue

            if n_e > capacity:
                scores_e = gate_affinity[positions, e_r]  # (n_e,)
                top_pos_idx = torch.argsort(scores_e, descending=True)[:capacity]
                positions = positions[top_pos_idx]
                n_e = capacity

            inputs_e = flat_x[positions]  # (n_e, d_model)
            outputs_e = self.experts[self.num_shared_experts + e_r](inputs_e)  # (n_e, d_model)

            weights_e = gate_affinity[positions, e_r].unsqueeze(-1)  # (n_e, 1)
            routed_sum[positions] += outputs_e * weights_e

            expert_load[e_r] = n_e

        if self.training:
            self.update_bias(expert_load)

        combined = (shared_sum + routed_sum).view(batch_size, seq_len, d_model)  # (batch, seq_len, d_model)
        return combined


class DecoderBlockMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            hidden_dim: int,
            num_experts: int,
            top_k: int = 2,
            num_shared_experts: int = 1,
            dropout: float = 0.1,
            noise_std: float = 0.0,
            capacity_factor: float = 1.0,
            bias_update_speed: float = 0.1,
            max_seq_len: int = 131072,
            latent_dim_q: Optional[int] = None,
            latent_dim_kv: Optional[int] = None,
    ):
        super().__init__()
        self.attn = MultiheadLatentAttention(
            d_model=dim,
            num_heads=num_heads,
            latent_dim_q=latent_dim_q,
            latent_dim_kv=latent_dim_kv,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.ff = MoEPositionwiseFeedForward(
            d_model=dim,
            d_ff=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            dropout=dropout,
            noise_std=noise_std,
            capacity_factor=capacity_factor,
            bias_update_speed=bias_update_speed,
        )
        self.sublayers = nn.ModuleList([
            SublayerConnection(dim, dropout),
            SublayerConnection(dim, dropout),
        ])

    def forward(
            self,
            x: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.sublayers[0](x, lambda _x: self.attn(_x, causal_mask=causal_mask))

        x_norm = self.sublayers[1].norm(attn_out)  # RMSNorm only
        moe_out = self.ff(x_norm)  # (batch, seq_len, dim)
        out = attn_out + self.sublayers[1].dropout(moe_out)
        return out


class DeepSeekV3(nn.Module):
    def __init__(
            self,
            vocab_size: int = 32000,
            dim: int = 16384,
            n_layers: int = 61,
            num_heads: int = 128,
            hidden_dim: int = 2048,
            num_experts: int = 257,
            top_k: int = 8,
            num_shared_experts: int = 1,
            max_len: int = 131072,
            dropout: float = 0.1,
            noise_std: float = 0.0,
            capacity_factor: float = 1.0,
            bias_update_speed: float = 0.1,
            latent_dim_q: Optional[int] = None,
            latent_dim_kv: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = InputEmbedding(vocab_size, dim, dropout)

        self.layers = nn.ModuleList([
            DecoderBlockMoE(
                dim=dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                dropout=dropout,
                noise_std=noise_std,
                capacity_factor=capacity_factor,
                bias_update_speed=bias_update_speed,
                max_seq_len=max_len,
                latent_dim_q=latent_dim_q,
                latent_dim_kv=latent_dim_kv,
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.token.embedding.weight  # weight tying

    def forward(
            self,
            x: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x_emb = self.embed(x)  # (batch, seq_len, dim)

        device = x.device
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )  # (seq_len, seq_len) with -inf above diagonal
        causal_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)  # (batch, 1, seq_len, seq_len)

        hidden = x_emb
        for layer in self.layers:
            hidden = layer(hidden, causal_mask=causal_mask)

        hidden = self.norm(hidden)  # (batch, seq_len, dim)
        logits = self.head(hidden)  # (batch, seq_len, vocab_size)

        return logits
