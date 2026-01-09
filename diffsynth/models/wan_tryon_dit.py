import torch
import torch.nn as nn
from typing import Tuple


from .wan_video_dit import (
    WanModel, 
    DiTBlock, 
    CrossAttention, 
    RMSNorm,      
    modulate, 
    flash_attention,
    sinusoidal_embedding_1d,
)

class WanProjectorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WanBasicBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn1 = CrossAttention(dim, num_heads, eps=eps, has_image_input=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn2 = CrossAttention(dim, num_heads, eps=eps, has_image_input=False) 
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ff = WanProjectorMLP(dim, int(dim * 4), dim)

    def forward(self, x, encoder_hidden_states=None):
        x = x + self.attn1(self.norm1(x), self.norm1(x))
        if encoder_hidden_states is not None:
            x = x + self.attn2(self.norm2(x), encoder_hidden_states)
        x = x + self.ff(self.norm3(x))
        return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    
    def forward(self, x, latents, shift=None, scale=None):
        x = self.norm1(x)
        latents_kv = self.norm2(latents)
        
        if shift is not None and scale is not None:
            latents_kv = latents_kv * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        q = self.to_q(latents_kv)
        kv_input = torch.cat((x, latents_kv), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        out = flash_attention(q, k, v, num_heads=self.heads)
        return self.to_out(out)

class TimeResampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=256,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    WanProjectorMLP(dim, int(dim * ff_mult), dim),
                    nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
                    nn.LayerNorm(dim) 
                ])
            )

        self.timestep_in_dim = timestep_in_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_in_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, timestep):
        t_freq = sinusoidal_embedding_1d(self.timestep_in_dim, timestep).to(x.dtype)
        t_emb = self.time_embedding(t_freq) 
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        x = x + t_emb[:, None, :] 

        # 解包时多接收一个 norm
        for attn, ff, adaLN_modulation, norm_ff in self.layers:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(t_emb).chunk(4, dim=1)
            
            # Attention Block
            latents = attn(x, latents, shift_msa, scale_msa) + latents
            
            # FFN Block (Pre-Norm logic)
            res = latents
            # 1. 先 Norm
            latents = norm_ff(latents)
            # 2. 再 Modulate (AdaLN)
            latents = latents * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            # 3. 最后 MLP
            latents = ff(latents) + res

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        return latents, t_emb


class CrossLayerCrossScaleProjector(nn.Module):
    def __init__(
        self,
        inner_dim=2688,
        num_attention_heads=42,
        cross_attention_dim=2688,
        num_layers=4,
        # Resampler 参数
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=1024,
        embedding_dim=1152 + 1536,
        output_dim=4096,
        timestep_in_dim=256,
    ):
        super().__init__()
        
        head_dim = inner_dim // num_attention_heads
        
        self.cross_layer_blocks = nn.ModuleList([
            WanBasicBlock(inner_dim, num_attention_heads, head_dim)
            for _ in range(num_layers)
        ])

        self.cross_scale_blocks = nn.ModuleList([
            WanBasicBlock(inner_dim, num_attention_heads, head_dim)
            for _ in range(num_layers)
        ])

        self.proj = WanProjectorMLP(inner_dim, inner_dim*2, inner_dim)
        self.proj_cross_layer = WanProjectorMLP(inner_dim, inner_dim*2, inner_dim)
        self.proj_cross_scale = WanProjectorMLP(inner_dim, inner_dim*2, inner_dim)

        self.resampler = TimeResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            timestep_in_dim=timestep_in_dim
        )

    def forward(self, low_res_shallow, low_res_deep, high_res_deep, timesteps, need_temb=True):
        cross_layer_hidden_states = low_res_deep
        for block in self.cross_layer_blocks:
            cross_layer_hidden_states = block(
                cross_layer_hidden_states, 
                encoder_hidden_states=low_res_shallow
            )
        cross_layer_hidden_states = self.proj_cross_layer(cross_layer_hidden_states)

        cross_scale_hidden_states = low_res_deep
        for block in self.cross_scale_blocks:
            cross_scale_hidden_states = block(
                cross_scale_hidden_states,
                encoder_hidden_states=high_res_deep
            )
        cross_scale_hidden_states = self.proj_cross_scale(cross_scale_hidden_states)
        
        hidden_states = self.proj(low_res_deep) + cross_scale_hidden_states
        hidden_states = torch.cat([hidden_states, cross_layer_hidden_states], dim=1) 
        
        hidden_states, timestep_emb = self.resampler(hidden_states, timesteps)
        return hidden_states, timestep_emb

class Adapter(nn.Module):
    """Lightweight residual adapter for Cloth Latents"""
    def __init__(self, dim, adapter_dim=64):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, dim)
        )
    def forward(self, x):
        return self.ffn(x)

class WanVTONCrossAttention(CrossAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, num_heads, eps, has_image_input=False)
        
        self.k_ip = nn.Linear(dim, dim)
        self.v_ip = nn.Linear(dim, dim)
        self.norm_k_ip = RMSNorm(dim, eps=eps)

        self.k_cloth = nn.Linear(dim, dim)
        self.v_cloth = nn.Linear(dim, dim)
        self.norm_k_cloth = RMSNorm(dim, eps=eps)
        self.adapter_cloth = Adapter(dim)

        with torch.no_grad():
            nn.init.zeros_(self.v_cloth.weight)
            nn.init.zeros_(self.v_cloth.bias)
            nn.init.zeros_(self.v_ip.weight)
            nn.init.zeros_(self.v_ip.bias)

    def forward(self, x, context, ip_hidden_states, cloth_latents, context_lens=None):
        """
        实现 Text + IP + Cloth 三路 Attention 并求和
        """
        q = self.norm_q(self.q(x))

        k_text = self.norm_k(self.k(context))
        v_text = self.v(context)
        x_text = flash_attention(q, k_text, v_text, num_heads=self.num_heads)

        k_ip = self.norm_k_ip(self.k_ip(ip_hidden_states))
        v_ip = self.v_ip(ip_hidden_states)
        x_ip = flash_attention(q, k_ip, v_ip, num_heads=self.num_heads)

        cloth_latents = self.adapter_cloth(cloth_latents) + cloth_latents
        k_cloth = self.norm_k_cloth(self.k_cloth(cloth_latents)) 
        v_cloth = self.v_cloth(cloth_latents) 
        x_cloth = flash_attention(q, k_cloth, v_cloth, num_heads=self.num_heads)

        return self.o(x_text + x_ip + x_cloth)


class WanVTONDiTBlock(DiTBlock):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, eps=eps)
        self.cross_attn = WanVTONCrossAttention(dim, num_heads, eps)

    def forward(self, x, context, t_mod, freqs, ip_hidden_states, cloth_latents):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        x = x + self.cross_attn(
            self.norm3(x), 
            context=context, 
            ip_hidden_states=ip_hidden_states, 
            cloth_latents=cloth_latents
        )

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x



class WanVTONDiT(WanModel):
    def __init__(
        self,
        # 保持与 WanModel 一致的基础参数
        dim: int = 1536,
        in_dim: int = 16,
        ffn_dim: int = 8960,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 12,
        num_layers: int = 30,
        cross_attention_dim: int = 2304, # DINO + SigLIP features dim? 
        **kwargs
    ):
        super().__init__(
            dim=dim, in_dim=in_dim, ffn_dim=ffn_dim, out_dim=out_dim,
            text_dim=text_dim, freq_dim=freq_dim, eps=eps, patch_size=patch_size,
            num_heads=num_heads, num_layers=num_layers,
            has_image_input=False, # 我们手动处理
            **kwargs
        )

        # 1. Projector (The heavy lifter)
        self.subject_image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=2304,          # 根据你的模型配置调整
            num_attention_heads=36,  # 2304 / 64
            cross_attention_dim=2304,
            embedding_dim=2304, 
            output_dim=dim,          # 映射到 DiT dim
            timestep_in_dim=freq_dim
        )

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.condition_embedding = nn.Conv3d(
            48, dim, kernel_size=patch_size, stride=patch_size
        )
        with torch.no_grad():
            nn.init.zeros_(self.condition_embedding.weight)
            nn.init.zeros_(self.condition_embedding.bias)

        self.cloth_patch_embedding = nn.Conv3d(
            48, dim, kernel_size=patch_size, stride=patch_size
        )

        self.blocks = nn.ModuleList([
            WanVTONDiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
    

    def forward(
        self,
        x: torch.Tensor,        
        timestep: torch.Tensor,
        context: torch.Tensor,  
        cloth_latents: torch.Tensor,
        densepose_latents: torch.Tensor,
        ip_hidden_states: dict,  
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        t_adapter = timestep.to(dtype=x.dtype)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        

        context = self.text_embedding(context)

        ip_feats = self.subject_image_proj_model(
            ip_hidden_states['image_embeds_low_res_shallow'],
            ip_hidden_states['image_embeds_low_res_deep'],
            ip_hidden_states['image_embeds_high_res_deep'],
            timesteps=t_adapter, 
            need_temb=True
        )[0] 
        
        x, (f, h, w) = self.patchify(x) 
  
        y_dense = self.condition_embedding(densepose_latents)
        y_dense = y_dense.flatten(2).transpose(1, 2)
        x = x + y_dense

        x_cloth = self.cloth_patch_embedding(cloth_latents)
        x_cloth = x_cloth.flatten(2).transpose(1, 2)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs, ip_feats, x_cloth, # 传参
                    use_reentrant=False
                )
            else:
                x = block(
                    x, context, t_mod, freqs, 
                    ip_hidden_states=ip_feats, 
                    cloth_latents=x_cloth
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x