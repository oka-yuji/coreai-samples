import torch
from torch import nn
from torch.nn import functional as F


class RealAttention:
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, **kwargs):
        cosine, sine = rotary_emb
        values = []
        for projection, norm in ((attn.to_q, attn.norm_q), (attn.to_k, attn.norm_k), (attn.to_v, None)):
            value = projection(hidden_states).unflatten(-1, (attn.heads, attn.inner_dim // attn.heads))
            if norm is not None:
                value = norm(value)
                pairs = value.float().reshape(*value.shape[:-1], -1, 2)
                real, imag = pairs[..., 0], pairs[..., 1]
                value = torch.stack((real * cosine - imag * sine, imag * cosine + real * sine), dim=-1).flatten(-2).to(hidden_states.dtype)
            values.append(value.transpose(1, 2))
        value = F.scaled_dot_product_attention(*values, attn_mask=attention_mask).transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](value))


class Denoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_in, self.txt_in = model.img_in, model.txt_in
        self.time_text_embed, self.modulation = model.time_text_embed, model.modulation
        self.blocks = model.transformer_blocks
        self.norm_out, self.proj_out = model.norm_out, model.proj_out
        for block in self.blocks:
            block.attn.set_processor(RealAttention())

    def forward(self, latents, text, timestep, text_index, image_index, image_mask, cosine, sine, mask, target):
        image_hidden, text_hidden = self.img_in(latents), self.txt_in(text)
        hidden = torch.where(image_mask[None, :, None], image_hidden[:, image_index], text_hidden[:, text_index])
        temb = self.time_text_embed(torch.cat((timestep, torch.zeros_like(timestep))), hidden)
        modulation = self.modulation(temb)
        for block in self.blocks:
            hidden = block(hidden, modulation, rotary_emb=(cosine, sine), attention_mask=mask, target_token_mask=target)
        return self.proj_out(self.norm_out(hidden, temb, target))


class VAEPart(nn.Module):
    def __init__(self, vae, kind):
        super().__init__()
        self.vae, self.kind = vae, kind

    def forward(self, value):
        if self.kind == "vae_encoder":
            return self.vae.encode(value).latent_dist.mode()
        return self.vae.decode(value, return_dict=False)[0]
