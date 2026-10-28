from functools import lru_cache

import torch
from coreai_torch import MetalParameter, TorchMetalKernel


def reference_rotary(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    pairs = value.reshape(-1, 32, 64, 2)
    frequency = torch.complex(cosine, sine).reshape(-1, 1, 64)
    return torch.view_as_real(torch.view_as_complex(pairs.contiguous()) * frequency).reshape(-1)


@lru_cache(maxsize=1)
def rotary_kernel():
    source = """
        const uint offset = id * 2;
        const uint frequency = (id / (32 * 64)) * 64 + id % 64;
        const float real = value[offset];
        const float imag = value[offset + 1];
        const float co = cosine[frequency];
        const float si = sine[frequency];
        result[offset] = fma(real, co, -(imag * si));
        result[offset + 1] = fma(real, si, imag * co);
    """
    return TorchMetalKernel(name="qwen_image21_rotary_f32", input_names=["value", "cosine", "sine"],
                            result_names=["result"], src=source, torch_defn=reference_rotary,
                            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")])


class Float32Rotary(torch.nn.Module):
    def __init__(self, *, native_complex=False):
        super().__init__()
        self.native_complex = native_complex
        self.kernel = None if native_complex else rotary_kernel()

    @property
    def arithmetic(self):
        return "float32-native-complex" if self.native_complex else "float32-explicit-metal-fma"

    def forward(self, value, cosine, sine):
        if value.dtype != torch.float32 or cosine.dtype != torch.float32 or sine.dtype != torch.float32:
            raise ValueError("The rotary boundary requires Float32 inputs")
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[-2:] != (32, 128) or not 1 <= value.shape[1] <= 2304:
            raise ValueError("The rotary boundary requires one batch, 32 heads and 128 channels")
        if cosine.shape != (1, value.shape[1], 1, 64) or sine.shape != cosine.shape:
            raise ValueError("The rotary frequencies must cover every token and pair")
        if self.native_complex:
            return reference_rotary(value, cosine, sine).reshape(value.shape)
        return self.kernel(value.reshape(-1), cosine.reshape(-1), sine.reshape(-1),
                           threads_per_grid=(value.numel() // 2, 1, 1), threads_per_thread_group=(256, 1, 1),
                           result_shapes=[[value.numel()]]).reshape(value.shape)
