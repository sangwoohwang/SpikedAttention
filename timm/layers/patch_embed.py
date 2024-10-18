""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from .format import Format, nchw_to
from .helpers import to_2tuple
from .trace_utils import _assert

_logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_layer = norm_layer
    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def SpikeSim_Energy(flop, time_steps, ratio,potential_ratio,weight_num):
    N_rd = flop
    N_neuron = flop/weight_num
    E_ad = 0.9 ##0.03
    E_mul = 3.7 ##0.2
    E_mem = 5.00#1.25

    E_rd = N_rd* (E_mem*33/32)*ratio
    E_acc = N_rd * E_ad*ratio
    
    E_state = (E_mem + E_mul + E_ad + E_ad + E_ad + E_mem)*N_neuron*potential_ratio
    E_offmap = (E_mem/32)*N_neuron*potential_ratio

    E_snn = E_rd + E_acc + E_state + E_offmap

    return E_snn

def ANN_Energy(flop, weight_num,sparsity):
    N_rd = flop
    N_neuron = flop/weight_num
    E_ad = 0.9 ##0.03
    E_mul = 3.7 ##0.2
    E_mem = 5.00#1.25

    E_rd = N_rd* (E_mem*2) *sparsity
    E_acc = N_rd * (E_ad+E_mul) *sparsity
    E_offmap = (E_mem)*N_neuron

    E_snn = E_rd + E_acc +  E_offmap

    return E_snn




class batch_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            relu_patch: bool =False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.timestep =0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_layer = norm_layer
        self.relu_patch = relu_patch
        if(self.relu_patch==True):
            self.patch_embed_if  = nn.ReLU()
        else:
            self.patch_embed_if  = nn.Identity()
        self.snn_mode = False
        self.time =0
        self.tp =0
        self.embed_dim =embed_dim
        self.in_chans = in_chans
    def forward(self, x):
        
        if(x is not None):
            B, C, H, W = x.shape
            if self.img_size is not None:
                if self.strict_img_size:
                    _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                    _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
                elif not self.dynamic_img_pad:
                    _assert(
                        H % self.patch_size[0] == 0,
                        f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                    )
                    _assert(
                        W % self.patch_size[1] == 0,
                        f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                    )
            if self.dynamic_img_pad:
                assert False, "dynamic_img_pad"
                pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
                pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
            
            x = self.proj(x)

   
            B, C, H, W = x.shape
            x = self.norm(x.flatten(2).contiguous()).reshape(B, C, H, W).contiguous()

            
            if self.output_fmt != Format.NCHW:
                x = nchw_to(x, self.output_fmt)

        x= self.patch_embed_if(x)

        return x
    def flops(self):
        flops=0
        flops = self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += self.num_patches * self.embed_dim
        return flops
    def flops_snn(self, input_ratio):
        flops=0
        flops += SpikeSim_Energy(self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1]),self.timestep, input_ratio, self.patch_embed_if.mem_count_meter.avg,self.num_patches * self.embed_dim  )

        return flops,self.patch_embed_if.spike_count_meter.avg

    def flops_ANN(self,input_ratio):
        flops=0
        flops += ANN_Energy(self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1]),self.num_patches * self.embed_dim,input_ratio  )

        return flops,self.patch_embed_if.spike_count_meter.avg


    def flops_snn2(self,input_ratio):
        flops=0
        flops = self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])*input_ratio

        return flops,self.patch_embed_if.spike_count_meter.avg

class PatchEmbedWithSize(PatchEmbed):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H % self.patch_size[0] == 0, f"Input image height ({H}) must be divisible by patch size ({self.patch_size[0]}).")
            _assert(W % self.patch_size[1] == 0, f"Input image width ({W}) must be divisible by patch size ({self.patch_size[1]}).")

        x = self.proj(x)
        grid_size = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x, grid_size


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        import functorch
        vmap = functorch.vmap
    except ImportError:
        if hasattr(torch, 'vmap'):
            vmap = torch.vmap
        else:
            assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


