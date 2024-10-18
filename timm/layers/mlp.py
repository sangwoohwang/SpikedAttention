""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

from torch import nn as nn

from .grn import GlobalResponseNorm
from .helpers import to_2tuple


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
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

class Batch_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm =nn.BatchNorm1d(hidden_features)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features =hidden_features
        self.in_feature = in_features
        self.out_features = out_features
        self.timestep = 0

        # self.fc1_if = nn.Identity()

        self.mlp_if = nn.Identity()

    def forward(self, x):
        if(x is not None):
            x = self.fc1(x)
        x = self.act(x)

        if(x is not None):
            x = self.drop1(x)
            x = self.norm(x.permute(0,2,1).contiguous()).permute(0, 2,1).contiguous()
            
        x = self.mlp_if(x)

        if(x is not None):
            x = self.fc2(x)
            x = self.drop2(x)
        return x
    def flops_snn(self,input_ratio,H,W):
        flops =0
        flops += SpikeSim_Energy(H * W * self.in_feature * self.hidden_features,self.timestep, input_ratio, self.act.mem_count_meter.avg,self.in_feature )
        return flops,self.act.spike_count_meter.avg#,H * W * self.out_features * self.hidden_features,self.hidden_features


    def flops_ANN(self,H,W,input_ratio):
        flops =0
        flops += ANN_Energy(H * W * self.in_feature * self.hidden_features,self.in_feature ,input_ratio)
        return flops,self.act.spike_count_meter.avg#,H * W * self.out_features * self.hidden_features,self.hidden_features


    def flops_snn2(self,input_ratio,H,W):
        flops =0
        flops += H * W * self.in_feature * self.hidden_features*input_ratio
        flops += H * W * self.out_features * self.hidden_features *self.act.spike_count_meter.avg
        return flops,self.act.spike_count_meter.avg




class Batch_Mlp_relu(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm =nn.BatchNorm1d(hidden_features)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.hidden_features =hidden_features
        self.in_feature = in_features
        self.out_features = out_features
        # self.fc1_if = nn.ReLU()

        # self.mlp_if = nn.Identity()

    def forward(self, x):
        if(x is not None):
            x = self.fc1(x)

        if(x is not None):
            x = self.drop1(x)
            x = self.norm(x.permute(0,2,1).contiguous()).permute(0, 2,1).contiguous()
            
        x = self.act(x)

        if(x is not None):
            x = self.fc2(x)
            x = self.drop2(x)
        return x

    def flops_snn(self,input_ratio,H,W):
        flops =0
        flops += SpikeSim_Energy(H * W * self.in_feature * self.hidden_features,1, input_ratio, self.act.mem_count_meter.avg,self.in_feature )
        return flops,self.act.spike_count_meter.avg#,H * W * self.out_features * self.hidden_features,self.hidden_features

    def flops_snn2(self,input_ratio,H,W):
        flops =0
        flops += H * W * self.in_feature * self.hidden_features*input_ratio
        flops += H * W * self.out_features * self.hidden_features *self.act.spike_count_meter.val
        return flops,self.act.spike_count_meter.avg
    def flops_ANN(self,H,W,input_ratio):
        flops =0
        flops += ANN_Energy(H * W * self.in_feature * self.hidden_features,self.in_feature ,input_ratio)
        return flops,self.act.spike_count_meter.avg#,H * W * self.out_features * self.hidden_features,self.hidden_features






class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate
        self.norm = nn.BatchNorm1d(hidden_features)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
