import torch
import torch.nn as nn
from modules_neuron import StraightThrough, WTA_layer_Neuron, ANN_neruon,ScaledNeuron_onespike_time_bipolar, ScaledNeuron_onespike_time_relu


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False


def have_bias(name):
    if 'Linear' in name.lower() or 'BatchNorm' in name.lower() or 'Conv2d' in name.lower():
        return True
    return False


def issigmoid(name):
    if 'sigmoid' in name.lower():
        return True
    return False


def replace_identity_by_module(model, i_layer, batch_size):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], i_layer = replace_identity_by_module(
                module, i_layer, batch_size)

        if ((module.__class__.__name__ == "Identity" or module.__class__.__name__ == "ReLU") and name != "downsample" and name != "drop_path1" and name != "drop_path2" and name != "flatten"):

            model._modules[name] = ANN_neruon(batch_size=batch_size)
            model._modules[name].i_layer = i_layer
            model._modules[name].name = name
            if (name != "q_if" and name != "k_if"):
                i_layer += 1

            if (module.__class__.__name__ == "ReLU"):
                model._modules[name].relu_bool = True
    return model, i_layer


def replace_ANN_neruon_by_neuron_wait(model, timestep, wait, n_layer, tau):

    if hasattr(model, "snn_mode"):
            model.snn_mode = True
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], n_layer = replace_ANN_neruon_by_neuron_wait(
                module, timestep, wait, n_layer, tau)
        if hasattr(module, "snn_mode"):
            model._modules[name].snn_mode = True
        if hasattr(module, "tau"):
            model._modules[name].tau = tau
        if hasattr(module, "timestep"):
            model._modules[name].timestep = timestep
        if hasattr(module, "max_act"):
            if (model._modules[name].relu_bool == True):
                neuron_act = ScaledNeuron_onespike_time_relu
            else:
                neuron_act = ScaledNeuron_onespike_time_bipolar

            if (isinstance(module.max_act, int)):
                assert False, (name, n_layer)
            if module.max_act.any():
                if (n_layer == 0):
                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait *
                                                      n_layer, i_layer=n_layer, tau=tau, convert=True, scale_full=model._modules[name].full_scale)
                    n_layer += 1
                elif (name == "stdp_av"):
                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait*n_layer, i_layer=n_layer,
                                                      tau=tau, convert=False, modulename=name, trace_bool=True, stdp_bool=True, scale_full=model._modules[name].full_scale)
                    n_layer += 1

                elif (name == "q_if" or name == "k_if" or name == "v_if"):

                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait*n_layer, i_layer=n_layer,
                                                      tau=tau, convert=False, modulename=name, trace_bool=True, stdp_bool=False, scale_full=model._modules[name].full_scale)
                    if (name == "v_if"):
                        n_layer += 1
                elif (name == "softmax_if"):
                    model._modules[name] = WTA_layer_Neuron(
                        scale=1, timestep=timestep, wait=wait, start_time=wait*n_layer, i_layer=n_layer, tau=tau, convert=False, modulename=name, trace_bool=True)
                    n_layer += 1
                elif (name == "last_fc_if"):
                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait*n_layer,
                                                      i_layer=n_layer, tau=tau, convert=False, modulename=name, scale_full=model._modules[name].full_scale, final_bool=True)
                    n_layer += 1
                else:

                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait*n_layer,
                                                      i_layer=n_layer, tau=tau, convert=False, modulename=name, scale_full=model._modules[name].full_scale)
                    n_layer += 1

            else:
                if (n_layer == 0):
                    model._modules[name] = neuron_act(scale=1, timestep=timestep, start_time=wait*n_layer,
                                                      i_layer=n_layer, tau=tau, convert=True, scale_full=model._modules[name].full_scale)
                    n_layer += 1
                elif (name == "stdp_qk"):
                    model._modules[name] = neuron_act(scale=module.max_act, timestep=timestep, wait=wait, start_time=wait*n_layer, i_layer=n_layer,
                                                      tau=tau, convert=False, modulename=name, trace_bool=True, stdp_bool=True, scale_full=model._modules[name].full_scale)
                    n_layer += 1

                elif (name == "q_if" or name == "k_if" or name == "v_if"):
                    model._modules[name] = neuron_act(scale=1, timestep=timestep, start_time=wait*n_layer, i_layer=n_layer,
                                                      tau=tau, convert=False, modulename=name, trace_bool=True, scale_full=model._modules[name].full_scale)
                    if (name == "v_if"):
                        n_layer += 1
                elif (name == "softmax_if"):
                    model._modules[name] = WTA_layer_Neuron(
                        scale=1.0, timestep=timestep, wait=wait, start_time=wait*n_layer, i_layer=n_layer, tau=tau, convert=False, modulename=name, trace_bool=True)
                    n_layer += 1

                elif (name != "drop_path1"):
                    model._modules[name] = neuron_act(scale=1, timestep=timestep, start_time=wait*n_layer,
                                                      i_layer=n_layer, tau=tau, convert=False, scale_full=model._modules[name].full_scale)
                    n_layer += 1
            # print(name,n_layer-1)

    return model, n_layer


def modif_bias(model, timestep, base, i_layer_bias, i_layer_mean):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], i_layer_bias, i_layer_mean = modif_bias(
                module, timestep, base, i_layer_bias, i_layer_mean)

        if hasattr(module, "bias"):
            if (torch.is_tensor(model._modules[name].bias)):
                if (i_layer_bias > 1):
                    model._modules[name].bias = nn.Parameter(
                        model._modules[name].bias / ((1-1/(base**timestep))/(1-1/base)-1))
                i_layer_bias += 1

        if hasattr(module, "running_mean"):
            if (torch.is_tensor(model._modules[name].running_mean)):
                if (i_layer_mean > 0):
                    model._modules[name].running_mean = nn.Parameter(
                        model._modules[name].running_mean/((1-1/(base**timestep))/(1-1/base)-1))
                i_layer_mean += 1

    return model, i_layer_bias, i_layer_mean


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
