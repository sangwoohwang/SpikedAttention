#!/usr/bin/env python3

import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name, resume_checkpoint
from timm.utils import ApexScaler, NativeScaler
from utils_neuron import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


logger = logging.getLogger('Spiked-Attention')

# file_handler = logging.FileHandler('Swin-SpikedAttention.log')
# logger.addHandler(file_handler)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')

group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='swin_tiny_patch4_window7_224', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--num-classes', type=int, default=1000, metavar='N',
                   help='number of label classes (Model default if None)')

group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('--model-kwargs', nargs='*',
                   default={}, action=utils.ParseKwargs)

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    'Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')

group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--base', default=1.15, type=float, metavar='N',
                    help='base of one spike')
parser.add_argument('--timestep',  default=40, type=int,
                    metavar='N', help='number of timestep')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    # args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        logger.info(
            'Test in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Test with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        global_pool=args.gp,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )

    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # FIXME handle model default vs config num_classes more elegantly
        args.num_classes = model.num_classes

    if utils.is_primary(args):
        logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(
        vars(args), model=model, verbose=utils.is_primary(args))

    model.to(device=device)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(
                torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            log_info=utils.is_primary(args),
        )

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[
                              device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data

    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
    )

    eval_workers = 1  # args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
    )

    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    try:

        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            args,
            amp_autocast=amp_autocast,
        )

        eval_metrics = snn_validate(
            model,
            loader_eval,
            validate_loss_fn,
            args,
            amp_autocast=amp_autocast,
        )

    except KeyboardInterrupt:
        pass


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    if utils.is_primary(args):
        logger.info(
            "#1: Run pre-trained Model(ANN) ")

    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    model, n_layer = replace_identity_by_module(model, 0, int(args.batch_size))

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'ANN' + log_suffix
                logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})  '
                )

    if utils.is_primary(args):

        metrics = OrderedDict(
            [('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        log_name = 'Pre-trained ANN Performance' + log_suffix
        if args.distributed:
            energy = model.module.flops_ANN(1.0)
            flops = model.module.flops()            
        else:
            energy = model.flops_ANN(1.0)
            flops = model.flops()            
        logger.info(
            f'{log_name}:  '
            f'Acc@1: {top1_m.avg:>7.3f}  '
            f'Acc@5: {top5_m.avg:>7.3f}  '
            f"number of GFLOPs: {flops / 1e9 :>7.3f}  "
            f"SynOP ANN Energy (mJ): {flops*4.6 / 1e9 :>7.3f}  "
            f"Total ANN Energy (mJ): {energy / 1e9 :>7.3f}  "
        )


        return metrics
    return


def snn_validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    timestep = args.timestep
    wait = args.timestep
    input_count_meter = utils.AverageMeter()

    base = float(args.base)

    model, n_layer = replace_ANN_neruon_by_neuron_wait(model, timestep, wait, 0, base)
    model, i_layer_bias, i_layer_mean = modif_bias(model, timestep, base, 0, 0)
    if utils.is_primary(args):
        logger.info("Embed Spiking Neuron on each layer")
        logger.info(f"Base of SNN: {args.base}  ")
        logger.info(f"total timesteps of SNN: {args.timestep}  ")
        logger.info("Validate Spiked-Attention: Accuracy and Energy")

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)

            images_tp = input.clone() / input.max()
            input_sign = input.sign()

            with amp_autocast():
                output = 0
                input_max = input.max()
                input_count = 0

                for t in range((n_layer-1)*wait+timestep):
                    if (t < timestep):
                        images_tp *= 2.0
                        input_spike = torch.where(
                            images_tp.abs() >= 1, 1, 0)*input_sign
                        images_tp = torch.where(
                            input_spike != 0, images_tp-input_spike, images_tp)
                        output_tp = model(input_spike*input_max)
                        input_count += torch.count_nonzero(input_spike)/(
                            input_spike.flatten()).size()[0]

                    else:
                        input = None
                        output = model(input)
                input_count_meter.update(input_count)

                reset_net(model)
                if isinstance(output, (tuple, list)):
                    output = output[0]

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                if args.distributed:
                    energy = model.module.flops_snn(input_count_meter.avg)
                    flops = model.module.flops_snn2(input_count_meter.avg)

                else:
                    energy = model.flops_snn(input_count_meter.avg)
                    flops = model.flops_snn2(input_count_meter.avg)                
                log_name = 'SNN' + log_suffix
                logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})  '
                    f"SynOP ANN Energy (mJ): ({flops *0.9 / 1e9 :>7.3f})  "
                    f"Total ANN Energy (mJ): ({energy / 1e9 :>7.3f})  "
                )

            if (top1_m.avg < 10):
                print("break bc of too much low accuracy")
                break

    if utils.is_primary(args):

        log_name = 'Converted Spiked-Attention performance' + log_suffix
        if args.distributed:
            energy = model.module.flops_snn(input_count_meter.avg)
            flops = model.module.flops_snn2(input_count_meter.avg)

        else:
            energy = model.flops_snn(input_count_meter.avg)
            flops = model.flops_snn2(input_count_meter.avg)


        metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
        logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})  '
                    f"SynOP ANN Energy (mJ): ({flops *0.9 / 1e9 :>7.3f})  "
                    f"Total ANN Energy (mJ): ({energy / 1e9 :>7.3f})  "
        )

        return metrics
    return


if __name__ == '__main__':
    main()
