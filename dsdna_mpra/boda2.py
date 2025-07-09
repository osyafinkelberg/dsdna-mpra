"""
This module includes functions adapted from the boda2 project:
    https://github.com/sjgosai/boda2

Original authors: sjgosai and contributors

License: The original code is licensed under MIT (see LICENSE or LICENSE.mit in the boda2 repo).
         This permits use, modification, and distribution under the terms of the MIT License,
         including inclusion of the copyright and license notice in all copies.

Modifications:
    The original functions have been slightly modified to suit this project.
"""
import typing as tp
from pathlib import Path
import sys
import shutil
import tarfile
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, TensorDataset)
from torch.distributions.categorical import Categorical
import lightning.pytorch as ptl
import math
from collections import OrderedDict

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


def add_criterion_specific_args(parser, criterion_name):
    def add_reduction(group, default='mean'):
        group.add_argument(
            '--reduction',
            type=str,
            default=default,
            help=(
                'Specifies reduction applied when loss is calculated: `"none"`|`"mean"`|`"sum"`.'
                'See torch.nn docs for more details.'
            )
        )
    group = parser.add_argument_group('Criterion args')
    if criterion_name in {'L1Loss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss',
                          'MultiLabelMarginLoss', 'SmoothL1Loss',
                          'SoftMarginLoss', 'MultiLabelSoftMarginLoss'}:
        add_reduction(group)
    elif criterion_name == 'CrossEntropyLoss':
        add_reduction(group)
        group.add_argument(
            '--ignore_index', type=int, default=-100,
            help=(
                'Specifies a target value that is ignored and does not contribute to the input gradient.'
                ' See torch.nn docs for more details.'
            )
        )
        group.add_argument(
            '--label_smooting', type=float, default=0.0,
            help=(
                'A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where'
                ' 0.0 means no smoothing. See torch.nn docs for more details.'
            )
        )
    elif criterion_name == 'CTCLoss':
        group.add_argument('--blank', type=int, default=0, help='blank label.')
        add_reduction(group)
        group.add_argument(
            '--zero_infinity', type=str2bool, default=False,
            help='Whether to zero infinite losses and the associated gradients.'
        )
    elif criterion_name == 'NLLLoss':
        add_reduction(group)
        group.add_argument(
            '--ignore_index', type=int, default=-100,
            help=(
                'Specifies a target value that is ignored and does not contribute to the input gradient.'
                ' See torch.nn docs for more details.'
            )
        )
    elif criterion_name == 'PoissonNLLLoss':
        add_reduction(group)
        group.add_argument('--log_input', type=str2bool, default=True, help='See torch.nn docs for details.')
        group.add_argument(
            '--full', type=str2bool, default=False,
            help='Whether to compute full loss, i.e. to add the Stirling approximation term.'
        )
        group.add_argument(
            '--eps', type=float, default=1e-8,
            help='Small value to avoid log(0) when `log_input` is False.'
        )
    elif criterion_name == 'GaussianNLLLoss':
        add_reduction(group)
        group.add_argument(
            '--full', type=str2bool, default=False,
            help='Include the constant term in the loss calculation.'
        )
        group.add_argument('--eps', type=float, default=1e-6, help='Small value to clamp.')
    elif criterion_name == 'KLDivLoss':
        add_reduction(group)
        group.add_argument(
            '--log_target', type=str2bool, default=False,
            help='Specifies whether target is in the log space.'
        )
    elif criterion_name == 'MarginRankingLoss':
        add_reduction(group)
        group.add_argument('--margin', type=float, default=0.0)
    elif criterion_name == 'HingeEmbeddingLoss':
        add_reduction(group)
        group.add_argument('--margin', type=float, default=1.0)
    elif criterion_name == 'HuberLoss':
        add_reduction(group)
        group.add_argument(
            '--delta', type=float, default=1.0,
            help=(
                'Specifies the threshold at which to change between delta-scaled L1 and L2 loss.'
                ' Must be positive.'
            )
        )
    elif criterion_name == 'CosineEmbeddingLoss':
        add_reduction(group)
        group.add_argument(
            '--margin', type=float, default=0.0,
            help='Should be a number from -1 to 1. Values in 0 to 0.5 are suggested.'
        )
    elif criterion_name == 'MultiMarginLoss':
        add_reduction(group)
        group.add_argument('-p', type=int, default=1, help='Can be 1 or 2.')
        group.add_argument('--margin', type=float, default=1.0)
    elif criterion_name == 'TripletMarginLoss':
        add_reduction(group)
        group.add_argument('--margin', type=float, default=1.0)
        group.add_argument('--p', type=int, default=2, help='The norm degree for pairwise distance.')
        group.add_argument('--eps', type=float, default=1e-6, help='Small constant for numerical stability.')
        group.add_argument(
            '--swap', type=str2bool, default=False,
            help=(
                'Described in "Learning shallow convolutional feature descriptors with triplet losses"'
                ' by V. Balntas, E. Riba et al.'
            )
        )
    elif criterion_name == 'TripletMarginWithDistanceLoss':
        add_reduction(group)
        group.add_argument(
            '--margin', type=float, default=1.0,
            help=(
                'A nonnegative margin representing the minimum difference between the positive and'
                ' negative distances required for zero loss.'
            )
        )
        group.add_argument(
            '--swap', type=str2bool, default=False,
            help=(
                'Described in "Learning shallow convolutional feature descriptors with triplet losses"'
                ' by V. Balntas, E. Riba et al.'
            )
        )
    elif criterion_name in {'MSEKLmixed', 'L1KLmixed', 'MSEwithEntropy', 'L1withEntropy'}:
        default_reduction = 'batchmean' if criterion_name == 'MSEKLmixed' else 'mean'
        add_reduction(group, default=default_reduction)
        group.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for the MSE loss term.')
        group.add_argument('--beta', type=float, default=1.0, help='Scaling factor for the second loss term.')
    elif criterion_name in {'DirichletNLLLoss', 'JeffreysDivLoss'}:
        add_reduction(group)
        if criterion_name == 'JeffreysDivLoss':
            group.add_argument(
                '--log_target', type=str2bool, default=False,
                help='Specifies whether target is in the log space.'
            )
    else:
        raise RuntimeError(
            f'{criterion_name} not supported. Try one of the supported loss types.'
        )
    return parser


class Conv1dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True, weight_norm=True):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_channels, eps=1e-05, momentum=0.1,
                affine=True, track_running_stats=True
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.conv(input))
        except AttributeError:
            return self.conv(input)


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 batch_norm=True, weight_norm=True):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_features, eps=1e-05, momentum=0.1,
                affine=True, track_running_stats=True
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.linear(input))
        except AttributeError:
            return self.linear(input)


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)
        return reorg


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self, in_features, hidden_group_size, out_group_size,
        n_branches=1, n_layers=1, activation='ReLU', dropout_p=0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.branches = OrderedDict()
        self.nonlin = getattr(torch.nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)
        i = -1
        for i in range(self.n_layers-1):
            hook = getattr(self, f'branched_layer_{i+1}')(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f'branched_layer_{i+2}')(hook)
        return hook


class L1KLmixed(nn.Module):
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch', ''))
        self.KL = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, preds, targets):
        preds_log_prob = preds - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        MSE_loss = self.MSE(preds, targets)
        KL_loss = self.KL(preds_log_prob, target_log_prob)
        combined_loss = MSE_loss.mul(self.alpha) + KL_loss.mul(self.beta)
        return combined_loss.div(self.alpha+self.beta)


class BassetBranched(ptl.LightningModule):
    """
    A PyTorch Lightning module representing the BassetBranched model.

    Args:
        input_len (int): Fixed sequence length of inputs.
        conv1_channels (int): Number of channels for the first convolutional layer.
        conv1_kernel_size (int): Kernel size for the first convolutional layer.
        conv2_channels (int): Number of channels for the second convolutional layer.
        conv2_kernel_size (int): Kernel size for the second convolutional layer.
        conv3_channels (int): Number of channels for the third convolutional layer.
        conv3_kernel_size (int): Kernel size for the third convolutional layer.
        n_linear_layers (int): Number of linear (fully connected) layers.
        linear_channels (int): Number of channels in linear layers.
        linear_activation (str): Activation function for linear layers (default: 'ReLU').
        linear_dropout_p (float): Dropout probability for linear layers (default: 0.3).
        n_branched_layers (int): Number of branched linear layers.
        branched_channels (int): Number of output channels for branched layers.
        branched_activation (str): Activation function for branched layers (default: 'ReLU6').
        branched_dropout_p (float): Dropout probability for branched layers (default: 0.0).
        n_outputs (int): Number of output units.
        loss_criterion (str): Loss criterion class name (default: 'MSEKLmixed').
        criterion_reduction (str): Reduction type for loss criterion (default: 'mean').
        mse_scale (float): Scale factor for MSE loss component (default: 1.0).
        kl_scale (float): Scale factor for KL divergence loss component (default: 1.0).
        use_batch_norm (bool): Use batch normalization (default: True).
        use_weight_norm (bool): Use weight normalization (default: False).

    Methods:
        add_model_specific_args(parent_parser): Add model-specific arguments to the provided argparse ArgumentParser.
        add_conditional_args(parser, known_args): Add conditional model-specific arguments based on known arguments.
        process_args(grouped_args): Process grouped arguments and extract model-specific arguments.
        encode(x): Encode input data through the model's encoder layers.
        decode(x): Decode encoded data through the model's linear and branched layers.
        classify(x): Classify data using the output layer.
        forward(x): Forward pass through the entire model.

    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the provided argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): The parent ArgumentParser.

        Returns:
            argparse.ArgumentParser: The ArgumentParser with added model-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Model Module args')

        group.add_argument('--input_len', type=int, default=600)
        group.add_argument('--conv1_channels', type=int, default=300)
        group.add_argument('--conv1_kernel_size', type=int, default=19)
        group.add_argument('--conv2_channels', type=int, default=200)
        group.add_argument('--conv2_kernel_size', type=int, default=11)
        group.add_argument('--conv3_channels', type=int, default=200)
        group.add_argument('--conv3_kernel_size', type=int, default=7)
        group.add_argument('--n_linear_layers', type=int, default=2)
        group.add_argument('--linear_channels', type=int, default=1000)
        group.add_argument('--linear_activation', type=str, default='ReLU')
        group.add_argument('--linear_dropout_p', type=float, default=0.3)
        group.add_argument('--n_branched_layers', type=int, default=1)
        group.add_argument('--branched_channels', type=int, default=1000)
        group.add_argument('--branched_activation', type=str, default='ReLU')
        group.add_argument('--branched_dropout_p', type=float, default=0.3)
        group.add_argument('--n_outputs', type=int, default=280)
        group.add_argument('--use_batch_norm', type=str2bool, default=True)
        group.add_argument('--use_weight_norm', type=str2bool, default=False)
        group.add_argument('--loss_criterion', type=str, default='L1KLmixed')
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_criterion_specific_args(parser, known_args.loss_criterion)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Perform any required processessing of command line args required
        before passing to the class constructor.

        Args:
            grouped_args (Namespace): Namespace of known arguments with
            `'Model Module args'` key and conditionally added
            `'Criterion args'` key.

        Returns:
            Namespace: A modified namespace that can be passed to the
            associated class constructor.
        """
        model_args = grouped_args['Model Module args']
        model_args.loss_args = vars(grouped_args['Criterion args'])
        return model_args

    def __init__(
        self, input_len=600,
        conv1_channels=300, conv1_kernel_size=19,
        conv2_channels=200, conv2_kernel_size=11,
        conv3_channels=200, conv3_kernel_size=7,
        n_linear_layers=2, linear_channels=1000,
        linear_activation='ReLU', linear_dropout_p=0.3,
        n_branched_layers=1, branched_channels=250,
        branched_activation='ReLU6', branched_dropout_p=0.,
        n_outputs=280, use_batch_norm=True, loss_criterion='L1KLmixed',
        use_weight_norm=False, loss_args={}
    ):
        """
        Initialize the BassetBranched model.
        Args:
            conv1_channels (int): Number of channels for the first convolutional layer.
            conv1_kernel_size (int): Kernel size for the first convolutional layer.
            conv2_channels (int): Number of channels for the second convolutional layer.
            conv2_kernel_size (int): Kernel size for the second convolutional layer.
            conv3_channels (int): Number of channels for the third convolutional layer.
            conv3_kernel_size (int): Kernel size for the third convolutional layer.
            n_linear_layers (int): Number of linear (fully connected) layers.
            linear_channels (int): Number of channels in linear layers.
            linear_activation (str): Activation function for linear layers (default: 'ReLU').
            linear_dropout_p (float): Dropout probability for linear layers (default: 0.3).
            n_branched_layers (int): Number of branched linear layers.
            branched_channels (int): Number of output channels for branched layers.
            branched_activation (str): Activation function for branched layers (default: 'ReLU6').
            branched_dropout_p (float): Dropout probability for branched layers (default: 0.0).
            n_outputs (int): Number of output units.
            loss_criterion (str): Loss criterion class name (default: 'MSEKLmixed').
            loss_args (dict): Args to construct loss_criterion.
            use_batch_norm (bool): Use batch normalization (default: True).
            use_weight_norm (bool): Use weight normalization (default: False).
        """
        super().__init__()
        self.input_len = input_len
        self.conv1_channels = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        self.conv2_channels = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)
        self.conv3_channels = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        self.n_linear_layers = n_linear_layers
        self.linear_channels = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p = linear_dropout_p
        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p = branched_dropout_p
        self.n_outputs = n_outputs
        assert loss_criterion == 'L1KLmixed', 'the adapted code only supports L1KLmixed loss'
        self.loss_criterion = loss_criterion
        self.loss_args = loss_args
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.pad1 = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(
            4, self.conv1_channels, self.conv1_kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm
        )
        self.pad2 = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(
            self.conv1_channels, self.conv2_channels, self.conv2_kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm
        )
        self.pad3 = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(
            self.conv2_channels, self.conv3_channels, self.conv3_kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm
        )
        self.pad4 = nn.ConstantPad1d((1, 1), 0.)
        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        next_in_channels = self.conv3_channels * self.get_flatten_factor(self.input_len)
        for i in range(self.n_linear_layers):
            setattr(
                self, f'linear{i+1}',
                LinearNorm(
                    next_in_channels, self.linear_channels, bias=True,
                    batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm
                )
            )
            next_in_channels = self.linear_channels
        self.branched = BranchedLinear(
            next_in_channels, self.branched_channels, self.branched_channels,
            self.n_outputs, self.n_branched_layers, self.branched_activation,
            self.branched_dropout_p
        )

        self.output = GroupedLinear(self.branched_channels, 1, self.n_outputs)
        self.nonlin = getattr(torch.nn, self.linear_activation)()
        self.dropout = nn.Dropout(p=self.linear_dropout_p)
        self.criterion = L1KLmixed(**self.loss_args)

    def get_flatten_factor(self, input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0
        return (hook + 2) // 4

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        return hook

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(
                self.nonlin(
                    getattr(self, f'linear{i+1}')(hook)
                )
            )
        hook = self.branched(hook)
        return hook

    def classify(self, x):
        output = self.output(x)
        return output

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output = self.classify(decoded)
        return output


def unpack_model_artifact(
    artifact_path: Path, download_path: Path = config.MALINOIS_MODEL_DIR
) -> None:
    assert artifact_path.exists(), "Could not find file at expected path."
    assert tarfile.is_tarfile(artifact_path), f"Expected a tarfile at {artifact_path}. Not found."
    if download_path.exists():
        return
    shutil.unpack_archive(artifact_path, download_path.parent)
    print(f'archive unpacked in {download_path}', file=sys.stderr)


def load_malinois_model(
    model_path: Path = config.MALINOIS_MODEL_DIR,
    to_cuda: bool = False
):
    checkpoint = torch.load(model_path / 'torch_checkpoint.pt', weights_only=False)
    assert checkpoint['model_module'] == 'BassetBranched'
    model = BassetBranched(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    if to_cuda:
        assert torch.cuda.device_count() >= 1, 'CUDA not available :('
        model.cuda()
    return model


def dna2tensor(sequence_str, vocab_list=config.DNA_BASES):
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letterIdx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letterIdx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor


def reverse_complement_onehot(x):
    comp_alphabet = [config.DNA_COMPLEMENT_MAP[nt] for nt in config.DNA_BASES]
    permutation = [config.DNA_BASES.index(nt) for nt in comp_alphabet]
    return torch.flip(x[..., permutation, :], dims=[-1])


class FlankBuilder(nn.Module):
    def __init__(
        self, left_flank=None, right_flank=None, batch_dim=0, cat_axis=-1
    ):
        super().__init__()
        self.register_buffer('left_flank', left_flank.detach().clone())
        self.register_buffer('right_flank', right_flank.detach().clone())
        self.batch_dim = batch_dim
        self.cat_axis = cat_axis

    def add_flanks(self, my_sample):
        *batch_dims, channels, length = my_sample.shape
        pieces = []
        if self.left_flank is not None:
            pieces.append(self.left_flank.expand(*batch_dims, -1, -1))
        pieces.append(my_sample)
        if self.right_flank is not None:
            pieces.append(self.right_flank.expand(*batch_dims, -1, -1))
        return torch.cat(pieces, axis=self.cat_axis)

    def forward(self, my_sample):
        return self.add_flanks(my_sample)


class mpra_predictor(nn.Module):
    def __init__(
        self, model, pred_idx=0, ini_in_len=200,
        model_in_len=600, cat_axis=-1, dual_pred=False
    ):
        super().__init__()
        self.model = model
        self.pred_idx = pred_idx
        self.ini_in_len = ini_in_len
        self.model_in_len = model_in_len
        self.cat_axis = cat_axis
        self.dual_pred = dual_pred
        self.model.eval()
        self.register_flanks()

    def forward(self, in_tensor):
        if self.dual_pred:
            dual_tensor = reverse_complement_onehot(in_tensor)
            out_tensor = self.model(in_tensor)[:, self.pred_idx] + self.model(dual_tensor)[:, self.pred_idx]
            out_tensor = out_tensor / 2.0
        else:
            out_tensor = self.model(in_tensor)[:, self.pred_idx]
        return out_tensor

    def register_flanks(self):
        missing_len = self.model_in_len - self.ini_in_len
        left_idx = - missing_len // 2 + missing_len % 2
        right_idx = missing_len // 2 + missing_len % 2
        left_flank = dna2tensor(config.MPRA_FLANK_UPSTREAM[left_idx:]).unsqueeze(0)
        right_flank = dna2tensor(config.MPRA_FLANK_DOWNSTREAM[:right_idx]).unsqueeze(0)
        self.register_buffer('left_flank', left_flank)
        self.register_buffer('right_flank', right_flank)


def isg_contributions(
    sequences, predictor, num_steps=50, max_samples=20, eval_batch_size=1024,
    theta_factor=15, adaptive_sampling=False, DEVICE='cuda'
):
    """
    Calculate Integrated Sampled Gradients (ISG) contributions scores for sequences.

    Args:
        sequences (torch.Tensor): Input sequences.
        predictor (nn.Module): The predictor model.
        num_steps (int): Number of steps for in integrated linear path.
        max_samples (int): Maximum number of samples per step.
        eval_batch_size (int): Evaluation batch size for model queries.
        theta_factor (int): Theta factor to induce log probs.
        adaptive_sampling (bool): Whether to adapt sampling along the path.

    Returns:
        torch.Tensor: ISG contributions scores.
    """
    batch_size = eval_batch_size // (max_samples - 3)
    temp_dataset = TensorDataset(sequences)
    temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    slope_coefficients = [i / num_steps for i in range(1, num_steps + 1)]
    if adaptive_sampling:
        sneaky_exponent = np.log(max_samples - 3) / np.log(num_steps)
        sample_ns = np.flip((np.arange(0, num_steps)**sneaky_exponent).astype(int)).clip(min=2)
    else:
        sample_ns = [max_samples for i in range(0, num_steps + 1)]

    all_gradients = []
    for local_batch in temp_dataloader:
        target_thetas = (theta_factor * local_batch[0].to(DEVICE)).requires_grad_()
        line_gradients = []
        for i in range(0, num_steps):
            point_thetas = slope_coefficients[i] * target_thetas
            num_samples = sample_ns[i]
            point_distributions = nn.functional.softmax(point_thetas, dim=-2)
            nucleotide_probs = Categorical(torch.transpose(point_distributions, -2, -1))
            sampled_idxs = nucleotide_probs.sample((num_samples, ))
            sampled_nucleotides_T = nn.functional.one_hot(sampled_idxs, num_classes=4)
            sampled_nucleotides = torch.transpose(sampled_nucleotides_T, -2, -1)
            distribution_repeater = point_distributions.repeat(num_samples, *[1, 1, 1])
            sampled_nucleotides = sampled_nucleotides - distribution_repeater.detach() + distribution_repeater
            samples = sampled_nucleotides.flatten(0, 1)
            preds = predictor(samples)
            point_predictions = preds.unflatten(0, (num_samples, target_thetas.shape[0])).mean(dim=0)
            point_gradients = torch.autograd.grad(point_predictions.sum(), inputs=point_thetas, retain_graph=True)[0]
            line_gradients.append(point_gradients)
        gradients = torch.stack(line_gradients).mean(dim=0).detach()
        all_gradients.append(gradients)
    return theta_factor * torch.cat(all_gradients).cpu()


"""
Custom Functions
"""


def sequence_is_valid(seq: str) -> bool:
    if len(seq) != 200:
        return False
    valid_bases = {'A', 'C', 'G', 'T'}
    return all(base in valid_bases for base in seq)


def convert_sequences_to_malinois_input(
    tile_sequences: tp.Iterable[str],
) -> tuple[np.ndarray, torch.Tensor]:
    flank_len = 200  # number of bases to use from upstream / downstream flanks
    flank_builder = FlankBuilder(
        left_flank=dna2tensor(config.MPRA_FLANK_UPSTREAM[-flank_len:]).unsqueeze(0),
        right_flank=dna2tensor(config.MPRA_FLANK_DOWNSTREAM[:flank_len]).unsqueeze(0),
    ).to(DEVICE)
    tile_sequences = list(tile_sequences)  # ensure reusability
    valid_mask = np.array([sequence_is_valid(seq) for seq in tile_sequences])
    valid_sequences = [seq for seq, is_valid in zip(tile_sequences, valid_mask) if is_valid]
    if not valid_sequences:
        raise ValueError("No valid sequences provided.")
    tile_tensors = torch.stack([dna2tensor(seq) for seq in valid_sequences]).to(DEVICE)
    return valid_mask, flank_builder(tile_tensors)


def compute_malinois_model_predictions(
    malinois_model: 'BassetBranched',
    tile_sequences: tp.Iterable[str],
    batch_size: int = 500,
    use_tqdm: bool = False,
) -> np.ndarray:
    valid_mask, tile_tensors = convert_sequences_to_malinois_input(tile_sequences)
    predictions = np.full((valid_mask.size, 3), np.nan, dtype=np.float32)
    if tile_tensors.shape[0] == 0:
        return predictions
    malinois_model = malinois_model.eval().to(DEVICE)
    predicted_batches = []
    indices = range(0, tile_tensors.shape[0], batch_size)
    if use_tqdm:
        indices = tqdm(indices, desc="Predicting")
    for start in indices:
        end = start + batch_size
        batch = tile_tensors[start: end]
        with torch.no_grad():
            pred = malinois_model(batch)
        predicted_batches.append(pred.cpu().numpy())
    all_preds = np.concatenate(predicted_batches, axis=0)
    transformed_preds = np.log2(np.exp(all_preds) + 1)
    predictions[valid_mask] = transformed_preds
    return predictions


def compute_model_contribution_scores(
    model: nn.Module,
    tile_sequences: tp.Iterable[str],
    pred_idx: int = 0,
    batch_size: int = 500,
    use_tqdm: bool = True,
) -> np.ndarray:
    valid_mask, tile_tensors = convert_sequences_to_malinois_input(tile_sequences)
    contribution_scores = np.full((valid_mask.size, 4, 200), np.nan, dtype=np.float32)
    if tile_tensors.shape[0] == 0:
        return contribution_scores
    predictor = mpra_predictor(model, pred_idx=pred_idx).to(DEVICE).eval()
    all_scores = []
    indices = range(0, tile_tensors.shape[0], batch_size)
    if use_tqdm:
        indices = tqdm(indices, desc="Computing contribution scores")
    for start in indices:
        end = start + batch_size
        batch = tile_tensors[start:end]
        contrib = isg_contributions(batch, predictor)[:, :, 200:400]  # center region
        all_scores.append(contrib.cpu())
    all_scores = torch.cat(all_scores, dim=0).numpy()
    contribution_scores[valid_mask] = all_scores
    return contribution_scores
