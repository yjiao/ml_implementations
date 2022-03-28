from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import numpy as np
import logging
import os
from tqdm import tqdm
from typing import Callable, List, Optional


class Resnet(nn.Module):
    def __init__(
        self,
        n: int,
        num_classes: int = 10,
        return_logits: bool = True,
        kernel_size: int = 3,
        dim_in: int = 32,
        channels_out: int = 64,
        channels_in: int = 3,
    ):
        """Constructs the Resnet Cifar-10 model.

        Note we skip the first non-residual 3x3 conv for simplicity.

        n: as defined in paper
        num_classes: number of classes to output logits.
        return_logits: whether to return logits.
        kernel_size: size of kernel for conv ops.
        dim_in: input (image) dim. Ex. 32 = 32x32 images.
        channels_out: number of channels in output. Ex. 64 = 64 channels.
        """
        super().__init__()

        assert dim_in % 4 == 0, f"input dimension must be divisible by 4. Got {dim_in}."
        assert (
            channels_out % 4 == 0
        ), f"output channels must be divisible by 4. Got {channels_out}."
        self.return_logits = return_logits

        dims = [dim_in, dim_in // 2, dim_in // 4]
        channels = [channels_out // 4, channels_out // 2, channels_out]

        self.conv_ops = nn.Sequential(
            nn.Conv2d(
                channels_in,
                channels[0],
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            *[
                ForwardResidualBlock(
                    channels[0],
                    channels[0],
                    dim_in=dims[0],
                    dim_out=dims[0],
                    kernel_size=kernel_size,
                )
                for _ in range(n)
            ],
            ForwardResidualBlock(
                channels[0],
                channels[1],
                dim_in=dims[0],
                dim_out=dims[1],
                kernel_size=kernel_size,
            ),
            *[
                ForwardResidualBlock(
                    channels[1],
                    channels[1],
                    dim_in=dims[1],
                    dim_out=dims[1],
                    kernel_size=kernel_size,
                )
                for _ in range(n - 1)
            ],
            ForwardResidualBlock(
                channels[1],
                channels[2],
                dim_in=dims[1],
                dim_out=dims[2],
                kernel_size=kernel_size,
            ),
            *[
                ForwardResidualBlock(
                    channels[2],
                    channels[2],
                    dim_in=dims[2],
                    dim_out=dims[2],
                    kernel_size=kernel_size,
                )
                for _ in range(n - 1)
            ],
        )

        if return_logits:
            self.fc = nn.Linear(dims[2], num_classes)

    def forward(self, X):
        # (batch, channels, row, col)
        output = self.conv_ops(X)
        output = output.mean((2, 3))
        if self.return_logits:
            output = self.fc(output)
        return output


class ReverseResent(nn.Module):
    """Reverses the operations of a resnet.

    Note: We are not inverting the resnet ops."""

    def __init__(
        self,
        n: int,
        dim_out: int = 32,
        kernel_size: int = 3,
        channels_in: int = 64,
        channels_out: int = 3,
    ):
        super().__init__()
        #        assert channels_in % 4 == 0, "channels_in must be divisible by 4."

        channels = [channels_in, channels_in // 2, channels_in // 4]
        dims = [dim_out // 4, dim_out // 2, dim_out]

        self.conv_ops = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels[0],
                out_channels=channels[0],
                bias=False,
                kernel_size=dims[0],
            ),
            *[
                ReverseResidualBlock(
                    channels[0],
                    channels[0],
                    dim_in=dims[0],
                    dim_out=dims[0],
                    kernel_size=kernel_size,
                )
                for _ in range(n)
            ],
            ReverseResidualBlock(
                channels[0],
                channels[1],
                dim_in=dims[0],
                dim_out=dims[1],
                kernel_size=kernel_size,
            ),
            *[
                ReverseResidualBlock(
                    channels[1],
                    channels[1],
                    dim_in=dims[1],
                    dim_out=dims[1],
                    kernel_size=kernel_size,
                )
                for _ in range(n - 1)
            ],
            ReverseResidualBlock(
                channels[1],
                channels[2],
                dim_in=dims[1],
                dim_out=dims[2],
                kernel_size=kernel_size,
            ),
            *[
                ReverseResidualBlock(
                    channels[2],
                    channels[2],
                    dim_in=dims[2],
                    dim_out=dims[2],
                    kernel_size=kernel_size,
                )
                for _ in range(n - 1)
            ],
            nn.ConvTranspose2d(
                in_channels=channels[2],
                out_channels=channels_out,
                bias=False,
                kernel_size=1,
            ),
        )

    def forward(self, X):
        return self.conv_ops(X)


class ForwardResidualBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
    ):
        """Create residual block for forward resnet.

        Forward block is defined as described in He et al 2016 for Cifar-10
        dataset, implementing 0-padding shortcut operations.

        note that downsampling is down at the beginning.
        """
        super().__init__()
        downsample = dim_out < dim_in
        downsample_ratio = dim_in // dim_out
        if downsample:
            if dim_in % dim_out != 0:
                raise Exception(
                    f"dim_in must be a multiple of dim_out, got {dim_in} -> {dim_out}."
                )
            if channels_out < channels_in:
                raise Exception(
                    f"Expect resnet to have non-decreasing channel sizes, got {channels_in} -> {channels_out}."
                )
        else:
            assert (
                channels_in == channels_out
            ), "Without up- or down-sampling, channels_in must be equal to channels_out."

        if downsample:
            self.conv1 = nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=downsample_ratio,
                stride=downsample_ratio,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(channels_out)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.relu2 = nn.ReLU()

        if downsample:
            resize = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
            pad = torch.nn.ConstantPad3d(
                (0, 0, 0, 0, 0, channels_out - channels_in), value=0
            )
        else:
            resize = nn.Identity()
            pad = nn.Identity()

        self.shortcut = nn.Sequential(resize, pad)

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)

        return self.relu2(fx + self.shortcut(x))


class ReverseResidualBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
    ):
        """Create residual block for reverse resnet.

        Reverse block reverses the order of operations for the forward blocks,
        but are NOT inverses of these operations.

        note that downsampling is down at the beginning, upsampling is done at
        the end.
        """
        super().__init__()
        upsample = dim_in < dim_out
        upsample_ratio = dim_out // dim_in
        if upsample:
            if dim_out % dim_in != 0:
                raise Exception(
                    f"dim_out must be a multiple of dim_in, got {dim_in} -> {dim_out}."
                )
            if channels_out > channels_in:
                raise Exception(
                    f"Expect reverse resnet to have non-increasing channel sizes, got {channels_in} -> {channels_out}."
                )

        if not upsample:
            assert (
                channels_in == channels_out
            ), "Without up- or down-sampling, channels_in must be equal to channels_out."

        self.conv1 = nn.Conv2d(
            channels_in,
            channels_in,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu1 = nn.ReLU()

        if upsample:
            self.conv2 = nn.ConvTranspose2d(
                channels_in,
                channels_out,
                bias=False,
                stride=upsample_ratio,
                kernel_size=upsample_ratio,
            )
        else:
            self.conv2 = nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.relu2 = nn.ReLU()

        if upsample:
            resize = nn.Upsample(scale_factor=upsample_ratio, mode="nearest")
        else:
            resize = nn.Identity()

        if channels_in > channels_out:
            pad = TruncateChannels(channels_out)
        else:
            pad = nn.Identity()
        self.shortcut = nn.Sequential(resize, pad)

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)

        _, c, h, w = fx.shape

        return self.relu2(fx + self.shortcut(x))


class TruncateChannels(nn.Module):
    def __init__(self, channels_out: int):
        """Constructs the "transpose" of Resnet Cifar-10 model."""
        super().__init__()
        self.channels_out = channels_out

    def forward(self, x):
        return x[:, : self.channels_out, :, :]


class VAE(nn.Module):
    def __init__(
        self,
        resnet_n: int = 3,
        d_model: int = 64,
        d_z: int = 32,
        input_channels: int = 3,
        input_dim: int = 32,
    ):
        super().__init__()
        self.encoder = Resnet(
            n=resnet_n,
            channels_out=d_model,
            return_logits=False,
            channels_in=input_channels,
        )
        self.proj_z_u = nn.Linear(d_model, d_z)
        self.proj_z_s = nn.Linear(d_model, d_z)
        self.decoder = ReverseResent(
            n=resnet_n,
            channels_in=d_z,
            channels_out=input_channels,
            dim_out=input_dim,
        )

    def forward(self, x):
        enc = self.encoder(x)
        z_u = self.proj_z_u(enc)
        z_s = self.proj_z_s(enc)
        if self.training:
            e = torch.randn(z_u.shape).to(z_u.device)
            z = z_u + e * z_s
        else:
            z = z_u
        dec = self.decoder(z.view(*z.shape, 1, 1))
        return z_u, z_s, dec


def kl_divergence(z_u, z_s):
    # closed form KL divergence for 2 Gaussians copied from: https://leenashekhar.github.io/2019-01-30-KL-Divergence/
    # note we assume that p ~ N(0, 1) for KL[q || p]
    kl_divergence = torch.sum(
        -0.5 * (1.0 + torch.log(z_s * z_s) - z_u * z_u - z_s * z_s)
    )
    assert kl_divergence >= 0, f"{kl_divergence} should be >= 0! mu: {z_u}, sd: {z_s}"
    return kl_divergence


def reconstruction_loss(img_in, img_out, binary: bool = False):
    # roughly -logP(X|Z) where both ~Gaussian
    if binary:
        return F.binary_cross_entropy_with_logits(img_out, img_in, reduction="sum")
    return F.mse_loss(img_in, img_out, reduction="sum")


def elbo_loss(z_u, z_s, img_in, img_out, binary: bool = False):
    # elbo = log P(X|Z) - KL_divergence
    # want to maximize elbo -> minimize -elbo
    # -elbo = -log P(X|Z) + KL_divergence
    neg_logp = reconstruction_loss(img_in, img_out, binary=binary)
    kl = kl_divergence(z_u, z_s)
    loss = neg_logp + kl
    return loss


def train_epoch(
    model,
    loss_histories: dict,
    expt_key: str,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    save_every: int = float("inf"),  # number of examples between saves
    eval_every: int = 1000,  # number of examples between evals
    save_state_only: bool = False,
    model_dir: Optional[str] = None,
    lr: float = 1e-5,
    binary: bool = False,  # whether reconstruction loss is for binary images
):
    """Minimal train function that saves checkpoints and loss histories."""
    if expt_key not in loss_histories:
        loss_histories[expt_key] = dict()
        loss_histories[expt_key]["train_loss"] = []
        loss_histories[expt_key]["test_loss"] = dict()
        loss_histories[expt_key]["test_loss"]["kl"] = []
        loss_histories[expt_key]["test_loss"]["mse"] = []
        loss_histories[expt_key]["test_loss"]["loss"] = []
    else:
        logging.info("appending to", expt_key)
    i = 0

    model = model.to(device)

    if model_dir is None:
        model_dir = f"./saved_models/{expt_key}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logging.info(f"saving models to {model_dir}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    batch_sz = train_loader.batch_size
    if not loss_histories[expt_key]["train_loss"]:
        ex_seen = 0
    else:
        ex_seen = loss_histories[expt_key]["train_loss"][-1][0]

    save_threshold = ex_seen + max(save_every, batch_sz)
    save_threshold_step = save_threshold
    eval_threshold = ex_seen + max(eval_every, batch_sz)
    eval_threshold_step = eval_threshold

    def log_and_print(
        loss_histories, expt_key, ex_seen, train_loss, test_loss, test_kl, test_mse
    ):
        loss_histories[expt_key]["test_loss"]["kl"].append((ex_seen, test_kl))
        loss_histories[expt_key]["test_loss"]["mse"].append((ex_seen, test_mse))
        loss_histories[expt_key]["test_loss"]["loss"].append((ex_seen, test_loss))
        print(ex_seen)
        print(f"binary image: {binary}")
        print(f'{"training loss":>30} {train_loss:>20.4f}')
        print(f'{"test loss":>30} {test_loss:>20.4f}')
        print(f'{"test kl":>30} {test_kl:>20.4f}')
        print(f'{"test reconst loss":>30} {test_mse:>20.4f}')

    for i, (input_img, _) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        position=0,
        leave=True,
    ):
        hist_entry = dict()

        model.train()
        input_img = input_img.to(device)
        z_u, z_s, img_out = model(input_img)
        loss = elbo_loss(z_u, z_s, input_img, img_out, binary=binary)
        loss.requires_grad_()

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        opt.step()
        train_loss = loss.item()
        loss_histories[expt_key]["train_loss"].append((ex_seen, train_loss))

        ex_seen += batch_sz

        if ex_seen >= save_threshold:
            if save_state_only:
                torch.save(model.state_dict(), f"{model_dir}/state_{i}")
            else:
                torch.save(model, f"{model_dir}/model_{i}")
            save_threshold += save_threshold_step

        if ex_seen >= eval_threshold:
            eval_threshold += eval_threshold_step
            test_kl, test_mse, test_loss = eval_model(
                model, test_loader, device, binary=binary
            )
            log_and_print(
                loss_histories,
                expt_key,
                ex_seen,
                train_loss,
                test_loss,
                test_kl,
                test_mse,
            )

    test_kl, test_mse, test_loss = eval_model(model, test_loader, device, binary=binary)
    log_and_print(
        loss_histories, expt_key, ex_seen, train_loss, test_loss, test_kl, test_mse
    )
    if save_state_only:
        torch.save(model.state_dict(), f"{model_dir}/state_final")
    else:
        torch.save(model, f"{model_dir}/model_final")


@torch.no_grad()
def eval_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    binary: bool = False,
) -> float:
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_mse = 0.0
    n_batches = 0
    for i, (img_in, _) in enumerate(test_loader):
        img_in = img_in.to(device)
        z_u, z_s, img_out = model(img_in)
        kl = kl_divergence(z_u, z_s)
        mse = reconstruction_loss(img_in, img_out, binary=binary)
        loss = kl + mse

        total_kl += kl.item()
        total_mse += mse.item()
        total_loss += loss.item()

        n_batches += 1
    n_batches = float(n_batches)
    return total_kl / n_batches, total_mse / n_batches, total_loss / n_batches


def init_xavier(layer, verbose: bool = False):
    if hasattr(layer, "weight"):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
        else:
            if verbose:
                print("skipping initialization of layer:", layer)


def get_max_grad(model, verbose=False):
    max_grad = 0
    name = None
    for name, params in model.named_parameters():
        grad = torch.max(params.grad).item()
        if grad > max_grad:
            max_grad = grad
            name = name
        if verbose:
            print(f"{name:>50}", grad)
    return max_grad, name


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def print_numel(model, name):
    print(f"{name:>30}: {count_params(model):,}")
