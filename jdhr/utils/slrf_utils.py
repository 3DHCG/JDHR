#import torch
import jittor as jt
import numpy as np
from jittor import init
from jittor import nn
from jdhr.utils.net_utils import GradientModule

# Grouped convolution for SLRF


def grouped_mlp(I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU(), type='fused'):
    if type == 'fused':
        return FusedGroupedMLP(I, N, W, D, Z, actvn)  # fast, worse, # ? why: problem with grad magnitude
    elif type == 'gconv':
        return GConvGroupedMLP(I, N, W, D, Z, actvn)  # slow, better


class FusedGroupedMLP(GradientModule):
    # I: input dim
    # N: group count
    # W: network width
    # D: network depth
    # Z: output dim
    # actvn: network activation

    # Fisrt layer: (B, N * I, S) -> (B * N, I, S) -> (B * N, S, I)
    # Weight + bias: (N, I, W) + (N, W) -> pad to (B, N, I, W) -> (B * N, I, W)
    # Result: (B * N, S, W) + (B * N, S, W)
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        super(FusedGroupedMLP, self).__init__()
        self.N = N
        self.I = I
        self.Z = Z
        self.W = W
        self.D = D
        self.actvn = actvn

        self.Is = \
            [I] +\
            [W for _ in range(D - 2)] +\
            [W]\

        self.Zs = \
            [W] +\
            [W for _ in range(D - 2)] +\
            [Z]\

        self.weights = nn.ParameterList(
            [nn.Parameter(jt.empty(N, I, W))] +
            [nn.Parameter(jt.empty(N, W, W)) for _ in range(D - 2)] +
            [nn.Parameter(jt.empty(N, W, Z))]
        )

        self.biases = nn.ParameterList(
            [nn.Parameter(jt.empty(N, W))] +
            [nn.Parameter(jt.empty(N, W)) for _ in range(D - 2)] +
            [nn.Parameter(jt.empty(N, Z))]
        )

        for i, w in enumerate(self.weights):  # list stores reference
            ksqrt = np.sqrt(1 / self.Is[i])
            init.uniform_(w, -ksqrt, ksqrt)
        for i, b in enumerate(self.biases):
            ksqrt = np.sqrt(1 / self.Is[i])
            init.uniform_(b, -ksqrt, ksqrt)

    def forward(self, x: jt.Var):
        B, N, S, I = x.shape
        x = x.view(B * self.N, S, self.I)

        for i in range(self.D):
            I = self.Is[i]
            Z = self.Zs[i]
            w = self.weights[i]  # N, W, W
            b = self.biases[i]  # N, W

            w = w[None].expand(B, -1, -1, -1).reshape(B * self.N, I, Z)
            b = b[None, :, None].expand(B, -1, -1, -1).reshape(B * self.N, -1, Z)
            x = jt.baddbmm(b, x, w)  # will this just take mean along batch dimension?
            if i < self.D - 1:
                x = self.actvn(x)

        x = x.view(B, self.N, S, self.Z)  # ditching gconv

        return x


class GConvGroupedMLP(GradientModule):
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        # I: input dim
        # N: group count
        # W: network width
        # D: network depth
        # Z: output dim
        # actvn: network activation
        super(GConvGroupedMLP, self).__init__()
        self.mlp = nn.ModuleList(
            [nn.Conv1d(N * I, N * W, 1, groups=N), actvn] +
            [f for f in [nn.Conv1d(N * W, N * W, 1, groups=N), actvn] for _ in range(D - 2)] +
            [nn.Conv1d(N * W, N * Z, 1, groups=N)]
        )
        self.mlp = nn.Sequential(*self.mlp)

    def execute(self, x: jt.Var):
        # x: B, N, S, C
        B, N, S, I = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, N * I, S)
        x = self.mlp(x)
        x = x.reshape(B, N, -1, S)
        x = x.permute(0, 1, 3, 2)
        return x


class cVAE(GradientModule):  # group cVAE with grouped convolution
    def __init__(self,
                 group_cnt: int,
                 latent_dim: int,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,

                 encode_w: int,
                 encode_d: int,

                 decode_w: int,
                 decode_d: int,
                 ):
        super(cVAE, self).__init__()

        self.N = group_cnt
        self.L = latent_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # input: embedded time concatenated with multiplied pose
        # output: mu and log_var
        I = in_dim + cond_dim
        N, W, D, Z = group_cnt, encode_w, encode_d, latent_dim * 2
        self.encoder = grouped_mlp(I, N, W, D, Z)

        # input: reparameterized latent variable
        # output: high-dim embedding + 3D residual node trans
        I = latent_dim + cond_dim
        N, W, D, Z = group_cnt, decode_w, decode_d, out_dim
        self.decoder = grouped_mlp(I, N, W, D, Z)

    def encode(self, x: jt.Var):
        # x: B, N, S, I
        mu, log_var = self.encoder(x).split([self.L, self.L], dim=-1)
        return mu, log_var

    def decode(self, z: jt.Var):
        # z: B, N, S, 8
        out = self.decoder(z)
        return out

    def reparameterize(self, mu: jt.Var, log_var: jt.Var):
        std = jt.exp(0.5 * log_var)
        eps = jt.randn_like(std)
        return eps * std + mu

    def execute(self, x: jt.Var, c: jt.Var):
        input_ndim = x.ndim
        if input_ndim == 3 and self.N == 1:
            x = x[:, None]
            c = c[:, None]
        elif input_ndim == 2 and self.N == 1:
            x = x[:, None, None]
            c = c[:, None, None]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        # x: B, N, S, C, where C is N * in_dim
        # where in_dim should be embedded time concatenated with multiplied pose
        mu, log_var = self.encode(jt.cat([x, c], dim=-1))  # this second is a lot slower than decode, why?
        z = self.reparameterize(mu, log_var)
        out = self.decode(jt.cat([z, c], dim=-1))
        # out: B, N, S, out_dim(1)
        # mu: B, N, S, 8, log_var: B, N, S, 8, z: B, N, S, 8

        if input_ndim == 3 and self.N == 1:
            out = out[:, 0]
            mu = mu[:, 0]
            log_var = log_var[:, 0]
            z = z[:, 0]
        elif input_ndim == 2 and self.N == 1:
            out = out[:, 0, 0]
            mu = mu[:, 0, 0]
            log_var = log_var[:, 0, 0]
            z = z[:, 0, 0]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        return out, mu, log_var, z

# Resnet Blocks


class ResnetBlock(nn.Module):
    """
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, kernel_size, size_out=None, size_h=None):
        super(ResnetBlock, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        padding = kernel_size // 2
        self.conv_0 = nn.Conv2d(size_in, size_h, kernel_size=kernel_size, padding=padding)
        self.conv_1 = nn.Conv2d(size_h, size_out, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, kernel_size=kernel_size, padding=padding)

    def execute(self, x):
        net = self.conv_0(self.activation(x))
        dx = self.conv_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
