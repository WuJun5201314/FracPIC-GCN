"""Fractional graph projection layer (FGPL) with Frobenius-norm clipping.

The fractional weight-gradient construction uses the following terms:

    w_main = x * (|wf| + eps) ** (1 - alpha) / Gamma(2 - alpha)
    w_partial = ((x @ wf.T) - x * wf + b[0])
                * sign(wf) * (|wf| + eps) ** (-alpha)
                / Gamma(1 - alpha)
    w_fractional = (w_main + clip_L2(w_partial, c)).T

Only the implementation around those equations is made device-safe,
batch-safe, mixed-precision-safe, and suitable for use in a GNN.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter


def _clip_matrix_l2_norm(matrix: Tensor, max_norm: Tensor) -> Tensor:
    """Clip the global L2/Frobenius norm without a CPU synchronization."""
    norm = torch.linalg.vector_norm(matrix)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return matrix * scale


class _FractionalOrderMatrixDifferential(torch.autograd.Function):
    """Custom backward implementing the original fractional-order derivative."""

    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        weight_t: Tensor,
        bias: Optional[Tensor],
        alpha: float,
        c: float,
    ) -> Tensor:
        # weight_t follows the original solver layout: [in_features, out_features].
        has_bias = bias is not None
        saved_bias = (
            bias
            if has_bias
            else inputs.new_zeros(weight_t.shape[-1])
        )
        alpha_tensor = inputs.new_tensor(alpha)
        c_tensor = inputs.new_tensor(c)

        ctx.has_bias = has_bias
        ctx.input_shape = inputs.shape
        ctx.save_for_backward(
            inputs, weight_t, saved_bias, alpha_tensor, c_tensor
        )

        output = torch.matmul(inputs, weight_t)
        if has_bias:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inputs, weight_t, bias, alpha, c = ctx.saved_tensors

        # GNN features are commonly [N, F], while sampled/batched graphs may
        # use [..., F]. Flattening only the leading axes preserves the original
        # 2-D equations and extends them to arbitrary leading batch dimensions.
        inputs_2d = inputs.reshape(-1, inputs.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        input_fractional, weight_fractional = (
            _FractionalOrderMatrixDifferential.fractional_linear_terms(
                inputs_2d, weight_t, bias, alpha, c
            )
        )

        grad_input = torch.matmul(
            grad_output_2d, input_fractional
        ).reshape(ctx.input_shape)
        grad_weight_t = torch.matmul(weight_fractional, grad_output_2d)
        grad_bias = (
            grad_output_2d.sum(dim=0) if ctx.has_bias else None
        )

        return grad_input, grad_weight_t, grad_bias, None, None

    @staticmethod
    def fractional_linear_terms(
        inputs: Tensor,
        weight_t: Tensor,
        bias: Tensor,
        alpha: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Compute the fractional branch in FP32 under FP16/BF16 autocast.
        # This does not change the equations; it prevents eps underflow and
        # unstable powers near zero.
        original_dtype = inputs.dtype
        work_dtype = (
            torch.float32
            if original_dtype in (torch.float16, torch.bfloat16)
            else original_dtype
        )
        x = inputs.to(work_dtype)
        w = weight_t.to(work_dtype)
        b = bias.to(work_dtype)
        a = alpha.to(work_dtype)
        max_norm = c.to(work_dtype)

        wf = w[:, 0].reshape(1, -1)
        abs_wf_eps = torch.abs(wf) + 1e-8

        # Same gamma denominators as the original scipy.special.gamma calls.
        gamma_2_minus_alpha = torch.exp(torch.lgamma(2.0 - a))
        gamma_1_minus_alpha = torch.exp(torch.lgamma(1.0 - a))

        # Core fractional-order main term -- unchanged.
        w_main = (
            x
            * abs_wf_eps.pow(1.0 - a)
            / gamma_2_minus_alpha
        )

        # Core fractional-order partial term -- unchanged.
        w_partial = (
            (
                torch.matmul(x, wf.transpose(0, 1)).expand_as(x)
                - x * wf
                + b[0]
            )
            * torch.sgn(wf)
            * abs_wf_eps.pow(-a)
            / gamma_1_minus_alpha
        )

        clipped_partial = _clip_matrix_l2_norm(w_partial, max_norm)
        weight_fractional = (w_main + clipped_partial).transpose(0, 1)

        return (
            w.transpose(0, 1).to(original_dtype),
            weight_fractional.to(original_dtype),
        )


class FGPL(nn.Module):
    """Standard linear projection with a fractional weight gradient for graph features.

    Input shape:
        ``[..., in_features]``
    Output shape:
        ``[..., out_features]``

    The stored weight uses PyTorch's standard ``[out_features, in_features]``
    layout, so the module is easy to place where an ``nn.Linear`` is used.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.9,
        c: float = 1.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if c <= 0.0:
            raise ValueError("c must be positive")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = float(alpha)
        self.c = float(c)

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last input dimension {self.in_features}, "
                f"but received {inputs.shape[-1]}"
            )
        return _FractionalOrderMatrixDifferential.apply(
            inputs, self.weight.transpose(0, 1), self.bias, self.alpha, self.c
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"alpha={self.alpha}, c={self.c}, "
            f"bias={self.bias is not None}"
        )


class FractionalGraphConv(nn.Module):
    """Minimal dependency-free mean-aggregation GNN layer using ``FGPL``.

    ``edge_index[0]`` contains source nodes and ``edge_index[1]`` destination
    nodes. Self features are included by default. For PyG/DGL projects, the
    ``FGPL`` above can instead be placed directly after their aggregator.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.9,
        c: float = 1.0,
        bias: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.add_self_loops = add_self_loops
        self.projection = FGPL(
            in_features,
            out_features,
            alpha=alpha,
            c=c,
            bias=bias,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("FractionalGraphConv expects x shaped [N, F]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")

        source, destination = edge_index
        num_nodes = x.shape[0]

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, destination, x[source])

        degree = x.new_zeros(num_nodes)
        degree.index_add_(
            0, destination, x.new_ones(destination.numel())
        )

        if self.add_self_loops:
            aggregated = aggregated + x
            degree = degree + 1.0

        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        return self.projection(aggregated)


__all__ = ["FGPL", "FractionalGraphConv"]
