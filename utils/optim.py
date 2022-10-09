"""
AllenNLP just uses
[PyTorch optimizers](https://pytorch.org/docs/master/optim.html),
with a thin wrapper to allow registering them and instantiating them `from_params`.
The available optimizers are
* [adadelta](https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta)
* [adagrad](https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad)
* [adam](https://pytorch.org/docs/master/optim.html#torch.optim.Adam)
* [adamw](https://pytorch.org/docs/master/optim.html#torch.optim.AdamW)
* [huggingface_adamw](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw-pytorch)
* [huggingface_adafactor](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch)
* [sparse_adam](https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam)
* [sgd](https://pytorch.org/docs/master/optim.html#torch.optim.SGD)
* [rmsprop](https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop)
* [adamax](https://pytorch.org/docs/master/optim.html#torch.optim.Adamax)
* [averaged_sgd](https://pytorch.org/docs/master/optim.html#torch.optim.ASGD)
"""  # noqa: E501
import copy
import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import transformers

from .optim_common import Params

logger = logging.getLogger(__name__)

ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]


class Optimizer(torch.optim.Optimizer):
    """
    This class just allows us to implement `Registrable` for Pytorch Optimizers.  We do something a
    little bit different with `Optimizers`, because they are implemented as classes in PyTorch, and
    we want to use those classes.  To make things easy, we just inherit from those classes, using
    multiple inheritance to also inherit from `Optimizer`.  The only reason we do this is to make
    type inference on parameters possible, so we can construct these objects using our configuration
    framework. If you are writing your own script, you can safely ignore these classes and just use
    the `torch.optim` classes directly.
    If you are implementing one of these classes, the `model_parameters` and `parameter_groups`
    arguments to `__init__` are important, and should always be present.  The trainer will pass
    the trainable parameters in the model to the optimizer using the name `model_parameters`, so if
    you use a different name, your code will crash.  Nothing will technically crash if you use a
    name other than `parameter_groups` for your second argument, it will just be annoyingly
    inconsistent.
    Most subclasses of `Optimizer` take both a `model_parameters` and a `parameter_groups`
    constructor argument. The `model_parameters` argument does not get an entry in a typical
    AllenNLP configuration file, but the `parameter_groups` argument does (if you want a non-default
    value).  See the documentation for the `make_parameter_groups` function for more information on
    how the `parameter_groups` argument should be specified.
    """

    default_implementation = "adam"

    @staticmethod
    def default(model_parameters: List) -> "Optimizer":
        return Optimizer.from_params(model_parameters=model_parameters,
                                     params=Params({}))


class DenseSparseAdam(torch.optim.Optimizer):
    """
    NOTE: This class has been copied verbatim from the separate Dense and
    Sparse versions of Adam in Pytorch.
    Implements Adam algorithm with dense & sparse gradients.
    It has been proposed in Adam: A Method for Stochastic Optimization.
    Registered as an `Optimizer` with name "dense_sparse_adam".
    # Parameters
    params : `iterable`
        iterable of parameters to optimize or dicts defining parameter groups
    lr : `float`, optional (default = `1e-3`)
        The learning rate.
    betas : `Tuple[float, float]`, optional (default = `(0.9, 0.999)`)
        coefficients used for computing running averages of gradient
        and its square.
    eps : `float`, optional, (default = `1e-8`)
        A term added to the denominator to improve numerical stability.
    """
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        # Parameters
        closure : `callable`, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if grad.is_sparse:
                    grad = grad.coalesce(
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(
                        old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(
                        grad)._values()
                    exp_avg_sq_update_values = (grad_values.pow(2).sub_(
                        old_exp_avg_sq_values).mul_(1 - beta2))
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group["eps"])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1**state["step"]
                    bias_correction2 = 1 - beta2**state["step"]
                    step_size = group["lr"] * math.sqrt(
                        bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))

                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad,
                                                    grad,
                                                    value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    bias_correction1 = 1 - beta1**state["step"]
                    bias_correction2 = 1 - beta2**state["step"]
                    step_size = group["lr"] * math.sqrt(
                        bias_correction2) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss