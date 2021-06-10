import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class CenteredClipResult:
    result: torch.Tensor
    n_clipped: torch.Tensor
    step_norm: torch.Tensor


def centered_clip(input_tensors: torch.Tensor, weights: torch.Tensor,
                  tau: float=1.0, n_iters: int=20) -> CenteredClipResult:
    result_shape = input_tensors.shape[1:]
    input_tensors = input_tensors.flatten(start_dim=1)

    result = input_tensors.median(dim=0).values

    for _ in range(n_iters):
        diff = input_tensors - result
        coeffs = tau / diff.norm(dim=1)
        n_clipped = (coeffs < torch.tensor(1.0)).sum()

        coeffs = weights * torch.minimum(torch.tensor(1.0), coeffs)
        step = (diff * coeffs[:, None]).sum(dim=0) / weights.sum()
        result += step

    return CenteredClipResult(result=result, n_clipped=n_clipped, step_norm=step.norm())
