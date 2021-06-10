import dataclasses
from typing import Sequence, Tuple

import torch
import torch.distributed as dist


def split_into_parts(tensors: Sequence[torch.Tensor], num_parts: int) -> Tuple[torch.Tensor, ...]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    total_size = sum(t.numel() for t in tensors)
    parts = list(map(torch.Tensor.flatten, tensors))
    if total_size % num_parts:
        parts.append(torch.zeros(num_parts - total_size % num_parts, device=parts[0].device))
    flat_tensor = torch.cat(parts)
    return torch.split(flat_tensor, len(flat_tensor) // num_parts, dim=0)


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_tensor = torch.cat(tuple(chunks))[:sum(result_sizes)]
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))


@dataclasses.dataclass(frozen=False)
class CenteredClipResult:
    result: torch.Tensor
    n_clipped: torch.Tensor
    step_norm: torch.Tensor
    num_steps: torch.Tensor
    std: torch.Tensor


def centered_clip(input_tensors: torch.Tensor, weights: torch.Tensor,
                  tau: float, n_iters: int=20, eps: float=1e-6) -> CenteredClipResult:
    result_shape = input_tensors.shape[1:]
    input_tensors = input_tensors.flatten(start_dim=1)

    result = input_tensors.median(dim=0).values
    one = torch.tensor(1.0, device=result.device)

    for i in range(n_iters):
        diff = input_tensors - result
        coeffs = tau / diff.norm(dim=1)
        n_clipped = (coeffs < 1.0).sum()
        coeffs = weights * torch.min(one, coeffs)
        step = (diff * coeffs[:, None]).sum(dim=0) / weights.sum()
        result += step
        if step.norm() <= eps:
            break
    
    vector_std = torch.mean((input_tensors - input_tensors.mean(dim=0)).norm(dim=1) ** 2) ** 0.5
    return CenteredClipResult(result=result, n_clipped=n_clipped, step_norm=step.norm(),
                              num_steps=i, std=vector_std)


def decentralized_centered_clip(local_tensors, **kwargs):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    tensor_parts = list(split_into_parts(local_tensors, num_parts=world_size))
    device = tensor_parts[0].device
    gathered_from_peers = torch.empty(world_size, len(tensor_parts[rank]), device=device)
    handles = []
    for j in range(world_size):
        handles.append(dist.scatter(
            gathered_from_peers[j], tensor_parts if rank == j else None, src=j, async_op=True))
    for handle in handles:
        handle.wait()
        
    clipped = centered_clip(gathered_from_peers, weights=torch.ones(world_size, device=device), **kwargs)
    
    dist.barrier()
    dist.all_gather(tensor_parts, clipped.result)
    return restore_from_parts(tensor_parts, [t.shape for t in local_tensors]), clipped
    