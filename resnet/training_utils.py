import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from centered_clip import decentralized_centered_clip


transform_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_deterministic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class NormalParticipant:
    """ An overridable code that either behaves normally or performs an attack """
    def __init__(self, model, optimizer, scheduler):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
    
    def compute_grads(self, inputs, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        loss.backward()


def train_with_centerclip(config, device: torch.device, writer: Optional[SummaryWriter] = None, verbose: int = 0):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(config.GLOBAL_SEED)  # seed for init
    model = config.MODEL().to(device)
    
    torch.manual_seed(config.GLOBAL_SEED * world_size + rank)  # seed for minibatches
    if verbose:
        print(f'==> [worker {rank}] Preparing data..')
        
    transform_train = transform_augment if config.AUGMENT_DATA else transform_deterministic
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_deterministic)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE_PER_WORKER, shuffle=True, num_workers=0)

    
    # optimizers and LR
    steps_per_global_epoch = int(len(trainset) / config.BATCH_SIZE_PER_WORKER / config.NUM_WORKERS)
    steps_per_local_epoch = int(len(trainset) / config.BATCH_SIZE_PER_WORKER)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.BASE_LR, momentum=config.MOMENTUM,
                                nesterov=config.NESTEROV, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.COSINE_T_MAX_RATE * config.MAX_EPOCHS_PER_WORKER * steps_per_local_epoch)
    
    participant_cls = config.BYZANTINE_PARTICIPANT if rank in config.BYZANTINE_IDS else config.BENIGN_PARTICIPANT
    participant = participant_cls(model, optimizer, scheduler)
    
    loss_history, acc_history = [], []
    total_steps = 0
    
    if hasattr(config, 'INITIAL_CHECKPOINT'):
        print(f'[*] Resuming from step {config.INITIAL_STEP}, state `{config.INITIAL_CHECKPOINT}`...')
        
        total_steps = config.INITIAL_STEP
        
        with open(config.INITIAL_CHECKPOINT, 'rb') as f:
            state = pickle.load(f)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['opt'])

        # Empty memory to fit the evaluator to 1080 Ti
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for _ in range(total_steps):
                scheduler.step()

        print('[+] State loaded')
    
    dist.barrier()
    start_time = time.time()
        
    for epoch_i in range(config.MAX_EPOCHS_PER_WORKER):
        if verbose:
            print(f'==> [worker {rank}] Began epoch {epoch_i}..')
        train_loss, train_acc = 0, 0
            
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.train(True)
            optimizer.zero_grad()

            outputs = model(inputs)
            participant.compute_grads(inputs, outputs, targets)
            
            with torch.no_grad():
                loss = F.cross_entropy(outputs, targets)
                acc = torch.mean((torch.argmax(outputs, dim=-1) == targets).to(torch.float))
                del outputs
            
            with torch.no_grad():
                grads = [param.grad for param in model.parameters()]
                clipped_grads, clip_stats = decentralized_centered_clip(
                    grads, tau=config.CCLIP_TAU, n_iters=config.CCLIP_MAX_ITERS, eps=config.CCLIP_EPS)
                for grad, clipped in zip(grads, clipped_grads):
                    grad[...] = clipped

            optimizer.step()
            scheduler.step()
            
            # Fixes a bug
            max_metrics_tuple = torch.tensor([clip_stats.n_clipped, clip_stats.step_norm,
                                              clip_stats.num_steps, clip_stats.std])
            dist.all_reduce(max_metrics_tuple, op=dist.ReduceOp.MAX)
            
            metrics_tuple = torch.tensor([loss, acc, clip_stats.n_clipped, clip_stats.step_norm,
                                          clip_stats.num_steps, clip_stats.std])
            dist.all_reduce(metrics_tuple, op=dist.ReduceOp.SUM)
            metrics_tuple /= world_size
            loss, acc, clip_stats.n_clipped, clip_stats.step_norm, clip_stats.num_steps, clip_stats.std = list(metrics_tuple)

            loss_history.append(loss.item())
            acc_history.append(acc.item())
            
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_step=total_steps)
                writer.add_scalar('train/acc', acc.item(), global_step=total_steps)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=total_steps)
                writer.add_scalar('train/global_epoch', total_steps / steps_per_global_epoch, global_step=total_steps)
                writer.add_scalar('train/local_epoch', epoch_i + batch_idx * config.BATCH_SIZE_PER_WORKER / len(trainset),
                                  global_step=total_steps)
                
                writer.add_scalar('util/n_clipped', clip_stats.n_clipped, global_step=total_steps)
                writer.add_scalar('util/final_step_norm', clip_stats.step_norm, global_step=total_steps)
                writer.add_scalar('util/std', clip_stats.std, global_step=total_steps)
                writer.add_scalar('util/num_steps', clip_stats.num_steps, global_step=total_steps)
                writer.add_scalar('util/mean_vector_std', clip_stats.std, global_step=total_steps)
                
                writer.add_scalar('util/max_n_clipped', max_metrics_tuple[0], global_step=total_steps)
                writer.add_scalar('util/max_step_norm', max_metrics_tuple[1], global_step=total_steps)
                writer.add_scalar('util/max_num_steps', max_metrics_tuple[2], global_step=total_steps)
                writer.add_scalar('util/max_vector_std', max_metrics_tuple[3], global_step=total_steps)
                
            checkpoint_dump_steps = getattr(config, 'CHECKPOINT_DUMP_STEPS', [])
            if rank == 0 and total_steps in checkpoint_dump_steps:
                filename = f'state_step_{total_steps}_exp_{config.EXP_NAME}.pickle'
                state = {
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                }
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
                print(f'[+] Saved checkpoint to {filename}')
            
            if total_steps % config.EVAL_EVERY == 0:
                checksum_match = verify_equal_parameters(model)
                if rank == 0:
                    val_acc = evaluate_accuracy(model, testset, config.EVAL_BATCH_SIZE)
                    if writer:
                        writer.add_scalar('test/accuracy', val_acc, global_step=total_steps)
                    if verbose:
                        print(end=f'step {str(total_steps).rjust(5, "0")}\t| val accuracy = {val_acc:.5f}\t| training for {time.time() - start_time:.5f}s.\t| checksum ok = {checksum_match}\n')
                
                dist.barrier()
                if verbose >= 2:
                    print(end=f"worker {str(rank).rjust(2, '0')}, step {total_steps}\t| loss: {np.mean(loss_history[-config.EVAL_EVERY:]):.5f},"
                              f" acc: {np.mean(acc_history[-config.EVAL_EVERY:]):.5f}\n")
                    if rank == 0:
                        print()
                        
            total_steps += 1
            if total_steps >= getattr(config, 'EARLY_STOP_STEPS', float('inf')):
                if verbose:
                    print(f"worker {str(rank).rjust(2, '0')} stopping at {total_steps}")
                return model, optimizer
            

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int = 0):
    """ Evaluate on the entire dataset """
    model.train(False)
    device = next(iter(model.parameters())).device
    acc_numerator = acc_denominator = 0
    for inputs, targets in torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        inputs, targets = inputs.to(device), targets.to(device)
        acc_numerator += (model(inputs).argmax(-1) == targets).to(torch.float32).sum()
        acc_denominator += len(inputs)
    return acc_numerator / acc_denominator


@torch.no_grad()
def verify_equal_parameters(model):
    """ debug call to check that all workers have the same model parameters """
    with torch.no_grad():
        checksum = sum([p.sum() for p in model.parameters()]).cpu()
        checksums = torch.randn(dist.get_world_size())
        dist.all_gather(list(checksums), checksum)
        return torch.allclose(checksums[:-1], checksums[1:])
