import random

import torch
import torch.nn.functional as F


class SignFlipper:
    def __init__(self, model, optimizer, scheduler, ban_prob: float, attack_start: int,
                 direction_seed: int = 0, attack_every: int = 1):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.ban_prob, self.attack_start = ban_prob, attack_start
        self.num_steps, self.banned = 0, False
        self.attack_every = attack_every
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.ban_prob=}, {self.attack_start=}, {self.flip_scale=})"

    def compute_grads(self, inputs, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        if self.num_steps > self.attack_start and self.num_steps % self.attack_every == 0 and not self.banned:
            print(end=f"ATTACK@{self.num_steps}\n")
            with torch.no_grad():
                for param in self.model.parameters():
                    param.grad *= -1000
            
            if random.random() < self.ban_prob:
                print(f"BANNED@{self.num_steps}\n")
                self.banned = True
                # note: after peer is "banned", it starts sending correct gradients
                # as if it was replaced with a normal peer
        
        self.num_steps += 1
        
        
class LabelFlipper:
    def __init__(self, model, optimizer, scheduler, ban_prob: float, attack_start: int,
                 direction_seed: int = 0, attack_every: int = 1):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.ban_prob, self.attack_start = ban_prob, attack_start
        self.num_steps, self.banned = 0, False
        self.attack_every = attack_every
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.ban_prob=}, {self.attack_start=}, {self.flip_scale=})"

    def compute_grads(self, inputs, outputs, targets):
        if self.num_steps > self.attack_start and self.num_steps % self.attack_every == 0 and not self.banned:
            print(end=f"ATTACK@{self.num_steps}\n")
            
            loss = F.cross_entropy(outputs, 9 - targets)
            loss.backward()
                    
            if random.random() < self.ban_prob:
                print(f"BANNED@{self.num_steps}\n")
                self.banned = True
                # note: after peer is "banned", it starts sending correct gradients
                # as if it was replaced with a normal peer
        else:
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        
        self.num_steps += 1
        
        
class ConstantDirection:
    def __init__(self, model, optimizer, scheduler, ban_prob: float, attack_start: int,
                 direction_seed: int = 0, attack_every: int = 1):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.ban_prob, self.attack_start = ban_prob, attack_start
        self.num_steps, self.banned = 0, False
        self.direction_seed = direction_seed
        self.attack_every = attack_every
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.ban_prob=}, {self.attack_start=}, {self.flip_scale=})"

    def compute_grads(self, inputs, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        if self.num_steps > self.attack_start and self.num_steps % self.attack_every == 0 and not self.banned:
            print(end=f"ATTACK@{self.num_steps}\n")
            grad_devices = {param.grad.device for param in self.model.parameters()}
            with torch.no_grad(), torch.random.fork_rng(grad_devices):
                torch.manual_seed(self.direction_seed)
                for param in self.model.parameters():
                    rand_buf = torch.randn_like(param.grad)
                    param.grad[...] = rand_buf * (1 / rand_buf.norm() * param.grad.norm() * 1000)
            
            if random.random() < self.ban_prob:
                print(f"BANNED@{self.num_steps}\n")
                self.banned = True
                # note: after peer is "banned", it starts sending correct gradients
                # as if it was replaced with a normal peer
        
        self.num_steps += 1

        
class DelayedGradientAttacker:
    def __init__(self, model, optimizer, scheduler, ban_prob: float,
                 attack_start: int, direction_seed: int = 0, attack_every: int = 1,
                 delay: int = 1000, attack_length: int = 500):
        # Default attack_length is chosen for the worst case

        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.ban_prob = ban_prob
        self.attack_start = attack_start
        self.attack_end = attack_start + attack_length * attack_every
        self.delay = delay
        self.num_steps, self.banned = 0, False
        self.attack_every = attack_every
        
        self.old_grads = {}
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.ban_prob=}, {self.attack_start=}, {self.flip_scale=})"

    def compute_grads(self, inputs, outputs, targets):
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        reuse_step = self.num_steps + self.delay
        if self.attack_start < reuse_step < self.attack_end and reuse_step % self.attack_every == 0:
            print(f'[+] Saved grads for step {self.num_steps}')
            self.old_grads[self.num_steps] = [param.grad.to('cpu')
                                              for param_group in self.optimizer.param_groups
                                              for param in param_group['params']]

        if self.attack_start < self.num_steps < self.attack_end and self.num_steps % self.attack_every == 0 and not self.banned:
            print(end=f"ATTACK@{self.num_steps}\n")
            
            old_grads = iter(self.old_grads[self.num_steps - self.delay])
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad[...] = next(old_grads).to(param.grad.device)

            if random.random() < self.ban_prob:
                print(f"BANNED@{self.num_steps}\n")
                self.banned = True
                # note: after peer is "banned", it starts sending correct gradients
                # as if it was replaced with a normal peer

        self.num_steps += 1
