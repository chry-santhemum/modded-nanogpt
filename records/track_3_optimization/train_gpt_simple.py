"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def apply_raw_second_moment_torch(update: Tensor, raw_second_moment: Tensor) -> Tensor:
    """Apply E[x x^T] to a PyTorch Linear update in (fan_out, fan_in) layout."""
    return update.float() @ raw_second_moment.float()


def precondition_by_isotropic_plus_mean(
    momentum_update: Tensor,
    mean: Tensor,
    sigma_sq: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Apply (sigma_sq I + mean mean^T)^(-1) on the input side."""
    momentum_update = momentum_update.float()
    mean = mean.float()
    sigma_sq = sigma_sq.float()
    mean_sq = torch.sum(mean * mean)
    mean_proj = momentum_update @ mean
    return (
        momentum_update
        - mean_proj[:, None] * (mean[None, :] / (sigma_sq + mean_sq + eps))
    ) / (sigma_sq + eps)


def estimate_mean_iso_alpha(
    momentum_update: Tensor,
    mean: Tensor,
    sigma_sq: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    preconditioned_momentum = precondition_by_isotropic_plus_mean(
        momentum_update,
        mean,
        sigma_sq,
        eps=eps,
    )
    cross_scale = torch.linalg.vector_norm(preconditioned_momentum)
    cross_scale /= min(momentum_update.shape) ** 0.5
    return 1.0 / (cross_scale + eps)


def estimate_first_fw_alpha(
    momentum_update: Tensor,
    raw_second_moment: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    momentum_update = momentum_update.float()
    atom = zeropower_via_newtonschulz5(momentum_update)
    curved_atom = apply_raw_second_moment_torch(atom, raw_second_moment)
    num = torch.sum(momentum_update * atom)
    den = torch.sum(atom * curved_atom)
    return den / (num + eps)


def solve_spectral_fw(
    momentum_update: Tensor,
    raw_second_moment: Tensor,
    alpha: Tensor,
    fw_steps: int,
    fw_gamma_method: str,
    eps: float = 1e-8,
) -> Tensor:
    """Run the muon_foof Frank-Wolfe loop in PyTorch Linear weight layout."""
    target = alpha * momentum_update.float()
    raw_second_moment = raw_second_moment.float()
    update = torch.zeros_like(target)

    for i in range(fw_steps):
        residual = target - apply_raw_second_moment_torch(update, raw_second_moment)
        atom = zeropower_via_newtonschulz5(residual)
        direction = atom - update
        if fw_gamma_method == "line_search":
            direction_hess = apply_raw_second_moment_torch(direction, raw_second_moment)
            gamma = torch.sum(residual * direction) / (torch.sum(direction * direction_hess) + eps)
            gamma = torch.clamp(gamma, 0.0, 1.0)
        elif fw_gamma_method == "default":
            gamma = 2.0 / (i + 2.0)
        else:
            raise ValueError(f"Unknown FW gamma method: {fw_gamma_method}")
        update = update + gamma * direction
    return update


@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


class MuonFOOF(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        activation_stats,
        lr=0.02,
        beta_1=0.95,
        nesterov=True,
        fw_alpha_method="mean_iso",
        fw_alpha_mult=1.0,
        fw_steps=3,
        fw_gamma_method="line_search",
        weight_decay=0.02,
    ):
        assert isinstance(params, list) and len(params) >= 1
        assert isinstance(params[0], torch.nn.Parameter)
        assert fw_steps >= 1
        assert fw_alpha_mult >= 0.0
        assert fw_alpha_method in {"mean_iso", "first_fw"}
        assert fw_gamma_method in {"line_search", "default"}
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr,
            beta_1=beta_1,
            nesterov=nesterov,
            fw_alpha_method=fw_alpha_method,
            fw_alpha_mult=fw_alpha_mult,
            fw_steps=fw_steps,
            fw_gamma_method=fw_gamma_method,
            weight_decay=weight_decay,
            beta_prod=1.0,
        )
        super().__init__(params, defaults)
        self.activation_stats = activation_stats

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            beta = group["beta_1"]
            group["beta_prod"] *= beta
            beta_prod = group["beta_prod"]
            bias_correction = 1 - beta_prod
            params = group["params"]
            pad_count = world_size - len(params) % world_size
            params_pad = params + [torch.empty_like(params[-1])] * pad_count
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    stat = self.activation_stats[p]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                        state["mean_ema"] = torch.zeros_like(stat["mean"])
                        state["uncentered_cov_ema"] = torch.zeros_like(stat["uncentered_cov"])

                    grad = p.grad.float()
                    momentum = state["momentum"]
                    momentum.mul_(beta).add_(grad, alpha=1 - beta)
                    if group["nesterov"]:
                        beta_prod_next = beta * beta_prod
                        momentum_update = (
                            beta * momentum / (1 - beta_prod_next)
                            + (1 - beta) * grad / bias_correction
                        )
                    else:
                        momentum_update = momentum / bias_correction

                    state["mean_ema"].mul_(beta).add_(stat["mean"], alpha=1 - beta)
                    state["uncentered_cov_ema"].mul_(beta).add_(stat["uncentered_cov"], alpha=1 - beta)
                    mean = state["mean_ema"] / bias_correction
                    raw_second_moment = state["uncentered_cov_ema"] / bias_correction

                    if group["fw_alpha_method"] == "mean_iso":
                        sigma_sq = (torch.trace(raw_second_moment) - torch.sum(mean * mean)) / p.size(1)
                        alpha_base = estimate_mean_iso_alpha(momentum_update, mean, sigma_sq)
                    elif group["fw_alpha_method"] == "first_fw":
                        alpha_base = estimate_first_fw_alpha(
                            momentum_update,
                            raw_second_moment,
                        )
                    alpha = group["fw_alpha_mult"] * alpha_base
                    update = solve_spectral_fw(
                        momentum_update,
                        raw_second_moment,
                        alpha=alpha,
                        fw_steps=group["fw_steps"],
                        fw_gamma_method=group["fw_gamma_method"],
                    )
                    update *= max(1, p.size(0) / p.size(1))**0.5

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


def add_activation_stat_hooks(model, tracked_params):
    """
    Track per-step input statistics for Linear weights in tracked_params.

    For a Linear weight shaped (out_features, in_features), the tracked
    activation mean has shape (in_features,), and the uncentered covariance
    E[x.T @ x] has shape (in_features, in_features).
    """
    tracked_params = set(tracked_params)
    activation_stats = {}

    def record_activation_stats(module, args):
        if not module.training or module.weight not in tracked_params:
            return
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        stat = activation_stats[module.weight]
        stat["mean"].add_(x.sum(dim=0))
        stat["uncentered_cov"].addmm_(x.T, x)
        stat["count"].add_(x.shape[0])

    for module in model.modules():
        if isinstance(module, Linear) and module.weight in tracked_params:
            in_features = module.weight.shape[1]
            activation_stats[module.weight] = {
                "mean": torch.zeros(in_features, device=module.weight.device),
                "uncentered_cov": torch.zeros(in_features, in_features, device=module.weight.device),
                "count": torch.zeros((), device=module.weight.device),
            }
            module.register_forward_pre_hook(record_activation_stats)
    return activation_stats


def finalize_activation_stats(activation_stats):
    """
    Convert local per-step sums to global averages.

    This relies on each rank seeing the same number of activation rows, which
    holds for this script because batch_size is evenly split across ranks.
    """
    world_size = dist.get_world_size()
    for stat in activation_stats.values():
        assert stat["count"].item() > 0
        stat["mean"].div_(stat["count"])
        stat["uncentered_cov"].div_(stat["count"])
        dist.all_reduce(stat["mean"], op=dist.ReduceOp.SUM)
        dist.all_reduce(stat["uncentered_cov"], op=dist.ReduceOp.SUM)
        stat["mean"].div_(world_size)
        stat["uncentered_cov"].div_(world_size)


def clear_current_activation_stats(activation_stats):
    for stat in activation_stats.values():
        stat["mean"].zero_()
        stat["uncentered_cov"].zero_()
        stat["count"].zero_()


def reset_activation_stats(activation_stats):
    for stat in activation_stats.values():
        stat["mean"].zero_()
        stat["uncentered_cov"].zero_()
        stat["count"].zero_()


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
activation_stats = add_activation_stat_hooks(model, matrix_params)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3375

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
    reset_activation_stats(activation_stats)

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = MuonFOOF(
        matrix_params, activation_stats,
        lr=0.02,
        beta_1=0.95,
        nesterov=True,
        fw_alpha_method="mean_iso",
        fw_alpha_mult=2.0,
        fw_steps=3,
        fw_gamma_method="line_search",
        weight_decay=0.02,
    )
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        if step == train_steps or step % 125 == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        finalize_activation_stats(activation_stats)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        clear_current_activation_stats(activation_stats)
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
