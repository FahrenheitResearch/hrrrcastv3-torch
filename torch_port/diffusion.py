from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


NUM_DIFFUSION_STEPS = 200
NUM_INFERENCE_STEPS = 50
USE_LOGSNR_SPACED = True


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    beta_start = 0.0001
    beta_end = 0.9999

    def alpha_bar_fn(t: int) -> float:
        return np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2

    alphas_bar = np.array([alpha_bar_fn(t) for t in range(timesteps + 1)])
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, beta_start, beta_end)


def compute_log_snr_spaced_steps(
    num_inference_steps: int,
    num_diffusion_steps: int,
    alpha_bar: np.ndarray,
) -> np.ndarray:
    alpha_bar = np.array(alpha_bar, dtype=np.float64)
    alpha_bar = np.clip(alpha_bar, 1e-10, 1.0 - 1e-10)
    log_snr = np.log(alpha_bar / (1.0 - alpha_bar))
    target_log_snr = np.linspace(log_snr.max(), log_snr.min(), num_inference_steps * 178 // 100)

    seen = set()
    selected: list[int] = []
    for target in target_log_snr:
        idx = int(np.argmin(np.abs(log_snr - target)))
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)
            if len(selected) == num_inference_steps:
                break

    if len(selected) < num_inference_steps:
        remaining = [t for t in range(num_diffusion_steps) if t not in seen]
        remaining.sort(key=lambda t: log_snr[t], reverse=True)
        selected.extend(remaining[: num_inference_steps - len(selected)])

    return np.array(sorted(selected[:num_inference_steps]), dtype=np.int64)


BETA = cosine_beta_schedule(NUM_DIFFUSION_STEPS).astype(np.float32)
ALPHA = (1.0 - BETA).astype(np.float32)
ALPHA_BAR = np.cumprod(ALPHA, axis=0).astype(np.float32)

if USE_LOGSNR_SPACED:
    INFERENCE_STEPS = compute_log_snr_spaced_steps(
        NUM_INFERENCE_STEPS,
        NUM_DIFFUSION_STEPS,
        ALPHA_BAR,
    )
else:
    INFERENCE_STEPS = np.arange(
        0,
        NUM_DIFFUSION_STEPS,
        max(1, NUM_DIFFUSION_STEPS // NUM_INFERENCE_STEPS),
        dtype=np.int64,
    )


def compute_epsilon(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    t: torch.Tensor | int,
    *,
    alpha_bar: np.ndarray = ALPHA_BAR,
) -> torch.Tensor:
    alpha_bar_t = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
    t = torch.as_tensor(t, device=x_t.device, dtype=torch.long)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.index_select(0, t.view(-1))).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t.index_select(0, t.view(-1))).view(-1, 1, 1, 1)
    return (x_t - x_0 * sqrt_alpha_bar_t) / sqrt_one_minus_alpha_bar_t


def _make_noise_like(x_t: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x_t.device.type if x_t.is_cuda else "cpu")
    generator.manual_seed(int(seed))
    return torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)


def ddim(
    x_t: torch.Tensor,
    pred_noise: torch.Tensor,
    t_index: int,
    *,
    seed: int | Sequence[int] = 0,
    eta: float = 0.0,
    alpha_bar: np.ndarray = ALPHA_BAR,
    inference_steps: np.ndarray = INFERENCE_STEPS,
) -> torch.Tensor:
    alpha_bar_all = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
    steps = torch.as_tensor(inference_steps, device=x_t.device, dtype=torch.long)

    t = steps[t_index]
    tm1 = steps[t_index - 1]
    alpha_bar_t = alpha_bar_all[t]
    alpha_bar_tm1 = alpha_bar_all[tm1]

    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    sigma = eta * torch.sqrt((1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * torch.sqrt(
        1.0 - alpha_bar_t / alpha_bar_tm1
    )
    dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_tm1 - sigma**2, min=0.0)) * pred_noise
    mean = torch.sqrt(alpha_bar_tm1) * x0_pred + dir_xt

    if eta == 0.0:
        return mean

    if isinstance(seed, int):
        seeds = [seed] * x_t.shape[0]
    else:
        seeds = list(seed)
    noise = torch.cat(
        [_make_noise_like(x_t[batch_idx : batch_idx + 1], seeds[batch_idx] + int(t.item())) for batch_idx in range(x_t.shape[0])],
        dim=0,
    )
    return mean + sigma * noise
