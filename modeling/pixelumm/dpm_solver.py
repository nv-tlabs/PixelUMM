# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal flow-matching DPM-Solver++ used by PixelUMM image/video inference.

This is the exact release path: flow prediction, shifted uniform-flow time
steps, and second-order multistep DPM-Solver++. Generic VP schedules,
classifier/PAG guidance, inverse solving, adaptive stepping, and third-order
solvers are intentionally not part of the public runtime.
"""

from collections.abc import Callable

import torch
from tqdm import tqdm


def sample_flow_dpm(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x: torch.Tensor,
    *,
    num_steps: int,
    shift: float,
) -> torch.Tensor:
    """Sample image or video flow; the caller owns conditioning and CFG."""
    noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")

    def model(x: torch.Tensor, t_input: torch.Tensor) -> torch.Tensor:
        t_value = t_input.flatten()[0].item() / noise_schedule.total_N
        return velocity_fn(x, t_value)

    model_fn = model_wrapper(
        model, noise_schedule, model_type="flow", guidance_type="uncond"
    )
    solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
    return solver.sample(
        x,
        steps=num_steps,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        flow_shift=shift,
        t_start=1.0,
        t_end=0.001,
    )


class NoiseScheduleFlow:
    """Rectified-flow schedule used by the released PixelUMM checkpoint."""

    def __init__(self, schedule: str = "discrete_flow") -> None:
        if schedule != "discrete_flow":
            raise ValueError(f"PixelUMM requires schedule='discrete_flow', got {schedule!r}")
        self.T = 1.0
        self.t0 = 0.001
        self.schedule = schedule
        self.total_N = 1000

    @staticmethod
    def marginal_alpha(t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(self.marginal_alpha(t))

    @staticmethod
    def marginal_std(t: torch.Tensor) -> torch.Tensor:
        return t

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        return self.marginal_log_mean_coeff(t) - torch.log(self.marginal_std(t))


def _expand_dims(value: torch.Tensor, dims: int) -> torch.Tensor:
    return value[(...,) + (None,) * (dims - 1)]


def model_wrapper(
    model: Callable[..., torch.Tensor],
    noise_schedule: NoiseScheduleFlow,
    *,
    model_type: str = "flow",
    guidance_type: str = "uncond",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Convert PixelUMM's flow velocity into DPM-Solver noise prediction."""

    if model_type != "flow":
        raise ValueError(f"PixelUMM DPM inference requires model_type='flow', got {model_type!r}")
    if guidance_type != "uncond":
        raise ValueError(f"PixelUMM applies CFG inside the model; got guidance_type={guidance_type!r}")
    if noise_schedule.schedule != "discrete_flow":
        raise ValueError(f"Unsupported noise schedule {noise_schedule.schedule!r}")

    def model_fn(x: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        t_input = t_continuous * noise_schedule.total_N
        velocity = model(x, t_input)
        if not isinstance(velocity, torch.Tensor):
            raise TypeError(f"PixelUMM flow model must return a Tensor, got {type(velocity).__name__}")
        sigma_t = noise_schedule.marginal_std(t_continuous)
        return (1 - _expand_dims(sigma_t, x.dim()).to(x)) * velocity + x

    return model_fn


class DPM_Solver:
    """Second-order multistep DPM-Solver++ for the PixelUMM flow schedule."""

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise_schedule: NoiseScheduleFlow,
        algorithm_type: str = "dpmsolver++",
    ) -> None:
        if algorithm_type != "dpmsolver++":
            raise ValueError(f"PixelUMM requires algorithm_type='dpmsolver++', got {algorithm_type!r}")
        if noise_schedule.schedule != "discrete_flow":
            raise ValueError(f"Unsupported noise schedule {noise_schedule.schedule!r}")
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule

    def _data_prediction(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = self.model(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        return (x - sigma_t * noise) / alpha_t

    @staticmethod
    def _time_steps(
        *,
        t_start: float,
        t_end: float,
        steps: int,
        device: torch.device,
        flow_shift: float,
    ) -> torch.Tensor:
        betas = torch.linspace(t_start, t_end, steps + 1, device=device)
        sigmas = 1.0 - betas
        return (flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)).flip(dims=[0])

    def _first_order_update(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        model_s: torch.Tensor,
    ) -> torch.Tensor:
        schedule = self.noise_schedule
        h = schedule.marginal_lambda(t) - schedule.marginal_lambda(s)
        sigma_s = schedule.marginal_std(s)
        sigma_t = schedule.marginal_std(t)
        alpha_t = torch.exp(schedule.marginal_log_mean_coeff(t))
        phi_1 = torch.expm1(-h)
        return sigma_t / sigma_s * x - alpha_t * phi_1 * model_s

    def _second_order_update(
        self,
        x: torch.Tensor,
        model_prev: list[torch.Tensor],
        t_prev: list[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        schedule = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev[-2], model_prev[-1]
        t_prev_1, t_prev_0 = t_prev[-2], t_prev[-1]
        lambda_prev_1 = schedule.marginal_lambda(t_prev_1)
        lambda_prev_0 = schedule.marginal_lambda(t_prev_0)
        lambda_t = schedule.marginal_lambda(t)
        sigma_prev_0 = schedule.marginal_std(t_prev_0)
        sigma_t = schedule.marginal_std(t)
        alpha_t = torch.exp(schedule.marginal_log_mean_coeff(t))

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        d1 = (h / h_0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        return (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * d1
        )

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        *,
        steps: int = 20,
        t_start: float = 1.0,
        t_end: float = 0.001,
        order: int = 2,
        skip_type: str = "time_uniform_flow",
        method: str = "multistep",
        lower_order_final: bool = True,
        solver_type: str = "dpmsolver",
        flow_shift: float = 1.0,
    ) -> torch.Tensor:
        """Sample with the one solver configuration shipped by PixelUMM."""

        if steps < 2:
            raise ValueError(f"DPM-Solver++ requires at least two steps, got {steps}")
        if order != 2:
            raise ValueError(f"PixelUMM DPM inference requires order=2, got {order}")
        if skip_type != "time_uniform_flow":
            raise ValueError(f"PixelUMM requires skip_type='time_uniform_flow', got {skip_type!r}")
        if method != "multistep":
            raise ValueError(f"PixelUMM requires method='multistep', got {method!r}")
        if solver_type != "dpmsolver":
            raise ValueError(f"PixelUMM requires solver_type='dpmsolver', got {solver_type!r}")
        if not lower_order_final:
            raise ValueError("PixelUMM requires lower_order_final=True")
        if t_start <= 0 or t_end <= 0:
            raise ValueError("DPM-Solver time range must be positive")
        if flow_shift <= 0:
            raise ValueError(f"flow_shift must be positive, got {flow_shift}")

        timesteps = self._time_steps(
            t_start=t_start,
            t_end=t_end,
            steps=steps,
            device=x.device,
            flow_shift=flow_shift,
        )
        t_prev = [timesteps[0]]
        model_prev = [self._data_prediction(x, timesteps[0])]

        x = self._first_order_update(x, t_prev[-1], timesteps[1], model_prev[-1])
        t_prev.append(timesteps[1])
        model_prev.append(self._data_prediction(x, timesteps[1]))

        for step in tqdm(range(2, steps + 1), desc="DPM-Solver++"):
            t = timesteps[step]
            if step == steps:
                x = self._first_order_update(x, t_prev[-1], t, model_prev[-1])
            else:
                x = self._second_order_update(x, model_prev, t_prev, t)
            t_prev[0], t_prev[1] = t_prev[1], t
            model_prev[0] = model_prev[1]
            if step < steps:
                model_prev[1] = self._data_prediction(x, t)
        return x
