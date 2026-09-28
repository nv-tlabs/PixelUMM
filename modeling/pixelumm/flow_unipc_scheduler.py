# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal flow-prediction UniPC scheduler used by PixelUMM inference.

Adapted from the Apache-2.0 licensed Wan2.2 flow-UniPC implementation at
``wan/utils/fm_solvers_unipc.py`` (Wan-Video/Wan2.2 commit
42bf4cfaa384bc21833865abc2f9e6c0e67233dc), itself adapted from Diffusers
v0.31.0. This modified file intentionally retains only the fixed execution
path used by the released PixelUMM R07 inference contract: second-order UniPC,
flow prediction, B(h) solver type ``bh2``, no dynamic thresholding, and a
zero terminal sigma.

Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class FlowUniPCScheduler:
    """Fixed-contract flow-UniPC scheduler for released PixelUMM models."""

    solver_order = 2

    def __init__(self, *, num_train_timesteps: int = 1000) -> None:
        if num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be positive")
        self.num_train_timesteps = int(num_train_timesteps)

        alphas = np.linspace(
            1,
            1 / self.num_train_timesteps,
            self.num_train_timesteps,
        )[::-1].copy()
        sigmas = 1.0 - alphas
        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device="cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.timesteps = self.sigmas * self.num_train_timesteps
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 0
        self.step_index: int | None = None
        self.num_inference_steps: int | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device,
        shift: float,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("shift must be finite and positive")

        sigmas = np.linspace(
            self.sigma_max,
            self.sigma_min,
            num_inference_steps + 1,
        ).copy()[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0
        self.step_index = None

    @staticmethod
    def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def _convert_velocity(
        self,
        velocity: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        assert self.step_index is not None
        sigma = self.sigmas[self.step_index]
        return sample - sigma * velocity

    def _predictor_update(
        self,
        sample: torch.Tensor,
        *,
        order: int,
    ) -> torch.Tensor:
        assert self.step_index is not None
        if order not in (1, 2):
            raise ValueError(f"unsupported UniPC predictor order: {order}")

        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        if s0 is None or m0 is None:
            raise RuntimeError("UniPC predictor state is incomplete")

        sigma_t_raw = self.sigmas[self.step_index + 1]
        sigma_s0_raw = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t_raw)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0_raw)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = torch.expm1(hh)

        correction: torch.Tensor | int = 0
        if order == 2:
            previous = model_output_list[-2]
            if previous is None:
                raise RuntimeError("UniPC second-order predictor state is incomplete")
            previous_sigma = self.sigmas[self.step_index - 1]
            previous_alpha, previous_sigma = self._alpha_sigma(previous_sigma)
            previous_lambda = torch.log(previous_alpha) - torch.log(previous_sigma)
            rk = (previous_lambda - lambda_s0) / h
            d1 = (previous - m0) / rk
            correction = 0.5 * d1

        predicted = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        predicted = predicted - alpha_t * b_h * correction
        return predicted.to(sample.dtype)

    def _corrector_update(
        self,
        this_model_output: torch.Tensor,
        *,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        assert self.step_index is not None
        if order not in (1, 2):
            raise ValueError(f"unsupported UniPC corrector order: {order}")

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("UniPC corrector state is incomplete")

        sigma_t_raw = self.sigmas[self.step_index]
        sigma_s0_raw = self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t_raw)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0_raw)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_2 = h_phi_1 / hh - 1
        b_h = torch.expm1(hh)

        if order == 1:
            rhos = torch.tensor([0.5], dtype=last_sample.dtype, device=last_sample.device)
            correction: torch.Tensor | int = 0
        else:
            previous = model_output_list[-2]
            if previous is None:
                raise RuntimeError("UniPC second-order corrector state is incomplete")
            previous_sigma = self.sigmas[self.step_index - 2]
            previous_alpha, previous_sigma = self._alpha_sigma(previous_sigma)
            previous_lambda = torch.log(previous_alpha) - torch.log(previous_sigma)
            rk = (previous_lambda - lambda_s0) / h
            d1 = (previous - m0) / rk

            rks = torch.tensor([rk, 1.0], device=last_sample.device)
            matrix = torch.stack([torch.ones_like(rks), rks])
            h_phi_3 = h_phi_2 / hh - 0.5
            rhs = torch.tensor(
                [h_phi_2 / b_h, h_phi_3 * 2 / b_h],
                device=last_sample.device,
            )
            rhos = torch.linalg.solve(matrix, rhs).to(last_sample.dtype)
            correction = rhos[0] * d1

        d1_t = this_model_output - m0
        corrected = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        corrected = corrected - alpha_t * b_h * (correction + rhos[-1] * d1_t)
        return corrected.to(last_sample.dtype)

    def _index_for_timestep(self, timestep: torch.Tensor) -> int:
        indices = (self.timesteps == timestep.to(self.timesteps.device)).nonzero()
        if len(indices) == 0:
            raise ValueError(f"timestep {timestep.item()} is not in the inference schedule")
        position = 1 if len(indices) > 1 else 0
        return indices[position].item()

    def step(
        self,
        *,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("set_timesteps must be called before step")
        if self.step_index is None:
            self.step_index = self._index_for_timestep(timestep)

        use_corrector = self.step_index > 0 and self.last_sample is not None
        converted = self._convert_velocity(model_output, sample)
        if use_corrector:
            sample = self._corrector_update(
                converted,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        self.model_outputs[0] = self.model_outputs[1]
        self.timestep_list[0] = self.timestep_list[1]
        self.model_outputs[1] = converted
        self.timestep_list[1] = timestep

        remaining_steps = len(self.timesteps) - self.step_index
        order = min(self.solver_order, remaining_steps)
        self.this_order = min(order, self.lower_order_nums + 1)
        if self.this_order <= 0:
            raise RuntimeError("invalid UniPC solver order")

        self.last_sample = sample
        previous = self._predictor_update(sample, order=self.this_order)
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self.step_index += 1
        return previous


__all__ = ["FlowUniPCScheduler"]
