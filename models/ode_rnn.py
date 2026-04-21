from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from models.continuous import ODEFunc


class ODERNN(nn.Module):
    """Standalone ODE-RNN for irregularly sampled time series.

    The hidden state evolves continuously between observations via an ODE
    solve and is updated at observed context points with a GRUCell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        solver_type: str = "dopri5",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        start_time: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.solver_type = solver_type
        self.atol = atol
        self.rtol = rtol
        self.start_time = start_time

        self.hidden_dynamics = ODEFunc(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.update_cell = nn.GRUCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.last_nfe = 0

    def reset_nfe(self) -> None:
        """Resets forward-pass NFE accounting."""
        self.hidden_dynamics.nfe = 0
        self.last_nfe = 0

    def get_nfe(self) -> int:
        """Returns forward-pass NFE from the most recent call."""
        return self.last_nfe

    def _validate_inputs(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> None:
        if context_times.ndim != 2:
            raise ValueError(
                "context_times must have shape [batch, num_context_points]."
            )
        if context_observations.ndim != 3:
            raise ValueError(
                "context_observations must have shape "
                "[batch, num_context_points, input_dim]."
            )
        if context_mask.ndim != 2:
            raise ValueError(
                "context_mask must have shape [batch, num_context_points]."
            )
        if query_times.ndim != 2:
            raise ValueError(
                "query_times must have shape [batch, num_query_points]."
            )

        if context_times.shape[:2] != context_observations.shape[:2]:
            raise ValueError(
                "context_times and context_observations must agree on batch "
                "and sequence dimensions."
            )
        if context_times.shape != context_mask.shape:
            raise ValueError(
                "context_times and context_mask must have identical shapes."
            )
        if query_times.size(1) < context_times.size(1):
            raise ValueError(
                "query_times must include all context_times as a prefix."
            )
        if context_observations.size(-1) != self.input_dim:
            raise ValueError(
                "context_observations last dimension does not match input_dim."
            )

    def _solve_hidden(
        self,
        hidden: torch.Tensor,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
    ) -> torch.Tensor:
        """Integrates the hidden state between two timestamps."""
        if bool(torch.isclose(start_time, end_time).item()):
            return hidden

        integration_times = torch.stack([start_time, end_time]).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )

        hidden_trajectory = odeint(
            func=self.hidden_dynamics,
            y0=hidden,
            t=integration_times,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver_type,
        )
        return hidden_trajectory[-1]

    def _forward_single(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Runs ODE-RNN inference for a single sample."""
        if not torch.all(context_times[1:] > context_times[:-1]):
            raise ValueError(
                "context_times must be strictly increasing for every sample."
            )
        if not torch.all(query_times[1:] > query_times[:-1]):
            raise ValueError(
                "query_times must be strictly increasing for every sample."
            )

        num_context_points = context_times.size(0)
        if not torch.allclose(query_times[:num_context_points], context_times):
            raise ValueError(
                "query_times must start with the provided context_times."
            )

        hidden = torch.zeros(
            1,
            self.hidden_dim,
            device=context_times.device,
            dtype=context_observations.dtype,
        )
        previous_time = context_times.new_tensor(self.start_time)
        predictions: List[torch.Tensor] = []

        for step in range(num_context_points):
            current_time = context_times[step]
            hidden = self._solve_hidden(hidden, previous_time, current_time)

            if bool(context_mask[step].item()):
                observation = context_observations[step].unsqueeze(0)
                hidden = self.update_cell(observation, hidden)

            predictions.append(self.readout(hidden))
            previous_time = current_time

        for current_time in query_times[num_context_points:]:
            hidden = self._solve_hidden(hidden, previous_time, current_time)
            predictions.append(self.readout(hidden))
            previous_time = current_time

        return torch.cat(predictions, dim=0)

    def forward(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts values at query_times after conditioning on context.

        Args:
            context_times: Tensor of shape [batch, T_ctx].
            context_observations: Tensor of shape [batch, T_ctx, input_dim].
                Masked context positions should already be zero-filled.
            context_mask: Tensor of shape [batch, T_ctx].
            query_times: Tensor of shape [batch, T_query], with the first
                T_ctx entries equal to context_times.

        Returns:
            Tensor of shape [batch, T_query, output_dim].
        """
        self._validate_inputs(
            context_times=context_times,
            context_observations=context_observations,
            context_mask=context_mask,
            query_times=query_times,
        )
        self.reset_nfe()

        batch_predictions: List[torch.Tensor] = []
        for batch_index in range(context_times.size(0)):
            sample_predictions = self._forward_single(
                context_times=context_times[batch_index],
                context_observations=context_observations[batch_index],
                context_mask=context_mask[batch_index],
                query_times=query_times[batch_index],
            )
            batch_predictions.append(sample_predictions.unsqueeze(0))

        self.last_nfe = int(self.hidden_dynamics.nfe)
        return torch.cat(batch_predictions, dim=0)


class GRUTimeSeriesBaseline(nn.Module):
    """GRU baseline consuming value, mask, and time-gap inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        start_time: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.start_time = start_time

        self.update_cell = nn.GRUCell(
            input_size=input_dim + 2,
            hidden_size=hidden_dim,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def get_nfe(self) -> int:
        """Returns zero because the GRU baseline performs no ODE solves."""
        return 0

    def _validate_inputs(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> None:
        if context_times.ndim != 2:
            raise ValueError(
                "context_times must have shape [batch, num_context_points]."
            )
        if context_observations.ndim != 3:
            raise ValueError(
                "context_observations must have shape "
                "[batch, num_context_points, input_dim]."
            )
        if context_mask.ndim != 2:
            raise ValueError(
                "context_mask must have shape [batch, num_context_points]."
            )
        if query_times.ndim != 2:
            raise ValueError(
                "query_times must have shape [batch, num_query_points]."
            )

        if context_times.shape[:2] != context_observations.shape[:2]:
            raise ValueError(
                "context_times and context_observations must agree on batch "
                "and sequence dimensions."
            )
        if context_times.shape != context_mask.shape:
            raise ValueError(
                "context_times and context_mask must have identical shapes."
            )
        if context_observations.size(-1) != self.input_dim:
            raise ValueError(
                "context_observations last dimension does not match input_dim."
            )

    def _build_step_input(
        self,
        observation: torch.Tensor,
        is_observed: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Builds the GRU input [observation, mask, delta_t]."""
        mask_value = observation.new_tensor([float(is_observed.item())])
        delta_value = delta_t.reshape(1).to(dtype=observation.dtype)
        return torch.cat([observation, mask_value, delta_value], dim=0).unsqueeze(0)

    def _forward_single(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Runs GRU inference for a single sample."""
        if not torch.all(context_times[1:] > context_times[:-1]):
            raise ValueError(
                "context_times must be strictly increasing for every sample."
            )
        if not torch.all(query_times[1:] > query_times[:-1]):
            raise ValueError(
                "query_times must be strictly increasing for every sample."
            )

        num_context_points = context_times.size(0)
        if not torch.allclose(query_times[:num_context_points], context_times):
            raise ValueError(
                "query_times must start with the provided context_times."
            )

        hidden = torch.zeros(
            1,
            self.hidden_dim,
            device=context_times.device,
            dtype=context_observations.dtype,
        )
        previous_time = context_times.new_tensor(self.start_time)
        predictions: List[torch.Tensor] = []

        for step in range(num_context_points):
            current_time = context_times[step]
            delta_t = current_time - previous_time
            step_input = self._build_step_input(
                observation=context_observations[step],
                is_observed=context_mask[step],
                delta_t=delta_t,
            )
            hidden = self.update_cell(step_input, hidden)
            predictions.append(self.readout(hidden))
            previous_time = current_time

        zero_observation = context_observations.new_zeros(self.input_dim)
        zero_mask = context_mask.new_tensor(False)

        for current_time in query_times[num_context_points:]:
            delta_t = current_time - previous_time
            step_input = self._build_step_input(
                observation=zero_observation,
                is_observed=zero_mask,
                delta_t=delta_t,
            )
            hidden = self.update_cell(step_input, hidden)
            predictions.append(self.readout(hidden))
            previous_time = current_time

        return torch.cat(predictions, dim=0)

    def forward(
        self,
        context_times: torch.Tensor,
        context_observations: torch.Tensor,
        context_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts values at query_times after conditioning on context."""
        self._validate_inputs(
            context_times=context_times,
            context_observations=context_observations,
            context_mask=context_mask,
            query_times=query_times,
        )

        batch_predictions: List[torch.Tensor] = []
        for batch_index in range(context_times.size(0)):
            sample_predictions = self._forward_single(
                context_times=context_times[batch_index],
                context_observations=context_observations[batch_index],
                context_mask=context_mask[batch_index],
                query_times=query_times[batch_index],
            )
            batch_predictions.append(sample_predictions.unsqueeze(0))

        return torch.cat(batch_predictions, dim=0)