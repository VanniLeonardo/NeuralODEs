import pytest
import torch

from kaby.data.timeseries import IrregularSineWaveDataset
from kaby.models.ode_rnn import GRUTimeSeriesBaseline, ODERNN


def _build_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    dataset = IrregularSineWaveDataset(
        num_samples=batch_size,
        n_context_points=6,
        n_future_points=4,
        min_observed_context_points=2,
        observation_prob=0.6,
        noise_std=0.01,
        seed=123,
    )
    samples = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0].keys()
    }


def test_irregular_sine_dataset_masks_are_consistent() -> None:
    dataset = IrregularSineWaveDataset(
        num_samples=4,
        n_context_points=8,
        n_future_points=5,
        min_observed_context_points=2,
        observation_prob=0.6,
        seed=7,
    )
    sample = dataset[0]

    assert sample["context_mask"].sum().item() >= 2
    assert torch.all(sample["query_times"][1:] > sample["query_times"][:-1])
    assert torch.allclose(sample["query_times"][:8], sample["context_times"])
    assert not sample["future_mask"][:8].any()
    assert sample["future_mask"][8:].all()
    assert sample["interpolation_mask"][:8].any()


def test_ode_rnn_shape_consistency() -> None:
    batch = _build_batch()
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        context_observations=batch["context_observations"],
        context_mask=batch["context_mask"],
        query_times=batch["query_times"],
    )

    assert predictions.shape == batch["query_targets"].shape


def test_gru_baseline_shape_consistency() -> None:
    batch = _build_batch()
    model = GRUTimeSeriesBaseline(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
    )

    predictions = model(
        context_times=batch["context_times"],
        context_observations=batch["context_observations"],
        context_mask=batch["context_mask"],
        query_times=batch["query_times"],
    )

    assert predictions.shape == batch["query_targets"].shape


def test_ode_rnn_nfe_tracking() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    assert model.get_nfe() == 0

    _ = model(
        context_times=batch["context_times"],
        context_observations=batch["context_observations"],
        context_mask=batch["context_mask"],
        query_times=batch["query_times"],
    )

    assert model.get_nfe() > 0


def test_ode_rnn_gradient_flow() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        context_observations=batch["context_observations"],
        context_mask=batch["context_mask"],
        query_times=batch["query_times"],
    )
    loss = predictions.sum()
    loss.backward()

    has_gradients = any(parameter.grad is not None for parameter in model.parameters())
    assert has_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_ode_rnn_device_agnostic() -> None:
    device = torch.device("cuda:0")
    batch = {
        key: value.to(device)
        for key, value in _build_batch(batch_size=2).items()
    }
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    ).to(device)

    predictions = model(
        context_times=batch["context_times"],
        context_observations=batch["context_observations"],
        context_mask=batch["context_mask"],
        query_times=batch["query_times"],
    )

    assert predictions.device == device